"""Chunk-steppable copy of the official LingBot-World-v2 causal-fast inference loop
(wan/image2video.py: WanI2VCausal._generate_causal_fast), single GPU, no offload.
The official repo is imported from `fastworld.paths.LINGBOT_DIR`; weights from LINGBOT_CKPT / WAN21_DIR.

`prepare_inputs` is the official preamble verbatim (noise, mask, VAE image latent, Plücker camera embedding, T5 context,
timesteps). `EagerSession.gen_chunk(k)` is the body of the official chunk loop (4 denoising forwards with the official
model.forward + rolling KV cache, x0 conversion, add_noise, then the clean-latent KV-cache-update forward).
`official_latents` runs the unmodified pipe.generate() and returns its latents (VAE decode stubbed) so the engine can be
checked bitwise against it.
"""
import hashlib, math, os, sys
import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

from fastworld.paths import LINGBOT_CKPT, LINGBOT_DIR, WAN21_DIR

REPO = str(LINGBOT_DIR)      # official clone (unmodified); examples/<ex>/{image.jpg,prompt.txt,poses.npy,intrinsics.npy}
CKPT = str(LINGBOT_CKPT)     # HF robbyant/lingbot-world-v2-1.3b-causal-fast (DiT only)
ASSETS = str(WAN21_DIR)      # T5 encoder, Wan2.1_VAE.pth, google/umt5-xxl tokenizer (assets_dir)
sys.path.insert(0, REPO)

import wan  # noqa: E402
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS  # noqa: E402
from wan.utils.cam_utils import (compute_relative_poses, interpolate_camera_poses, get_plucker_embeddings,  # noqa: E402
                                 get_Ks_transformed)

TASK, SIZE = "i2v-1.3B", "480*832"
CHUNK = 4                    # generate.py --chunk_size default
LOCAL_ATTN, SINK = 18, 6     # README / run_fast.sh
TIMESTEPS_INDEX = [0, 250, 500, 750]   # WanI2VCausal.generate default (passed through to _generate_causal_fast)
SEED = 42


def ex_prompt(ex):
    p = os.path.join(REPO, "examples", ex, "prompt.txt")
    return open(p).read().strip()


def ex_paths(ex):
    """(image path, action dir) of an official example."""
    return os.path.join(REPO, "examples", ex, "image.jpg"), os.path.join(REPO, "examples", ex)


def load_pipe(device_id=0):
    cfg = WAN_CONFIGS[TASK]
    pipe = wan.WanI2VCausal(config=cfg, checkpoint_dir=CKPT, device_id=device_id, rank=0, local_attn_size=LOCAL_ATTN,
                            sink_size=SINK, infer_mode="causal_fast", assets_dir=ASSETS)
    return pipe


class Inputs:
    pass


@torch.no_grad()
def prepare_inputs(pipe, prompt, image_path, action_path, n_chunks, seed=SEED, chunk_size=CHUNK,
                   timesteps_index=TIMESTEPS_INDEX, shift=None, max_area=MAX_AREA_CONFIGS[SIZE]):
    """Verbatim preamble of _generate_causal_fast (offload_model=False, single GPU)."""
    self = pipe
    shift = WAN_CONFIGS[TASK].sample_shift if shift is None else shift
    frame_num = 4 * chunk_size * n_chunks + 1
    img = Image.open(image_path).convert("RGB")
    c2ws = np.load(os.path.join(action_path, "poses.npy"))
    len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
    frame_num = ((frame_num - 1) // 4) * 4 + 1
    frame_num = min(frame_num, len_c2ws)
    c2ws = c2ws[:frame_num]
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
    F = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2])
    h = lat_h * self.vae_stride[1]
    w = lat_w * self.vae_stride[2]
    lat_f = (F - 1) // self.vae_stride[0] + 1
    lat_f = int(lat_f - (lat_f % chunk_size))
    F = (lat_f - 1) * 4 + 1
    max_seq_len = chunk_size * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=self.device)
    msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    self.scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
    timesteps = self.scheduler.timesteps[timesteps_index]
    cache_key = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    if cache_key in self._t5_cache:
        context = self._t5_cache[cache_key]
    else:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([prompt], self.device)
        self.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        self._t5_cache[cache_key] = context
    Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
    Ks = get_Ks_transformed(Ks, height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h,
                            width_final=w)
    Ks = Ks[0]
    len_c2ws = len(c2ws)
    len_c2ws_ = int((len_c2ws - 1) // 4) + 1
    len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
    c2ws_infer = interpolate_camera_poses(src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                                          src_rot_mat=c2ws[:, :3, :3], src_trans_vec=c2ws[:, :3, 3],
                                          tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_))
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
    Ks = Ks.repeat(len(c2ws_infer), 1)
    c2ws_infer = c2ws_infer.to(self.device)
    Ks = Ks.to(self.device)
    c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
    c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)', c1=int(h // lat_h),
                                 c2=int(w // lat_w))
    c2ws_plucker_emb = c2ws_plucker_emb[None, ...]
    c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h,
                                 w=lat_w).to(self.param_dtype)
    y = self.vae.encode([
        torch.concat([torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                      torch.zeros(3, F - 1, h, w)], dim=1).to(self.device)])[0]
    y = torch.concat([msk, y])
    frame_seqlen = int(noise.shape[-2] * noise.shape[-1] // 4)
    I = Inputs()
    I.__dict__.update(prompt=prompt, context=context, noise=noise, y=y, plucker=c2ws_plucker_emb, timesteps=timesteps,
                      g_state=seed_g.get_state(), lat_f=lat_f, lat_h=lat_h, lat_w=lat_w, max_seq_len=max_seq_len,
                      frame_seqlen=frame_seqlen, chunk_size=chunk_size, n_chunks=lat_f // chunk_size,
                      kv_size=frame_seqlen * LOCAL_ATTN if LOCAL_ATTN > -1 else frame_seqlen * lat_f)
    return I


class EagerSession:
    """The official chunk loop, one chunk per call."""

    def __init__(self, pipe, I):
        self.pipe, self.I = pipe, I
        self.reset()

    def reset(self):
        pipe, I = self.pipe, self.I
        cfg = pipe.model.config
        head_dim = cfg.dim // cfg.num_heads
        self.g = torch.Generator(device=pipe.device)
        self.g.set_state(I.g_state)
        self.kv = pipe._initialize_self_kv_cache(num_layers=cfg.num_layers,
                                                 shape=[1, I.kv_size, cfg.num_heads // pipe.sp_size, head_dim],
                                                 dtype=pipe.pipe_dtype, device=pipe.device)
        self.xkv = pipe._initialize_crossattn_cache(num_layers=cfg.num_layers, shape=[1, 512, cfg.num_heads, head_dim],
                                                    dtype=pipe.pipe_dtype, device=pipe.device)
        pipe._cross_attn_initialized = False
        self.lat = I.noise.split(I.chunk_size, dim=1)
        self.cond = I.y.split(I.chunk_size, dim=1)
        self.cam = I.plucker.split(I.chunk_size, dim=2)

    def state_bytes(self):
        ts = [t for d in self.kv + self.xkv for t in d.values() if torch.is_tensor(t)]
        return sum(t.numel() * t.element_size() for t in ts)

    @torch.no_grad()
    def gen_chunk(self, chunk_id, cam=None):
        """Body of the official chunk loop. `cam` optionally overrides this chunk's Plücker embedding (branching)."""
        self_, I = self.pipe, self.I
        timesteps = I.timesteps
        with torch.amp.autocast('cuda', dtype=self_.param_dtype):
            current_latent = self.lat[chunk_id]
            current_condition = self.cond[chunk_id]
            current_c2ws_plucker_emb = self.cam[chunk_id] if cam is None else cam
            dit_cond_dict = {"c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0)}
            kwargs = {
                'context': [I.context[0]], 'seq_len': I.max_seq_len, 'y': [current_condition],
                'dit_cond_dict': dit_cond_dict, 'kv_cache': self.kv, 'crossattn_cache': self.xkv,
                'current_start': chunk_id * I.chunk_size * I.frame_seqlen, 'max_attention_size': I.kv_size,
                'frame_seqlen': I.frame_seqlen,
            }
            for timestep_idx in range(len(timesteps)):
                latent_model_input = [current_latent.to(self_.device)]
                current_timestep = [timesteps[timestep_idx]]
                timestep = torch.stack(current_timestep).to(self_.device)
                noise_pred = self_.model(x=latent_model_input, t=timestep,
                                         cross_attn_first_call=not self_._cross_attn_initialized, **kwargs)[0]
                self_._cross_attn_initialized = True
                x0 = self_._convert_flow_pred_to_x0(flow_pred=noise_pred, xt=current_latent,
                                                    timestep=current_timestep[0], scheduler=self_.scheduler)
                if timestep_idx < len(timesteps) - 1:
                    next_timestep = timesteps[timestep_idx + 1]
                    current_latent = self_.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=self.g, device=x0.device, dtype=x0.dtype), next_timestep)
                else:
                    break
            context_timestep = [timesteps[-1] * 0.0]
            timestep = torch.stack(context_timestep).to(self_.device)
            self_.model(x=[x0], t=timestep, cross_attn_first_call=False, **kwargs)
        return x0


@torch.no_grad()
def official_latents(pipe, prompt, image_path, action_path, n_chunks, seed=SEED):
    """Unmodified pipe.generate() (offload off); VAE decode stubbed to return the latents."""
    dec = pipe.vae.decode
    pipe.vae.decode = lambda zs: [zs[0].clone()]
    try:
        img = Image.open(image_path).convert("RGB")
        lat = pipe.generate(prompt, img, action_path=action_path, chunk_size=CHUNK, max_area=MAX_AREA_CONFIGS[SIZE],
                            frame_num=4 * CHUNK * n_chunks + 1, shift=WAN_CONFIGS[TASK].sample_shift, seed=seed,
                            offload_model=False, max_attention_size=None)
    finally:
        pipe.vae.decode = dec
    return lat

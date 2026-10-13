"""Eager Matrix-Game-3.0 engine: the official single-GPU loop of pipeline/inference_pipeline.py::generate, split into
per-iteration pieces we can time (prep = host work + memory retrieval, denoise = 3 DiT forwards + UniPC, commit, decode).
Every line of the loop body is transcribed from the official generate(); `scripts/mg3/check_official.py` verifies the
latents are bitwise identical to running the official generate() itself.

The official code (MG3_DIR, see fastworld.paths) is put on sys.path here; it also reads `assets/...` relative to its
own directory, so run from MG3_DIR (scripts/mg3/job.sh does this).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from fastworld.paths import MG3_CKPT, MG3_DIR, ROOT

REPO = str(MG3_DIR)
CKPT = str(MG3_CKPT)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

IMAGE = os.path.join(REPO, "demo_images/001/image.png")
PROMPT = open(os.path.join(REPO, "demo_images/001/prompt.txt")).read().strip()

CLIP, FIRST, PAST = 56, 57, 16


def make_args(**kw):
    a = dict(size="704*1280", ckpt_dir=CKPT, ulysses_size=1, t5_fsdp=False, t5_cpu=False, dit_fsdp=False,
             prompt=PROMPT, seed=42, image=IMAGE, num_iterations=8, convert_model_dtype=False,
             output_dir=str(ROOT / "results" / "mg3" / "output"), save_name="eager", sample_shift=5.0, sample_guide_scale=5.0,
             num_inference_steps=3, lightvae_pruning_rate=None, use_async_vae=False, async_vae_warmup_iters=0,
             compile_vae=False, vae_type="mg_lightvae_v2", use_int8=True, verify_quant=False, fa_version="3",
             interactive=False, use_base_model=False)
    a.update(kw)
    return argparse.Namespace(**a)


class Engine:
    def __init__(self, **kw):
        from wan.configs import WAN_CONFIGS
        from pipeline.inference_pipeline import MatrixGame3Pipeline
        self.args = make_args(**kw)
        self.cfg = WAN_CONFIGS["matrix_game3"]
        self.p = MatrixGame3Pipeline(config=self.cfg, checkpoint_dir=CKPT, device_id=0, rank=0, args=self.args,
                                     fa_version=self.args.fa_version, use_base_model=False)
        self.model, self.vae, self.device = self.p.model, self.p.vae, self.p.device
        self._ctx_cache = {}

    def free_t5(self):
        """Prompt embeddings are cached per prompt; the 11 GB T5 can then leave the GPU."""
        self.p.text_encoder.model.to("cpu")
        torch.cuda.empty_cache()

    def encode_prompt(self, text):
        if text not in self._ctx_cache:
            self._ctx_cache[text] = self.p.text_encoder([text], device=self.device)
        return self._ctx_cache[text]

    # ------------------------------------------------------------------ session
    def start(self, seed=None, num_iterations=None, image=None, prompt=None):
        """Everything generate() does before the iteration loop. Returns nothing; state lives on self."""
        from PIL import Image
        from utils.misc import set_seed
        from utils.utils import get_data
        from utils.cam_utils import get_intrinsics
        a = self.args
        seed = a.seed if seed is None else seed
        self.n_iter = a.num_iterations if num_iterations is None else num_iterations
        set_seed(seed)   # generate.py seeds python/numpy/torch; the random action trajectory comes from python `random`
        pil_image = Image.open(image or a.image).convert("RGB")
        text = prompt or a.prompt
        p = self.p
        height, width = int(a.size.split("*")[0]), int(a.size.split("*")[1])
        self.height, self.width = height, width
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        num_frames = FIRST + (self.n_iter - 1) * 40
        current_image, self.extrinsics_all, self.kb_all, self.ms_all = get_data(
            num_frames, height, width, pil_image, device=self.device, dtype=torch.bfloat16)
        self.cond = self.encode_prompt(text)
        max_area = 704 * 1280
        aspect_ratio = current_image.shape[-2] / current_image.shape[-1]
        self.lat_h = round(np.sqrt(max_area * aspect_ratio) // p.vae_stride[1] // p.patch_size[1] * p.patch_size[1])
        self.lat_w = round(np.sqrt(max_area / aspect_ratio) // p.vae_stride[2] // p.patch_size[2] * p.patch_size[2])
        self.target_h, self.target_w = self.lat_h * p.vae_stride[1], self.lat_w * p.vae_stride[2]
        self.base_K = get_intrinsics(self.target_h, self.target_w)
        self.img_cond = self.vae.encode([current_image[0]])[0].unsqueeze(0).to(device=self.device, dtype=torch.bfloat16).contiguous()
        max_lat_f = (FIRST - 1) // p.vae_stride[0] + 1
        self.max_seq_len = (max_lat_f + 5) * self.lat_h * self.lat_w // (p.patch_size[1] * p.patch_size[2])
        self.all_latents = []
        self.vae_cache = [None for _ in range(32)]
        self.k = 0

    # ------------------------------------------------------------------ one iteration
    def prep(self, k):
        """Host side of iteration k: frame ranges, FOV memory retrieval, plucker embeddings, conditions, noise."""
        from utils.cam_utils import compute_relative_poses, select_memory_idx_fov, _interpolate_camera_poses_handedness
        from utils.utils import build_plucker_from_c2ws, build_plucker_from_pose
        dev, wdt = self.device, torch.bfloat16
        first_clip = k == 0
        ext = self.extrinsics_all
        align = lambda f: (f - 1) // 4 * 4 + 1 if f > 0 else 1
        lidx = lambda f: (f - 1) // 4 + 1
        end = FIRST if first_clip else FIRST + k * (CLIP - PAST)
        start = 0 if first_clip else end - CLIP
        c2ws_chunk = ext[start:end]
        src_indices = np.linspace(start, end - 1, FIRST if first_clip else CLIP)
        tgt_len = (FIRST - 1) // 4 + 1 if first_clip else (CLIP // 4)
        tgt_indices = np.linspace(0 if first_clip else start + 3, end - 1, tgt_len)
        plucker = build_plucker_from_c2ws(c2ws_chunk.to(device=dev), src_indices, tgt_indices, framewise=True,
                                          base_K=self.base_K, target_h=self.target_h, target_w=self.target_w,
                                          lat_h=self.lat_h, lat_w=self.lat_w)
        t_ret = 0.0
        if first_clip:
            x_memory = mm = mk = latent_idx = timestep_memory = None
            selected_index = None
        else:
            t0 = time.perf_counter()
            base = [end - o for o in range(1, 34, 8)]
            selected_index = select_memory_idx_fov(ext, start, base, use_gpu=True)
            selected_index[-1] = 4
            t_ret = time.perf_counter() - t0
            memory_pluckers, latent_idx = [], []
            for mem_idx, reference_idx in zip(selected_index, base):
                latent_idx.append(lidx(mem_idx))
                a_ = align(mem_idx)
                mem_block = ext[a_:a_ + 4]
                mem_pose = _interpolate_camera_poses_handedness(
                    src_indices=np.linspace(a_, a_ + 3, mem_block.shape[0]),
                    src_rot_mat=mem_block[:, :3, :3].cpu().numpy(), src_trans_vec=mem_block[:, :3, 3].cpu().numpy(),
                    tgt_indices=np.array([a_ + 3], dtype=np.float32))
                rel_pair = torch.cat([ext[reference_idx:reference_idx + 1], mem_pose], dim=0)
                rel_pose = compute_relative_poses(rel_pair, framewise=False)[1:2]
                memory_pluckers.append(build_plucker_from_pose(rel_pose.to(device=dev), base_K=self.base_K,
                                                               target_h=self.target_h, target_w=self.target_w,
                                                               lat_h=self.lat_h, lat_w=self.lat_w))
            plucker = torch.cat(memory_pluckers + [plucker], dim=2)
            x_memory = torch.cat(self.all_latents, dim=2)[:, :, latent_idx]
            mm = torch.ones((1, len(selected_index), 2)).to(device=dev, dtype=wdt)
            mk = -torch.ones((1, len(selected_index), 6)).to(device=dev, dtype=wdt)
            timestep_memory = x_memory.new_zeros((1, x_memory.shape[2] * x_memory.shape[3] * x_memory.shape[4] // 4))
        ls, le = lidx(start), lidx(end)
        latents = torch.randn((1, 48, le - ls, self.img_cond.shape[-2], self.img_cond.shape[-1]),
                              generator=self.generator, device=dev, dtype=wdt)
        latents = torch.cat([self.img_cond, latents[:, :, self.img_cond.shape[2]:]], dim=2)
        return dict(k=k, first=first_clip, latents=latents, img_cond=self.img_cond, t_retrieval=t_ret,
                    selected_index=selected_index,
                    conds=dict(mouse_cond=self.ms_all[:, start:end], keyboard_cond=self.kb_all[:, start:end],
                               context=self.cond, plucker_emb=plucker.to(device=dev, dtype=wdt), x_memory=x_memory,
                               timestep_memory=timestep_memory, keyboard_cond_memory=mk, mouse_cond_memory=mm,
                               memory_latent_idx=latent_idx, predict_latent_idx=(ls, le),
                               fa_version=self.p.fa_version))

    def denoise(self, it):
        """The official 3-step distilled denoising loop (no CFG)."""
        from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
        sch = FlowUniPCMultistepScheduler()
        sch.set_timesteps(self.args.num_inference_steps, device=self.device, shift=self.args.sample_shift)
        latents, img_cond = it["latents"], it["img_cond"]
        for t in sch.timesteps:
            timestep = latents.new_full((latents.shape[2], latents.shape[3] * latents.shape[4] // 4), t)
            timestep[:img_cond.shape[2]].zero_()
            timestep = timestep.flatten().unsqueeze(0)
            noise_pred = self.model(x=latents, t=timestep, seq_len=self.max_seq_len, **it["conds"])
            latents = sch.step(noise_pred, t, latents, return_dict=False)[0]
            latents = torch.cat([img_cond, latents[:, :, img_cond.shape[2]:]], dim=2)
        return latents

    def commit(self, it, latents):
        self.img_cond = latents[:, :, -4:]
        den = latents if it["first"] else latents[:, :, -10:]
        self.all_latents.append(den)
        self.k = it["k"] + 1
        return den

    def decode(self, den, first):
        video, self.vae_cache = self.vae.stream_decode(den.to(dtype=self.vae.dtype), self.vae_cache, first_chunk=first,
                                                       segment_size=int(os.environ.get("WAN_VAE_SEGMENT_SIZE", "4")),
                                                       profiler={}, compile_decoder=False)
        return video

    @torch.no_grad()
    def step(self, decode=False):
        it = self.prep(self.k)
        lat = self.denoise(it)
        den = self.commit(it, lat)
        if decode:
            self.decode(den, it["first"])
        return den

    def state_bytes(self):
        """Session state carried between iterations: latent history (memory pool) + img_cond (+ VAE stream cache)."""
        n = sum(t.numel() * t.element_size() for t in self.all_latents) + self.img_cond.numel() * self.img_cond.element_size()
        v = sum(t.numel() * t.element_size() for t in self.vae_cache if torch.is_tensor(t))
        return n, v

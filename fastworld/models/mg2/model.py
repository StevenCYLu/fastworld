"""Matrix-Game-2 eager wrapper: the official pipeline, driven one chunk at a time.

Action-conditioned causal world model (Wan-1.3B based, 3-step distilled, 3 latent frames per chunk, 880 tokens per
frame, 352x640, rolling attention window of 6 frames = exactly one chunk of history). A request = start image + one
(keyboard, mouse) action per chunk. Chunk k depends only on the image and actions 0..k (its noise is seeded by a hash
of that path), so requests sharing an action prefix produce bitwise-identical chunks.

`Model` is the reference everything else is checked against: it calls the unmodified official generator with the
official KV caches, exactly as `inference_streaming.py` does, but exposes gen_chunk / prefill / reset so the
static runtime, the tests and the prefix cache can drive it.
"""
import hashlib, os, sys
from fastworld.paths import MG2_CKPT, MG2_DIR
MG = str(MG2_DIR)
sys.path.insert(0, MG)
import torch

FPB, TOK = 3, 880          # latent frames per chunk, tokens per latent frame
L = FPB * TOK
CKPT = str(MG2_CKPT)
KEYS = {"w": [1, 0, 0, 0], "s": [0, 1, 0, 0], "a": [0, 0, 1, 0], "d": [0, 0, 0, 1], "q": [0, 0, 0, 0]}
CAM = {"i": [0.1, 0], "k": [-0.1, 0], "j": [0, -0.1], "l": [0, 0.1], "u": [0, 0]}


def path_seed(path):
    return int.from_bytes(hashlib.sha1(repr(tuple(path)).encode()).digest()[:8], "little") & ((1 << 63) - 1)


class Model:
    def __init__(self, dev="cuda", max_chunks=24):
        cwd = os.getcwd(); os.chdir(MG)
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from pipeline import CausalInferencePipeline
        from utils.wan_wrapper import WanDiffusionWrapper
        from wan.vae.wanx_vae import get_wanx_vae_wrapper
        cfg = OmegaConf.load("configs/inference_yaml/inference_universal.yaml")
        cfg.pop("mode")
        gen = WanDiffusionWrapper(**getattr(cfg, "model_kwargs", {}), is_causal=True)
        p = CausalInferencePipeline(cfg, generator=gen, vae_decoder=None)
        p.generator.load_state_dict(load_file(f"{CKPT}/base_distilled_model/base_distill.safetensors"))
        vae = get_wanx_vae_wrapper(CKPT, torch.float16)
        vae.requires_grad_(False); vae.eval()
        self.vae = vae.to(dev, torch.bfloat16)
        os.chdir(cwd)
        self.p = p.to(device=dev, dtype=torch.bfloat16).eval()
        bf = torch.bfloat16
        p._initialize_kv_cache(1, bf, dev); p._initialize_kv_cache_mouse_and_keyboard(1, bf, dev); p._initialize_crossattn_cache(1, bf, dev)
        self.dev, self.steps, self.max_chunks = dev, p.denoising_step_list, max_chunks
        self.images = {}

    @torch.no_grad()
    def add_image(self, name, path):
        from diffusers.utils import load_image
        from torchvision.transforms import v2
        img = load_image(path)
        w, h = img.size; th, tw = 352, 640
        if h / w > th / tw: nw, nh = w, int(w * th / tw)
        else: nh, nw = h, int(h * tw / th)
        img = img.crop(((w - nw) / 2, (h - nh) / 2, (w + nw) / 2, (h + nh) / 2))
        tf = v2.Compose([v2.Resize(size=(352, 640), antialias=True), v2.ToTensor(), v2.Normalize([0.5] * 3, [0.5] * 3)])
        x = tf(img)[None, :, None].to(self.dev, torch.bfloat16)
        nf = FPB * self.max_chunks
        pad = torch.zeros_like(x).repeat(1, 1, 4 * (nf - 1), 1, 1)
        lat = self.vae.encode(torch.cat([x, pad], 2), device=self.dev, tiled=True, tile_size=[44, 80], tile_stride=[23, 38]).to(self.dev)
        mask = torch.ones_like(lat); mask[:, :, 1:] = 0
        self.images[name] = dict(cond_concat=torch.cat([mask[:, :4], lat], 1).to(torch.bfloat16),
                                 visual_context=self.vae.clip.encode_video(x).to(torch.bfloat16))

    def cond(self, path, k):
        """conditional dict for chunk k: image conditioning + actions of chunks 0..k (pixel-frame resolution)."""
        img = self.images[path[0][0]]
        n_px = 1 + 4 * (FPB * (k + 1) - 1)
        kb = torch.zeros(1, n_px, 4, device=self.dev, dtype=torch.bfloat16)
        ms = torch.zeros(1, n_px, 2, device=self.dev, dtype=torch.bfloat16)
        for j in range(k + 1):
            a, b = (0 if j == 0 else 1 + 4 * (FPB * j - 1)), 1 + 4 * (FPB * (j + 1) - 1)
            kb[:, a:b] = torch.tensor(KEYS[path[j][1][0]], dtype=torch.bfloat16)
            ms[:, a:b] = torch.tensor(CAM[path[j][1][1]], dtype=torch.bfloat16)
        s = FPB * k
        return dict(cond_concat=img["cond_concat"][:, :, s:s + FPB], visual_context=img["visual_context"], mouse_cond=ms, keyboard_cond=kb)

    def caches(self):
        p = self.p
        return [("m", p.kv_cache1), ("ms", p.kv_cache_mouse), ("kb", p.kv_cache_keyboard)]

    def reset(self):
        for _, cs in self.caches():
            for c in cs:
                c["global_end_index"].zero_(); c["local_end_index"].zero_()

    def _fwd(self, x, path, k, t):
        for c in self.p.crossattn_cache:
            c["is_init"] = False
        ts = torch.full((1, FPB), int(t), device=self.dev, dtype=torch.int64)
        return self.p.generator(noisy_image_or_video=x, conditional_dict=self.cond(path, k), timestep=ts, kv_cache=self.p.kv_cache1,
                                kv_cache_mouse=self.p.kv_cache_mouse, kv_cache_keyboard=self.p.kv_cache_keyboard,
                                crossattn_cache=self.p.crossattn_cache, current_start=k * L)[1]

    @torch.no_grad()
    def gen_chunk(self, path, k):
        g = torch.Generator(device=self.dev).manual_seed(path_seed(path[:k + 1]))
        x = torch.randn((1, 16, FPB, 44, 80), generator=g, device=self.dev, dtype=torch.bfloat16)
        for i, t in enumerate(self.steps):
            den = self._fwd(x, path, k, t)
            if i < len(self.steps) - 1:
                d = den.transpose(1, 2).flatten(0, 1)
                x = self.p.scheduler.add_noise(d, torch.randn(d.shape, generator=g, device=self.dev, dtype=d.dtype),
                                               self.steps[i + 1] * torch.ones(FPB, device=self.dev, dtype=torch.long))
                x = x.unflatten(0, (1, FPB)).transpose(1, 2)
        self._fwd(den, path, k, 0)
        return den

    @torch.no_grad()
    def prefill(self, lat, path, k):
        self._fwd(lat, path, k, 0)

    def state_bytes(self):
        return sum(c[k].numel() * c[k].element_size() for _, cs in self.caches() for c in cs for k in ("k", "v"))

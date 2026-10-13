"""LongLive-2.0-5B eager wrapper: the OFFICIAL pipeline (BF16, `configs/inference.yaml` minus the LoRA adapter section,
merged `model_bf16.pt`), a fixed prompt schedule with scene cuts, fixed noise, and per-chunk timing hooks around the
official `CausalDiffusionInferencePipeline.inference` loop.

Locations come from `fastworld.paths`: LONGLIVE_DIR (the NVlabs/LongLive clone; the code is in its `LongLive2.0`
subdir), LONGLIVE_CKPT (dir holding `model_bf16.pt`), WAN22_DIR (Wan2.2-TI2V-5B assets: `config.json`, `Wan2.2_VAE.pth`,
`models_t5_umt5-xxl-enc-bf16.pth`, `google/umt5-xxl/`). The official loaders read
`wan_models/Wan2.2-TI2V-5B/...` relative to the working directory, so `official_cwd()` chdirs into a throwaway
directory holding only that symlink -> WAN22_DIR while a model is built.
"""
import contextlib, os, shutil, sys, tempfile, time
from fastworld.paths import LONGLIVE_CKPT, LONGLIVE_DIR, WAN22_DIR
LL = str(LONGLIVE_DIR / "LongLive2.0")
if LL not in sys.path:
    sys.path.insert(0, LL)
import torch

CKPT = str(LONGLIVE_CKPT / "model_bf16.pt")
WAN_NAME = "Wan2.2-TI2V-5B"   # model_kwargs.model_name in configs/inference.yaml -> the official code opens wan_models/<name>/
FPB = 8                 # latent frames per chunk
TOK = 880               # tokens per latent frame (44x80 latent, 2x2 patch)
CUT = "The scene transitions. "   # utils.dataset.DEFAULT_SCENE_CUT_PREFIX

PROMPTS = [
    "A compact silver robot walks through a clean robotics lab, passing workbenches covered with tools and glowing monitors.",
    "A red fox trots through a snowy pine forest at dawn, its breath visible in the cold air, soft golden light between the trees.",
    "A sailboat glides across a calm turquoise bay under a clear sky, gentle waves sparkling, a rocky coastline in the background.",
]


@contextlib.contextmanager
def official_cwd():
    """Run the official loaders with `wan_models/Wan2.2-TI2V-5B` resolving to WAN22_DIR (a temp dir with one symlink)."""
    tmp = tempfile.mkdtemp(prefix="fastworld-longlive2-")
    os.makedirs(os.path.join(tmp, "wan_models"))
    os.symlink(str(WAN22_DIR), os.path.join(tmp, "wan_models", WAN_NAME))
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        yield tmp
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def prompt_schedule(n_chunks, shot_len=4):
    """Per-chunk prompts in the dataset convention: the first chunk of every new shot carries the scene-cut prefix."""
    out = []
    for k in range(n_chunks):
        s = (k // shot_len) % len(PROMPTS)
        out.append((CUT if (k > 0 and k % shot_len == 0) else "") + PROMPTS[s])
    return out


def _patch_from_pretrained():
    """The released checkpoint is a full merged generator, so the 20 GB base Wan2.2 DiT weights are not needed:
    build CausalWanModel from config.json only, then load model_bf16.pt (strict)."""
    from wan_5b.modules.causal_model import CausalWanModel

    def fp(cls, path, **kw):
        return cls.from_config(cls.load_config(path), **kw)
    CausalWanModel.from_pretrained = classmethod(fp)


def build_pipeline(device="cuda", shot_clean_recache=False, compile_cfg=None):
    from omegaconf import OmegaConf
    from utils.config import normalize_config
    _patch_from_pretrained()
    cfg = OmegaConf.load(os.path.join(LL, "configs", "inference.yaml"))
    cfg.pop("adapter")                       # merged checkpoint: no LoRA (as the model card instructs)
    cfg.checkpoints.pop("lora_ckpt")
    cfg.checkpoints.generator_ckpt = CKPT
    cfg.inference.vae_device = None
    cfg.inference.shot_clean_recache = bool(shot_clean_recache)
    cfg = normalize_config(cfg)
    from pipeline import CausalDiffusionInferencePipeline
    from utils.inference_utils import load_generator_checkpoint
    torch.set_grad_enabled(False)
    with official_cwd():
        pipe = CausalDiffusionInferencePipeline(cfg, device=device)
        load_generator_checkpoint(pipe.generator, CKPT)
    pipe = pipe.to(device=device, dtype=torch.bfloat16)
    pipe.generator.model.eval().requires_grad_(False)
    if compile_cfg is not None:
        pipe.generator.configure_torch_compile(**compile_cfg)
    return pipe, cfg


def build_vae(device="cuda"):
    """The official Wan2.2 VAE wrapper (`decode_to_pixel`), loaded from WAN22_DIR."""
    from utils.wan_5b_wrapper import WanVAEWrapper
    with official_cwd():
        vae = WanVAEWrapper()
    return vae.to(device=device, dtype=torch.bfloat16)


def make_noise(n_chunks, seed=0, device="cuda"):
    from utils.misc import set_seed
    set_seed(seed)
    return torch.randn([1, n_chunks * FPB, 48, 44, 80], device=device, dtype=torch.bfloat16)


def encode_prompts(pipe, prompts):
    """Same call the official pipeline makes at the top of inference() (all chunk prompts in one T5 batch)."""
    from utils.prompt_conditioning import encode_prompt_blocks
    _, lst = encode_prompt_blocks(pipe.text_encoder, [prompts], 1)
    return lst


class ChunkTimer:
    """Wraps pipe._initialize_sample_scheduler (called exactly once at the top of every chunk of the official loop) to
    record a synchronized timestamp per chunk, and optionally step a profiler at chunk boundaries."""

    def __init__(self, pipe, prof=None):
        self.pipe, self.prof, self.ts = pipe, prof, []
        self._orig = pipe._initialize_sample_scheduler

        def hook(noise):
            torch.cuda.synchronize()
            self.ts.append(time.perf_counter())
            if self.prof is not None and len(self.ts) > 1:
                self.prof.step()
            return self._orig(noise)
        pipe._initialize_sample_scheduler = hook

    def reset(self):
        self.ts = []

    def finish(self):
        torch.cuda.synchronize()
        self.ts.append(time.perf_counter())
        return [b - a for a, b in zip(self.ts[:-1], self.ts[1:])]

    def remove(self):
        self.pipe._initialize_sample_scheduler = self._orig


def run_official(pipe, noise, prompts, timer=None):
    """One call of the official inference loop; returns latents [1, F, 48, 44, 80] and per-chunk wall times."""
    if timer is not None:
        timer.reset()
    with torch.inference_mode():
        lat = pipe.inference(noise=noise, text_prompts=[prompts], return_latents=True)
    times = timer.finish() if timer is not None else None
    return lat, times


def shrink_neg_cache(pipe):
    """guidance_scale=1.0 -> the official pipeline never reads kv_cache_neg / crossattn_cache_neg data, but allocates
    them (10.4 GB). Free their payload (keeps index tensors so the reset path still works)."""
    if pipe.kv_cache_neg is None:
        return
    for c in pipe.kv_cache_neg:
        c["k"] = c["k"].new_zeros(1); c["v"] = c["v"].new_zeros(1)
    for c in pipe.crossattn_cache_neg:
        c["k"] = c["k"].new_zeros(1); c["v"] = c["v"].new_zeros(1)
    torch.cuda.empty_cache()


class CachedTextEncoder(torch.nn.Module):
    """Runs the real umT5 once for a fixed prompt list, then serves the identical embeddings (frees ~11 GB of GPU)."""

    def __init__(self, enc, prompts):
        super().__init__()
        self.out = enc(text_prompts=list(prompts))["prompt_embeds"]
        self.prompts = list(prompts)

    def forward(self, text_prompts):
        assert list(text_prompts) == self.prompts, "CachedTextEncoder: unexpected prompts"
        return {"prompt_embeds": self.out}


def init_caches(pipe, device="cuda"):
    pipe._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=device)
    pipe._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=device)


def eager_state_bytes(pipe, include_neg=True):
    tot = 0
    caches = [pipe.kv_cache_pos, pipe.crossattn_cache_pos] + ([pipe.kv_cache_neg, pipe.crossattn_cache_neg] if include_neg else [])
    for cache in caches:
        for c in cache or []:
            for k in ("k", "v"):
                tot += c[k].numel() * c[k].element_size()
    return tot

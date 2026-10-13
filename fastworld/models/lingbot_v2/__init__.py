"""LingBot-World-v2-1.3B-causal-fast (Robbyant; Wan2.1-1.3B I2V DiT + camera injector, 4 latent frames per chunk,
4 distilled steps, 18-frame rolling KV window with a 6-frame sink)."""
from fastworld.models.lingbot_v2.engine import (ASSETS, CHUNK, CKPT, LOCAL_ATTN, REPO, SEED, SINK, TIMESTEPS_INDEX,
                                                EagerSession, Inputs, ex_paths, ex_prompt, load_pipe,
                                                official_latents, prepare_inputs)
from fastworld.models.lingbot_v2.static import StaticLingbot, eager_plans

__all__ = ["ASSETS", "CHUNK", "CKPT", "LOCAL_ATTN", "REPO", "SEED", "SINK", "TIMESTEPS_INDEX", "EagerSession",
           "Inputs", "StaticLingbot", "eager_plans", "ex_paths", "ex_prompt", "load_pipe", "official_latents",
           "prepare_inputs"]

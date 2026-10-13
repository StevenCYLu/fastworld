"""Matrix-Game-3.0 (Skywork, distilled 5B Wan2.2-TI2V DiT, 704x1280, 3 steps, FOV memory retrieval, no KV cache)."""
from fastworld.models.mg3.engine import CKPT, CLIP, FIRST, IMAGE, PAST, PROMPT, REPO, Engine, make_args
from fastworld.models.mg3.static import CachedUniPC, StaticMG3

__all__ = ["CKPT", "CLIP", "CachedUniPC", "Engine", "FIRST", "IMAGE", "PAST", "PROMPT", "REPO", "StaticMG3", "make_args"]

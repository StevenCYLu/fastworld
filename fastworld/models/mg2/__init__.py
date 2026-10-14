"""Matrix-Game-2 (Skywork, 1.3B causal Wan DiT, 352x640, 3 distilled steps, 3 latent frames per chunk)."""
from fastworld.models.mg2.model import CAM, FPB, KEYS, L, MG, TOK, Model, path_seed
from fastworld.models.mg2.static import StaticMG
from fastworld.models.mg2.vae import StaticVAE, TinyVAE
from fastworld.models.mg2.workloads import ACTS, SEG, explore_paths, rewind_paths, test_path

__all__ = ["ACTS", "CAM", "FPB", "KEYS", "L", "MG", "Model", "SEG", "StaticMG", "StaticVAE", "TinyVAE", "TOK",
           "explore_paths", "path_seed", "rewind_paths", "test_path"]

"""Where the official model code and weights live. Every location can be overridden with an environment variable, so
the repo works both with the default layout (`third_party/<repo>`, `weights/<model>` under the repo root) and with
pre-existing checkouts elsewhere (point the variable at them, or symlink into third_party/).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _p(env, default):
    return Path(os.environ.get(env, default)).expanduser()


THIRD_PARTY = _p("FASTWORLD_THIRD_PARTY", ROOT / "third_party")
WEIGHTS = _p("FASTWORLD_WEIGHTS", ROOT / "weights")

# Matrix-Game-2 (https://github.com/SkyworkAI/Matrix-Game, Matrix-Game-2 subdir), weights in <MG2_DIR>/ckpt as in its README
MG2_DIR = _p("FASTWORLD_MG2_DIR", THIRD_PARTY / "Matrix-Game-2")
MG2_CKPT = _p("FASTWORLD_MG2_CKPT", MG2_DIR / "ckpt")
# TAEHV tiny video decoders (https://github.com/madebyollin/taehv), used only by the optional lossy decoder
TAEHV_DIR = _p("FASTWORLD_TAEHV_DIR", THIRD_PARTY / "taehv")

# ---- ports (see docs/models/*.md for what to clone / download into these)
# LingBot-World-v2 (https://github.com/Robbyant/lingbot-world-v2); DiT weights from HF robbyant/lingbot-world-v2-1.3b-causal-fast;
# T5 + Wan2.1 VAE + tokenizer from a Wan2.1-T2V-1.3B checkout
LINGBOT_DIR = _p("FASTWORLD_LINGBOT_DIR", THIRD_PARTY / "lingbot-world-v2")
LINGBOT_CKPT = _p("FASTWORLD_LINGBOT_CKPT", WEIGHTS / "lingbot-world-v2-1.3b-causal-fast")
WAN21_DIR = _p("FASTWORLD_WAN21_DIR", WEIGHTS / "Wan2.1-T2V-1.3B")
# LongLive 2.0 (https://github.com/NVlabs/LongLive, LongLive2.0 subdir); HF Efficient-Large-Model/LongLive-2.0-5B + Wan2.2-TI2V-5B assets
LONGLIVE_DIR = _p("FASTWORLD_LONGLIVE_DIR", THIRD_PARTY / "LongLive")
LONGLIVE_CKPT = _p("FASTWORLD_LONGLIVE_CKPT", WEIGHTS / "LongLive-2.0-5B")
WAN22_DIR = _p("FASTWORLD_WAN22_DIR", WEIGHTS / "Wan2.2-TI2V-5B")
# Matrix-Game-3.0 (https://github.com/SkyworkAI/Matrix-Game, Matrix-Game-3 subdir); HF Skywork/Matrix-Game-3.0
MG3_DIR = _p("FASTWORLD_MG3_DIR", THIRD_PARTY / "Matrix-Game-3")
MG3_CKPT = _p("FASTWORLD_MG3_CKPT", WEIGHTS / "Matrix-Game-3.0")

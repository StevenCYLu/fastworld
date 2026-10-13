"""DejaVu fonts for the GIF builders, with a search over common locations."""
import glob
from PIL import ImageFont

_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", "/Library/Fonts", "C:/Windows/Fonts"]


def _find(name):
    for d in _DIRS:
        hits = glob.glob(f"{d}/**/{name}", recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name}: install the DejaVu fonts (e.g. apt install fonts-dejavu) or edit demo/fonts.py")


FONT, FONTB = _find("DejaVuSans.ttf"), _find("DejaVuSans-Bold.ttf")
MONO, MONOB = _find("DejaVuSansMono.ttf"), _find("DejaVuSansMono-Bold.ttf")

"""Action trajectories used by the tests, benchmarks and demos. A chunk key is (image_name, (keyboard, mouse)) with
keyboard in w/s/a/d/q (q = none) and mouse in i/k/j/l/u (up/down/left/right/none), see model.KEYS / model.CAM."""

# 13 varied actions for the exactness tests
ACTS = [("q", "u"), ("w", "u"), ("w", "j"), ("a", "l"), ("q", "i"), ("d", "k"), ("s", "u"), ("w", "l"), ("q", "j"),
        ("a", "u"), ("w", "i"), ("d", "j"), ("q", "l")]
# segments held for several chunks by the serving workloads
SEG = [("w", "u"), ("w", "j"), ("w", "l"), ("a", "u"), ("d", "u"), ("s", "u"), ("w", "i"), ("q", "l")]


def test_path(img, chunks, offset=0):
    return [(img, ACTS[(k + offset) % len(ACTS)]) for k in range(chunks)]


def explore_paths(img, depth=16, seg=4, branch=2):
    """`depth` chunks, one action per `seg`-chunk segment, branch factor `branch` at every segment boundary after the
    first: 2**(depth/seg - 1) players exploring from the same start image."""
    paths = [[]]
    for lvl in range(depth // seg):
        new = []
        for p in paths:
            choices = [SEG[0]] if lvl == 0 else [SEG[(lvl * branch + b) % len(SEG)] for b in range(branch)]
            for a in choices:
                new.append(p + [(img, a)] * seg)
        paths = new
    return paths


def rewind_paths(img, depth=16, seg=4):
    """One `depth`-chunk playthrough, then 'rewind to chunk t and try something else for 4 chunks' for
    t = 4, 6, ..., depth-2."""
    main = [(img, SEG[(k // seg) % len(SEG)]) for k in range(depth)]
    out = [main]
    for i, t in enumerate(range(4, depth, 2)):
        alt = SEG[(i + 3) % len(SEG)]
        if alt == main[t][1]:
            alt = SEG[(i + 4) % len(SEG)]
        out.append(main[:t] + [(img, alt)] * 4)
    return out

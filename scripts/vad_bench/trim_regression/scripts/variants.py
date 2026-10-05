"""Variant registry: name -> dict (det -> v) or a plain v for all dets. v = None means the old cuts (--vad-trim 0)."""
def sym(x): return {"pre": x, "post": x}
V = {"T0": None}
for t in (0.1, 0.2, 0.25, 0.3, 0.35, 0.5, 1.0): V[f"T{t}"] = sym(t)
def get(name, det):
    v = V[name]
    return v[det] if isinstance(v, dict) and det in v else v
# null controls (grid-aligned and unaligned perturbations of the default), asymmetric pads, min removed length, min length after trim
for t in (0.32, 0.4): V[f"T{t}"] = sym(t)
for a, b in ((0.5, 0.3), (0.6, 0.3), (0.4, 0.2), (0.3, 0.5), (0.5, 0.2)): V[f"P{a}/{b}"] = {"pre": a, "post": b}
for x in (0.5, 1.0, 2.0): V[f"E{x}"] = {"pre": 0.3, "post": 0.3, "min_edge": x}
for x in (0.5, 1.0): V[f"E{x}p0.5"] = {"pre": 0.5, "post": 0.5, "min_edge": x}
for x in (3.0, 6.0): V[f"L{x:g}"] = {"pre": 0.3, "post": 0.3, "minlen": x}
for a, b in ((0.5, 0.4), (0.4, 0.3), (0.4, 0.4), (0.6, 0.4)): V[f"P{a}/{b}"] = {"pre": a, "post": b}

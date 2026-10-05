"""Python replica of pk::segment_by_vad (master 2de154c), generalised with candidate trim rules.
Variant dict keys: pre, post (seconds kept before first / after last speech frame; None = no trim),
minlen (after trimming extend symmetrically, within the cut, to at least this long),
min_edge (trim an edge only if the audio removed there is at least this long)."""
import math
import numpy as np

def mask(p, n, o):
    fs = o["frame_sec"]
    sp = np.zeros(n, dtype=bool); k = min(n, len(p)); sp[:k] = p[:k] >= o["threshold"]
    def runs(val):
        out = []; f = 0
        while f < n:
            if sp[f] != val: f += 1; continue
            e = f
            while e < n and sp[e] == val: e += 1
            out.append((f, e)); f = e
        return out
    for a, b in runs(False):
        if a > 0 and b < n and (b - a) * fs + 1e-9 < o["bridge_sec"]: sp[a:b] = True
    for a, b in runs(True):
        if (b - a) * fs + 1e-9 < o["min_speech_sec"]: sp[a:b] = False
    return sp, runs

def segments(p, total, o, v=None):
    """v: None or {'pre':..,'post':..,'minlen':..,'min_edge':..}; pre=post=0 with v given means trim to speech exactly."""
    fs = o["frame_sec"]
    if total <= o["max_seg_sec"]: return [(0.0, total)]
    n = max(0, int(math.ceil(total / fs - 1e-9)))
    max_f = max(2, int(math.floor(o["max_seg_sec"] / fs + 1e-9)))
    min_f = min(max_f - 1, max(1, int(math.ceil(o["min_seg_sec"] / fs - 1e-9))))
    pause_f = max(1, int(math.ceil(o["min_pause_sec"] / fs - 1e-9)))
    sp, runs = mask(p, n, o)
    pauses = [(a, b) for a, b in runs(False) if b - a >= pause_f]
    cum = np.concatenate([[0], np.cumsum(sp)])
    out = []
    def emit(a, b, end_sec):
        be = min(b, n)
        if not (be > a and cum[be] - cum[a] > 0): return
        s0, e0 = a * fs, end_sec
        if v is not None and v.get("pre") is not None:
            fa, fb = a, be
            while fa < fb and not sp[fa]: fa += 1
            while fb > fa and not sp[fb - 1]: fb -= 1
            ts, te = fa * fs - v["pre"], fb * fs + v["post"]
            me = v.get("min_edge", 0.0) or 0.0
            ns, ne = s0, e0
            if ts - s0 >= me and ts > s0: ns = ts
            if e0 - te >= me and te < e0: ne = te
            ml = v.get("minlen", 0.0) or 0.0
            if ne - ns < ml:
                need = ml - (ne - ns); ns2 = max(s0, ns - need / 2); ne2 = min(e0, ne + need / 2)
                # give the unused share to the other side
                if ns2 - (ns - need / 2) < -1e-12: ne2 = min(e0, ne2 + (ns - need / 2 - ns2))
                if (ne + need / 2) - ne2 > 1e-12: ns2 = max(s0, ns2 - ((ne + need / 2) - ne2))
                ns, ne = ns2, ne2
            s0, e0 = ns, ne
        out.append((s0, e0))
    s = 0
    while total - s * fs > o["max_seg_sec"] + 1e-9:
        lo, hi = s + min_f, s + max_f
        c = -1; c_mid = -1
        for a, b in pauses:
            mid = (a + b) // 2
            if a >= lo and b <= hi: c = mid
            if lo <= mid <= hi: c_mid = mid
        if c < 0: c = c_mid
        if c <= s: c = hi
        emit(s, c, c * fs); s = c
    emit(s, n, total)
    return out

def base(trim): return None if trim is None else {"pre": trim, "post": trim}

"""Shared library for the real-recording VAD study. 10 ms evaluation grid; numba kernels.

Segmenter semantics are those of parakeet.cpp src/vad_segmenter.cpp (master da8bb4c)."""
import glob, json, os, hashlib, math
import numpy as np
from numba import njit

ROOT = os.environ.get("VAD_REAL_ROOT") or os.getcwd()
GR = 0.01
CLI_CLEAN = os.environ.get("PARAKEET_CLI", f"{ROOT}/parakeet-cli-clean")   # a parakeet-cli of master (no run gate needed)
MODELS = os.environ.get("VAD_REAL_MODELS", f"{ROOT}/models")             # silero-vad-f16, ultra-vad-q8_0, redux-vad GGUFs (and ASR GGUFs for wer_dec.py)
FS = {"silero": 0.032, "ultra": 0.08, "redux": 0.08}

# ---------------------------------------------------------------- grid helpers
def hold(p, fs, n):
    """sample-and-hold of per-frame values onto n 10 ms cells (cell centre decides the frame)."""
    idx = ((np.arange(n) + 0.5) * GR / fs).astype(np.int64)
    out = np.zeros(n, np.float32)
    ok = idx < len(p)
    out[ok] = p[idx[ok]]
    return out

@njit(cache=True)
def runs_nb(m):
    s = []; e = []
    n = len(m); i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]: j += 1
            s.append(i); e.append(j); i = j
        else:
            i += 1
    return np.array(s, np.int64), np.array(e, np.int64)

def runs(m):
    return runs_nb(np.ascontiguousarray(m, dtype=np.bool_))

@njit(cache=True)
def post_nb(m, bridge, min_speech, min_pause, pad, gr):
    """repo speech_regions semantics on a grid of period gr: bridge, drop short, merge closer than min_pause, pad (no overlap handling needed
    for the mask: padded regions just union)."""
    n = len(m); m = m.copy()
    # bridge silent gaps between two speech runs
    i = 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]: j += 1
            if i > 0 and j < n and (j - i) * gr + 1e-9 < bridge:
                for k in range(i, j): m[k] = True
            i = j
        else:
            i += 1
    # drop short speech runs
    i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]: j += 1
            if (j - i) * gr + 1e-9 < min_speech:
                for k in range(i, j): m[k] = False
            i = j
        else:
            i += 1
    pf = max(1, int(math.ceil(min_pause / gr - 1e-9)))
    out = np.zeros(n, np.bool_)
    ra = -1; rb = -1
    pp = int(round(pad / gr))
    i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]: j += 1
            if ra >= 0 and i - rb < pf:
                rb = j
            else:
                if ra >= 0:
                    for k in range(max(0, ra - pp), min(n, rb + pp)): out[k] = True
                ra = i; rb = j
            i = j
        else:
            i += 1
    if ra >= 0:
        for k in range(max(0, ra - pp), min(n, rb + pp)): out[k] = True
    return out

POST_HEAD = dict(bridge=0.1, min_speech=0.1, min_pause=0.2, pad=0.0)
POST_SIL = dict(bridge=0.1, min_speech=0.25, min_pause=0.1, pad=0.03)

def post(m, bridge=0.1, min_speech=0.1, min_pause=0.2, pad=0.0):
    return post_nb(np.ascontiguousarray(m, dtype=np.bool_), bridge, min_speech, min_pause, pad, GR)

# ---------------------------------------------------------------- exact native-frame replica of speech_regions
@njit(cache=True)
def native_mask(p, fs, total, thr, bridge, min_speech, min_pause, pad, n10):
    """speech_regions() on native frames, returned as a 10 ms bool grid (n10 cells) plus the regions."""
    n = max(0, int(math.ceil(total / fs - 1e-9)))
    sp = np.zeros(n, np.bool_)
    for f in range(min(n, len(p))): sp[f] = p[f] >= thr
    i = 0
    while i < n:
        if not sp[i]:
            j = i
            while j < n and not sp[j]: j += 1
            if i > 0 and j < n and (j - i) * fs + 1e-9 < bridge:
                for k in range(i, j): sp[k] = True
            i = j
        else:
            i += 1
    i = 0
    while i < n:
        if sp[i]:
            j = i
            while j < n and sp[j]: j += 1
            if (j - i) * fs + 1e-9 < min_speech:
                for k in range(i, j): sp[k] = False
            i = j
        else:
            i += 1
    pf = max(1, int(math.ceil(min_pause / fs - 1e-9)))
    rs = []; re_ = []
    i = 0
    while i < n:
        if sp[i]:
            j = i
            while j < n and sp[j]: j += 1
            if len(rs) > 0 and i - re_[-1] < pf:
                re_[-1] = j
            else:
                rs.append(i); re_.append(j)
            i = j
        else:
            i += 1
    R = len(rs)
    st = np.zeros(R); en = np.zeros(R)
    for r in range(R):
        st[r] = rs[r] * fs; en[r] = min(re_[r] * fs, total)
    if pad > 0.0:
        st2 = st.copy(); en2 = en.copy()
        for r in range(R):
            st2[r] = max(0.0, st[r] - pad); en2[r] = min(total, en[r] + pad)
        for r in range(R - 1):
            gap = st[r + 1] - en[r]
            if gap < 2.0 * pad:
                mid = en[r] + gap / 2.0
                en2[r] = mid; st2[r + 1] = mid
        st = st2; en = en2
    out = np.zeros(n10, np.bool_)
    for r in range(R):
        a = int(round(st[r] / GR)); b = int(round(en[r] / GR))
        for k in range(max(0, a), min(n10, b)): out[k] = True
    return out, st, en

# ---------------------------------------------------------------- fusion / gate
@njit(cache=True)
def two_stage_nb(S, H, pre, post_, Y):
    """Silero decides; head extends boundaries outward (pre cells before a start, post_ cells after an end) only where the head says speech
    contiguously; fills Silero gaps shorter than Y cells when the head calls >= 50 percent of the gap speech. (rules.py of the fusion study)"""
    n = len(S); out = S.copy()
    if pre > 0 or post_ > 0:
        i = 0
        while i < n:
            if S[i]:
                j = i
                while j < n and S[j]: j += 1
                a = i; b = j
                k = a
                while k > 0 and a - k < pre and H[k - 1]: k -= 1
                for q in range(k, a): out[q] = True
                k = b
                while k < n and k - b < post_ and H[k]: k += 1
                for q in range(b, k): out[q] = True
                i = j
            else:
                i += 1
    if Y > 0:
        i = 0
        while i < n:
            if not out[i]:
                j = i
                while j < n and not out[j]: j += 1
                if i > 0 and j < n and (j - i) < Y:
                    c = 0
                    for q in range(i, j):
                        if H[q]: c += 1
                    if c * 2 >= (j - i):   # mean >= 0.5
                        for q in range(i, j): out[q] = True
                i = j
            else:
                i += 1
    return out

def two_stage(S, H, pre, post_, Y):
    return two_stage_nb(np.ascontiguousarray(S, np.bool_), np.ascontiguousarray(H, np.bool_), int(pre), int(post_), int(Y))

def gate_frames(p, thr=0.5, med=0.92):
    """head gate on native frames: keep a run of p >= thr only if the median probability of its frames is >= med."""
    m = p >= thr
    keep = np.zeros(len(p), bool)
    s, e = runs(m)
    for a, b in zip(s, e):
        if np.median(p[a:b]) >= med: keep[a:b] = True
    return keep

# ---------------------------------------------------------------- segmenter replica (segment_by_vad)
LAST_CUTS = []
def segment_by_mask(sp_in, fs, total, max_seg=30.0, min_seg=1.0, min_pause=0.2, trim=0.3, policy="base", long_gap=None):
    """pk::segment_by_vad on an already smoothed boolean speech mask of period fs (the mask is NOT re-smoothed here).
    policy: base = the shipped rule (last pause fully inside the window, then pause midpoint in window, then hard cut);
            longest = cut at the longest pause in the window; soft = base, but with no pause use the longest silent gap of any length before a hard cut.
    long_gap: if set (seconds), additionally cut at every pause at least this long (segments may be shorter than min_seg then)."""
    if total <= max_seg: return [(0.0, total)], 0
    n = max(0, int(math.ceil(total / fs - 1e-9)))
    sp = np.zeros(n, bool); k = min(n, len(sp_in)); sp[:k] = sp_in[:k]
    max_f = max(2, int(math.floor(max_seg / fs + 1e-9)))
    min_f = min(max_f - 1, max(1, int(math.ceil(min_seg / fs - 1e-9))))
    pause_f = max(1, int(math.ceil(min_pause / fs - 1e-9)))
    sil_s, sil_e = runs(~sp)
    pauses = [(a, b) for a, b in zip(sil_s, sil_e) if b - a >= pause_f]
    cum = np.concatenate([[0], np.cumsum(sp)])
    out = []
    hard = 0
    cuts = []
    LAST_CUTS.clear()
    def emit(a, b, end_sec):
        be = min(b, n)
        if not (be > a and cum[be] - cum[a] > 0): return
        s0, e0 = a * fs, end_sec
        if trim > 0:
            fa, fb = a, be
            nz = np.flatnonzero(sp[a:be])
            fa = a + nz[0]; fb = a + nz[-1] + 1
            s0 = max(s0, fa * fs - trim); e0 = min(e0, fb * fs + trim)
        out.append((s0, e0))
    s = 0
    lg = None if long_gap is None else max(1, int(math.ceil(long_gap / fs - 1e-9)))
    while total - s * fs > max_seg + 1e-9:
        lo, hi = s + min_f, s + max_f
        c = -1; c_mid = -1
        if policy == "longest":
            best = -1
            for a, b in pauses:
                mid = (a + b) // 2
                if a >= lo and b <= hi and b - a > best: best = b - a; c = mid
            if c < 0:
                for a, b in pauses:
                    mid = (a + b) // 2
                    if lo <= mid <= hi: c_mid = mid
        else:
            for a, b in pauses:
                mid = (a + b) // 2
                if a >= lo and b <= hi: c = mid
                if lo <= mid <= hi: c_mid = mid
        if c < 0: c = c_mid
        if c < 0 and policy == "soft":
            best = 0
            for a, b in zip(sil_s, sil_e):
                aa, bb = max(a, lo), min(b, hi)
                if bb - aa > best: best = bb - aa; c = (aa + bb) // 2
        is_hard = False
        if c <= s:
            c = hi; hard += 1; is_hard = True
        cuts.append((c * fs, is_hard)); LAST_CUTS.append((c * fs, is_hard))
        emit(s, c, c * fs); s = c
    emit(s, n, total)
    if lg is not None:   # split every resulting segment at pauses >= long_gap (inside it), then trim again
        out2 = []
        for (a0, b0) in out:
            fa = int(round(a0 / fs)); fb = int(round(b0 / fs))
            seg = sp[fa:fb]
            ss, ee = runs(~seg)
            cuts = [fa + (x + y) // 2 for x, y in zip(ss, ee) if y - x >= lg and x > 0 and y < len(seg)]
            pos = fa
            for c in cuts + [fb]:
                nz = np.flatnonzero(sp[pos:c])
                if len(nz):
                    out2.append((max(pos * fs, (pos + nz[0]) * fs - trim), min(c * fs, (pos + nz[-1] + 1) * fs + trim)))
                pos = c
        out = out2
    return out, hard

def seg_mask(segs, n10):
    m = np.zeros(n10, bool)
    for a, b in segs: m[int(round(a / GR)):int(round(b / GR))] = True
    return m

# ---------------------------------------------------------------- metrics
def counts(p, r):
    n = min(len(p), len(r)); p = p[:n]; r = r[:n]
    return np.array([(p & r).sum(), (p & ~r).sum(), (~p & r).sum(), (~p & ~r).sum()], np.int64)

def prf(c):
    tp, fp, fn, _ = c
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    return P, R, 2 * P * R / max(1e-12, P + R)

# ---------------------------------------------------------------- data
def ref_mask(turns_s, turns_e, dur, merge_gap=0.1):
    n = int(round(dur / GR)); m = np.zeros(n, bool)
    iv = sorted(zip(turns_s, turns_e))
    mg = []
    for s, e in iv:
        if mg and s - mg[-1][1] < merge_gap: mg[-1][1] = max(mg[-1][1], e)
        else: mg.append([s, e])
    for s, e in mg: m[int(round(s / GR)):int(round(e / GR))] = True
    return m

def split_of(rid, domain):
    """recording-disjoint tune/held split. vox: dev=tune, test=held (different videos). ami: validation=tune, test=held (different meetings).
    others: md5 parity of the id."""
    if domain in ("vox", "ami"):
        return "tune" if rid.startswith(("dev", "validation")) else "held"
    return "tune" if int(hashlib.md5(rid.encode()).hexdigest(), 16) % 2 == 0 else "held"

def load_recordings():
    """returns list of dicts: id, domain, split, dur, n, wav, ref (bool or None for non-speech-only: all False), nonspeech(bool)"""
    D = []
    for dom in ("vox", "ami"):
        for j in sorted(glob.glob(f"{ROOT}/data/{dom}/*.json")):
            m = json.load(open(j)); rid = m["id"]
            D.append(dict(id=f"{dom}_{rid}", domain=dom, split=split_of(rid, dom), dur=m["dur"], wav=j[:-5] + ".wav",
                          ref=ref_mask(m["start"], m["end"], m["dur"]), nonspeech=False))
    for j in sorted(glob.glob(f"{ROOT}/data/ava/*.json")):
        m = json.load(open(j)); rid = os.path.basename(j)[:-5]
        import soundfile as sf
        dur = sf.info(j[:-5] + ".wav").duration
        D.append(dict(id=f"ava_{rid}", domain="ava", split=split_of(rid, "ava"), dur=dur, wav=j[:-5] + ".wav",
                      ref=ref_mask(m["onset"], m["offset"], dur), nonspeech=False))
    import soundfile as sf
    for dom, pat in (("music", "data/musan/music_*.wav"), ("noise", "data/musan/noise_*.wav"), ("esc", "data/esc/esc_*.wav")):
        for w in sorted(glob.glob(f"{ROOT}/{pat}")):
            rid = os.path.basename(w)[:-4]; dur = sf.info(w).duration
            D.append(dict(id=f"{dom}_{rid}", domain=dom, split=split_of(rid, dom), dur=dur, wav=w,
                          ref=np.zeros(int(round(dur / GR)), bool), nonspeech=True))
    return D

def load_probs(rid):
    return {k: np.load(f"{ROOT}/probs/{rid}.{k}.npy") for k in FS}

def boot_idx(n, B, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(B, n))

@njit(cache=True)
def smooth_native(p, fs, total, thr, bridge, min_speech):
    """the speech mask of segment_by_vad after bridging and dropping short runs (no merging at min_pause, no pad), native frames."""
    n = max(0, int(math.ceil(total / fs - 1e-9)))
    sp = np.zeros(n, np.bool_)
    for f in range(min(n, len(p))): sp[f] = p[f] >= thr
    i = 0
    while i < n:
        if not sp[i]:
            j = i
            while j < n and not sp[j]: j += 1
            if i > 0 and j < n and (j - i) * fs + 1e-9 < bridge:
                for k in range(i, j): sp[k] = True
            i = j
        else:
            i += 1
    i = 0
    while i < n:
        if sp[i]:
            j = i
            while j < n and sp[j]: j += 1
            if (j - i) * fs + 1e-9 < min_speech:
                for k in range(i, j): sp[k] = False
            i = j
        else:
            i += 1
    return sp

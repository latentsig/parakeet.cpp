from lib import *
def names(kind=None, split=None, conds=None, fam=None):
    out = []
    for m in MAN.values():
        if kind and m["kind"] != kind: continue
        if m.get("dur", 99) <= 30: continue
        if split and m["split"] != split: continue
        if kind == "sinr":
            c = m["cond"]
            if conds and c not in conds: continue
            if fam == "noisy" and c == "clean": continue
        out.append(m["name"])
    return out
W_ = ["white20", "white10", "white5", "white0"]; P_ = ["pink20", "pink10", "pink5", "pink0"]
def G(split):
    s = None if split == "all" else split
    return {
        f"talks[{split}]": names("talk", s),
        f"clean[{split}]": names("sinr", s, ["clean"]),
        f"white[{split}]": names("sinr", s, W_),
        f"pink[{split}]": names("sinr", s, P_),
        f"white0[{split}]": names("sinr", s, ["white0"]),
        f"pink0[{split}]": names("sinr", s, ["pink0"]),
        f"pink5[{split}]": names("sinr", s, ["pink5"]),
        f"noisy-all[{split}]": names("sinr", s, W_ + P_),
        f"sinr-all[{split}]": names("sinr", s, ["clean"] + W_ + P_),
    }

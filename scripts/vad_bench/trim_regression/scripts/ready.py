import sys, os; sys.path.insert(0, os.path.dirname(__file__))
from groups import *
det, kind, vs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
ok = tot = 0
for m in MAN.values():
    if m["kind"] != kind or m.get("dur", 99) <= 30: continue
    if len(sys.argv) > 4 and m["kind"] == "sinr" and m["cond"] not in sys.argv[4].split(","): continue
    tot += 1; c = dec.load_cache(det, m["name"])
    ok += all(dec.key(s, e) in c for v in vs for s, e in segs_for(det, m["name"], v))
print(det, kind, ok, "/", tot)

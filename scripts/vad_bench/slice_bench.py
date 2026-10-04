#!/usr/bin/env python3
"""Stand-alone head (slice) benchmark: speed, load time and peak memory, full ASR GGUF vs slice GGUF.

usage: slice_bench.py speed|load|rss
Environment: GGUF_DIR (full GGUFs: ultra-q8_0 ultra-f16 redux-keep redux-deq-f16, plus silero-vad-f16.gguf),
SLICE_DIR (the *-vad.gguf slices of PR 87), WAV300 (300 s clip), CLIP (a short clip for the memory run),
VADBENCH (binary built from vad_slice_bench.cpp inside the PR 87 tree), CLI (parakeet-cli), OUT (output dir).
"""
import subprocess, time, statistics, json, sys, os, re
G=os.environ["GGUF_DIR"]; V=os.environ["SLICE_DIR"]; B=os.environ["VADBENCH"]; WAV=os.environ["WAV300"]
CLIP=os.environ["CLIP"]; CLI=os.environ["CLI"]; OUT=os.environ.get("OUT",".")
S=G
CFG=[("ultra-q8_0 full",G+"/ultra-q8_0.gguf"),("ultra-q8_0 slice",V+"/ultra-q8_0-vad.gguf"),
     ("ultra-f16 full",G+"/ultra-f16.gguf"),("ultra-f16 slice",V+"/ultra-f16-vad.gguf"),
     ("redux-keep full",G+"/redux-keep.gguf"),("redux-keep slice",V+"/redux-keep-vad.gguf"),
     ("redux-deq-f16 full",G+"/redux-deq-f16.gguf"),("redux-deq-f16 slice",V+"/redux-deq-f16-vad.gguf"),
     ("silero-vad-f16",G+"/silero-vad-f16.gguf")]
log=open(OUT+"/gate.log","a")
def cpu_idle():
    r={}
    for l in open("/proc/stat"):
        if l.startswith("cpu") and l[3].isdigit():
            f=l.split(); v=list(map(int,f[1:])); r[int(f[0][3:])]=(v[3]+v[4],sum(v))
    return r
def gate(tag):
    a=cpu_idle(); runs=[]
    for _ in range(20):
        runs.append(int(open("/proc/loadavg").read().split()[3].split("/")[0])-1); time.sleep(0.5)
    b=cpu_idle(); load1=float(open("/proc/loadavg").read().split()[0])
    busy={c:1-(b[c][0]-a[c][0])/max(1,b[c][1]-a[c][1]) for c in a}
    order=sorted(busy,key=busy.get)
    m=statistics.mean(runs); ok=(m<4) or (load1<5)
    log.write(f"{time.strftime('%H:%M:%S')} {tag} mean_runnable={m:.1f} load1={load1} ok={ok} idlest={order[:8]}\n"); log.flush()
    return ok,order
def runb(args,cpus):
    cmd=["flock","/tmp/pk-bench.lock","taskset","-c",",".join(map(str,cpus))]+args
    return subprocess.run(cmd,capture_output=True,text=True,check=True).stdout.split()
mode=sys.argv[1]
if mode=="speed":
    res={(n,t):[] for n,_ in CFG for t in (1,8)}
    for rnd in range(5):
        for t in (1,8):
            while True:
                ok,order=gate(f"speed round{rnd} t{t}")
                if ok: break
                log.write("not quiet, retry in 30s\n"); log.flush(); time.sleep(30)
            cpus=order[:t]
            for n,p in CFG:
                s,x=runb([B,"speed",p,WAV,str(t)],cpus)
                res[(n,t)].append(float(x))
    out={f"{n}|{t}":v for (n,t),v in res.items()}
    json.dump(out,open(OUT+"/speed.json","w"),indent=1)
    for (n,t),v in res.items(): print(f"{n:22s} {t} thr best {max(v):7.1f}x median {statistics.median(v):7.1f}x all {v}")
elif mode=="load":
    out={}
    for n,p in CFG:
        if len(sys.argv)>2 and not n.startswith(sys.argv[2]): continue
        for cold in (False,True):
            while True:
                ok,order=gate(f"load {n} cold={cold}")
                if ok: break
                time.sleep(30)
            a=[B,"load",p,"10"]+(["cold"] if cold else [])
            if not cold: a=[B,"load",p,"11"]   # first load warms the page cache and is dropped
            v=list(map(float,runb(a,order[:8])))
            if not cold: v=v[1:]
            out[f"{n}|{'cold' if cold else 'warm'}"]=v
            print(f"{n:22s} {'cold' if cold else 'warm'} median {statistics.median(v):8.1f} ms  min {min(v):.1f} max {max(v):.1f}",flush=True)
    json.dump(out,open(OUT+"/load%s.json"%("-"+sys.argv[2] if len(sys.argv)>2 else ""),"w"),indent=1)
elif mode=="rss":
    for n,p in CFG:
        for what,cmd in (("load-only",[B,"load",p,"1"]),("load+vad33s",[CLI,"vad","--model",p,"--input",CLIP,"--threads","8"])):
            r=subprocess.run(["/usr/bin/time","-v"]+cmd,capture_output=True,text=True)
            m=re.search(r"Maximum resident set size \(kbytes\): (\d+)",r.stderr)
            print(f"{n:22s} {what:12s} peak RSS {int(m.group(1))/1024:8.1f} MiB",flush=True)

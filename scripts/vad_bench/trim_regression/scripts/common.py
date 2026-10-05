import json, os, re
import numpy as np
# Paths come from the environment. PK_WORK: work directory (data/, probs/, cache/, tmp/ live there; default
# is the current directory). PK_CLI: the patched parakeet-cli. PK_GGUF: directory with the model files.
W = os.path.abspath(os.environ.get("PK_WORK", os.getcwd()))
CLI = os.environ.get("PK_CLI", f"{W}/build/examples/cli/parakeet-cli")
GG = os.environ.get("PK_GGUF", f"{W}/gguf")
os.makedirs(f"{W}/tmp", exist_ok=True)
DET = {  # name: (asr model, silero or None)
    "ultra": (f"{GG}/ultra-q8_0.gguf", None),
    "redux": (f"{GG}/redux-keep.gguf", None),
    "v3": (f"{GG}/tdt-0.6b-v3-q8_0.gguf", f"{GG}/silero-vad-f16.gguf"),
}
HEAD = dict(threshold=0.5, frame_sec=0.08, max_seg_sec=30.0, min_pause_sec=0.2, min_seg_sec=1.0, bridge_sec=0.1, min_speech_sec=0.1)
SIL = dict(HEAD, frame_sec=0.032, min_speech_sec=0.25, min_pause_sec=0.1, bridge_sec=0.1)
OPTS = {"ultra": HEAD, "redux": HEAD, "v3": SIL}
def manifest(): return json.load(open(f"{W}/data/manifest.json"))
def probs(det, name): return np.fromfile(f"{W}/probs/{det}/{name}.f32", dtype=np.float32)
def norm(s): return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ")).split()

"""Independent reference of the Moondream Parakeet VAD head path, built on HF transformers
(ParakeetFeatureExtractor for log-mel, ParakeetEncoderSubsamplingConv2D for the 8x subsampler)
plus the head structure documented in parakeet.cpp docs/ternary.md. Weights come from the HF safetensors."""
import json, os, subprocess, numpy as np, torch, soundfile as sf
torch.set_num_threads(4)
from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig
from transformers.models.parakeet.modeling_parakeet import ParakeetEncoderSubsamplingConv2D
from transformers.models.parakeet.feature_extraction_parakeet import ParakeetFeatureExtractor
HERE = os.environ.get("NOISE_DIVE_WORK", os.path.dirname(os.path.abspath(__file__)))  # holds build/, ref/, hfmeta/
CLI = os.environ.get("PARAKEET_CLI", f"{HERE}/build/examples/cli/parakeet-cli")
SCR = os.environ.get("GGUF_ROOT", os.path.dirname(HERE))  # holds gguf/ and vad-card/
GG = {'redux': f"{SCR}/vad-card/redux-vad.gguf", 'ultra': f"{SCR}/vad-card/ultra-vad-q8_0.gguf", 'ultra-f16': f"{SCR}/gguf/ultra-f16.gguf", 'redux-keep': f"{SCR}/gguf/redux-keep.gguf"}
FE = ParakeetFeatureExtractor(feature_size=128)
EPS = 1e-5

def raw_logmel(x):
    """[T,128] un-normalised log-mel of a float32 mono 16 kHz array (HF code path)."""
    w = torch.tensor(np.asarray(x, np.float32))[None, :]
    pre = torch.cat([w[:, :1], w[:, 1:] - FE.preemphasis * w[:, :-1]], 1)
    return FE._torch_extract_fbank_features(pre)[0]

def norm_mel(m, mean=None, std=None, f64=True):
    """per-feature (over time) mean/std normalisation as HF/NeMo (HF masks the last STFT frame: stats over n_samples//160
    frames); or fixed mean/std (len-128 tensors). f64=True accumulates in double like parakeet.cpp; HF uses float32."""
    n = m.shape[0] - 1
    d = m.double() if f64 else m
    if mean is None:
        mv = d[:n]
        mean = mv.sum(0) / n
        std = torch.sqrt(((mv - mean) ** 2).sum(0) / (n - 1))
    else:
        mean, std = mean.to(d.dtype), std.to(d.dtype)
    out = ((d - mean) / (std + EPS)).float()
    return out[:n]     # the masked last STFT frame is dropped (parakeet.cpp keeps n_samples//160 frames)

class Ref:
    def __init__(self, name):
        self.name = name
        W = np.load(f"{HERE}/ref/{name}_weights.npz")
        cfg = ParakeetEncoderConfig(**{k: v for k, v in json.load(open(f"{HERE}/hfmeta/{name}.config.json"))['encoder_config'].items() if k != 'model_type'})
        self.sub = ParakeetEncoderSubsamplingConv2D(cfg).eval()
        sd = {k[len('encoder.subsampling.'):]: torch.tensor(v.astype(np.float32)) for k, v in W.items() if k.startswith('encoder.subsampling.')}
        self.sub.load_state_dict(sd)
        g = lambda k: torch.tensor(W[k].astype(np.float32))
        self.pw, self.pb = g('vad_head.proj.weight')[:, :, 0], g('vad_head.proj.bias')
        self.cw, self.cb = g('vad_head.ctx.weight'), g('vad_head.ctx.bias')
        self.ow, self.ob = g('vad_head.out.weight')[0, :, 0], g('vad_head.out.bias')[0]
    @torch.no_grad()
    def sub_out(self, mel_norm):
        return self.sub(mel_norm[None])[0]            # [T/8,1024]
    @torch.no_grad()
    def head_logits(self, s):
        h = torch.nn.functional.silu(s @ self.pw.T + self.pb)          # [T,128]
        c = torch.nn.functional.conv1d(h.T[None], self.cw, self.cb, padding=2)[0].T
        c = torch.nn.functional.silu(c)
        return c @ self.ow + self.ob                                   # logits [T]
    def logits(self, x, mean=None, std=None):
        return self.head_logits(self.sub_out(norm_mel(raw_logmel(x), mean, std))).numpy()
    def probs(self, x, **kw):
        return 1 / (1 + np.exp(-self.logits(x, **kw)))
    def feats(self, x, mean=None, std=None):
        return self.sub_out(norm_mel(raw_logmel(x), mean, std)).numpy()

_refs = {}
def ref(name):
    if name not in _refs: _refs[name] = Ref(name)
    return _refs[name]

def cli_probs(name, wav, thr=None):
    out = subprocess.run([CLI, 'vad', '--model', GG[name], '--input', wav, '--probabilities', '--threads', '4'], capture_output=True, text=True)
    j = json.loads(out.stdout)
    return np.array(j['probabilities'], np.float32)

def wr(path, x):
    sf.write(path, np.clip(x, -1, 1).astype(np.float32), 16000, subtype='FLOAT')

BLOCK = 120 * 16000
def logits_blocks(name, x, mean=None, std=None, block=BLOCK, tail=5 * 16000):
    """logits with parakeet.cpp's 120 s blocks (own mel normalisation per block), or fixed mean/std if given."""
    r = ref(name); out = []; pos = 0; fs = 1280
    while True:
        ln = min(block, len(x) - pos)
        if len(x) - pos - ln < tail: ln = len(x) - pos
        z = r.logits(x[pos:pos + ln], mean, std)
        last = pos + ln >= len(x)
        if not last: z = z[:block // fs]
        out.append(z); pos += ln
        if pos >= len(x): break
    return np.concatenate(out)
def sig(z): return 1 / (1 + np.exp(-z))
def frame_mask(spans, n, fs=0.08):
    t = (np.arange(n) + 0.5) * fs; m = np.zeros(n, bool)
    for a, b in spans: m |= (t >= a) & (t < b)
    return m

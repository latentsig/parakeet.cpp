#!/usr/bin/env python3
"""Convert the official Silero VAD ONNX model to a small GGUF file.

Source: https://github.com/snakers4/silero-vad (MIT licence), file
``src/silero_vad/data/silero_vad.onnx``. That file holds two complete weight
sets, one for 16 kHz and one for 8 kHz, in the two branches of a top-level
``If`` node. Both are written, with the tensor name prefixes ``vad16k.`` and
``vad8k.``. The names after the prefix are the names used in the ONNX graph.

Tensor shapes follow the PyTorch layout: gguf stores the dimensions reversed,
so a conv weight of shape (out, in, k) is read by ggml as ne = [k, in, out].

Only ``onnx`` and ``numpy`` and ``gguf`` are needed (no torch).

    python scripts/convert_silero_vad_to_gguf.py silero_vad.onnx silero-vad.gguf
    python scripts/convert_silero_vad_to_gguf.py silero_vad.onnx silero-vad-f16.gguf --dtype f16

``--dtype f16`` stores the conv and LSTM weights as F16. The STFT basis and all
biases stay F32. The C++ side converts to F32 at load time.
"""
import argparse
import hashlib
import pathlib
import sys

import numpy as np

try:
    import gguf
    import onnx
    from onnx import numpy_helper
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency: {e}", file=sys.stderr)
    sys.exit(2)

# ONNX constant name -> (expected shape for 16 kHz, expected shape for 8 kHz)
WANT = {
    "stft.forward_basis_buffer": ((258, 1, 256), (130, 1, 128)),
    "encoder.0.reparam_conv.weight": ((128, 129, 3), (128, 65, 3)),
    "encoder.0.reparam_conv.bias": ((128,), (128,)),
    "encoder.1.reparam_conv.weight": ((64, 128, 3), (64, 128, 3)),
    "encoder.1.reparam_conv.bias": ((64,), (64,)),
    "encoder.2.reparam_conv.weight": ((64, 64, 3), (64, 64, 3)),
    "encoder.2.reparam_conv.bias": ((64,), (64,)),
    "encoder.3.reparam_conv.weight": ((128, 64, 3), (128, 64, 3)),
    "encoder.3.reparam_conv.bias": ((128,), (128,)),
    "decoder.rnn.weight_ih": ((512, 128), (512, 128)),
    "decoder.rnn.weight_hh": ((512, 128), (512, 128)),
    "decoder.rnn.bias_ih": ((512,), (512,)),
    "decoder.rnn.bias_hh": ((512,), (512,)),
    "decoder.decoder.2.weight": ((1, 128, 1), (1, 128, 1)),
    "decoder.decoder.2.bias": ((1,), (1,)),
}

# Per-rate hyper-parameters (see src/silero_vad/utils_vad.py in the official repo).
RATES = {
    16000: dict(branch="then", prefix="vad16k", chunk=512, context=64, n_fft=256, hop=128, right_pad=64),
    8000: dict(branch="else", prefix="vad8k", chunk=256, context=32, n_fft=128, hop=64, right_pad=32),
}


def branch_weights(model, branch):
    top = [n for n in model.graph.node if n.op_type == "If"]
    if len(top) != 1:
        raise SystemExit("unexpected graph: expected exactly one top-level If node")
    sub = [a for a in top[0].attribute if a.name == f"{branch}_branch"][0].g
    out = {}
    for n in sub.node:
        if n.op_type != "Constant":
            continue
        name = n.output[0].split("Inline_0__")[-1]
        if name in WANT:
            out[name] = numpy_helper.to_array(n.attribute[0].t)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", help="silero_vad.onnx (the 8k + 16k file)")
    ap.add_argument("output", help="output .gguf")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32")
    ap.add_argument("--version", default="", help="silero-vad release the file comes from, e.g. 6.2.3")
    args = ap.parse_args()

    raw = pathlib.Path(args.onnx).read_bytes()
    model = onnx.load_from_string(raw)

    w = gguf.GGUFWriter(args.output, "silero_vad")
    w.add_string("general.name", "silero-vad")
    w.add_string("general.license", "MIT")
    w.add_string("general.url", "https://github.com/snakers4/silero-vad")
    w.add_string("silero_vad.source.sha256", hashlib.sha256(raw).hexdigest())
    if args.version:
        w.add_string("silero_vad.source.version", args.version)
    w.add_array("silero_vad.sample_rates", sorted(RATES))
    w.add_uint32("silero_vad.encoder.n_layers", 4)
    w.add_array("silero_vad.encoder.strides", [1, 2, 2, 1])
    w.add_array("silero_vad.encoder.channels", [128, 64, 64, 128])
    w.add_uint32("silero_vad.encoder.kernel", 3)
    w.add_uint32("silero_vad.lstm.hidden", 128)

    for sr, p in RATES.items():
        k = f"silero_vad.{sr}"
        w.add_uint32(f"{k}.chunk_samples", p["chunk"])
        w.add_uint32(f"{k}.context_samples", p["context"])
        w.add_uint32(f"{k}.stft.n_fft", p["n_fft"])
        w.add_uint32(f"{k}.stft.hop", p["hop"])
        w.add_uint32(f"{k}.stft.right_reflect_pad", p["right_pad"])
        tens = branch_weights(model, p["branch"])
        for name, shapes in WANT.items():
            if name not in tens:
                raise SystemExit(f"missing {name} in the {sr} Hz branch")
            t = tens[name]
            want = shapes[0 if sr == 16000 else 1]
            if tuple(t.shape) != want or t.dtype != np.float32:
                raise SystemExit(f"{sr} Hz {name}: got {t.shape} {t.dtype}, want {want} float32")
            is_big = name.endswith("weight") or name.endswith("weight_ih") or name.endswith("weight_hh")
            if args.dtype == "f16" and is_big and not name.startswith("stft."):
                t = t.astype(np.float16)
            w.add_tensor(f"{p['prefix']}.{name}", np.ascontiguousarray(t))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output} ({pathlib.Path(args.output).stat().st_size} bytes)")


if __name__ == "__main__":
    main()

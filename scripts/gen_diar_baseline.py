#!/usr/bin/env python3
"""Dump a NeMo speaker-diarization reference to a baseline GGUF.

Used by tests/test_diarization_accuracy.cpp to check the C++ diarization
pipeline (nvidia/Nemotron-3-Diarization and compatible Sortformer models)
against NeMo on the same audio.

Needs a NeMo with self_attention_model='rope' support (NeMo main / >= 3.1;
NeMo 3.0 cannot instantiate Nemotron-3-Diarization).

Stored tensors (numpy shapes; the C++ side reads them outer..inner):

* ``audio``          ``[S]``           the 16 kHz mono clip the reference used,
                                       so the test does not depend on a wav path
* ``offline_probs``  ``[n_spk, T]``    offline forward() speaker probabilities
                                       (``streaming_mode=False``), one frame per
                                       10 ms mel frame
* ``offline_segs``   ``[N, 3]``        offline diarize() segments as
                                       (speaker, start_s, end_s)
* ``stream_probs``   ``[n_spk, T]``    streaming forward() probabilities
                                       (``streaming_mode=True``, the model's
                                       own chunk / speaker-cache config)
* ``stream_segs``    ``[N, 3]``        streaming diarize() segments

``dither`` is forced to 0 so the mel is deterministic (the C++ side has no
dither).

Usage:
    python scripts/gen_diar_baseline.py \\
        --model /path/to/Nemotron-3-Diarization.nemo \\
        --audio tests/fixtures/two_speakers.wav \\
        --output /tmp/diar_baseline.gguf
"""
import argparse
import sys

import numpy as np

try:
    import gguf
    import soundfile as sf
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel
except ImportError as e:  # pragma: no cover - env guard
    print(f"gen_diar_baseline: missing dependency: {e}", file=sys.stderr)
    sys.exit(2)


def _segments(model, path):
    out = model.diarize(audio=[path], batch_size=1)
    rows = []
    for s in out[0]:
        start, end, spk = s.split()
        rows.append([float(spk.split("_")[-1]), float(start), float(end)])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".nemo path or HF id")
    ap.add_argument("--audio", required=True, help="16 kHz mono wav")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.model.endswith(".nemo"):
        m = SortformerEncLabelModel.restore_from(args.model, map_location="cpu")
    else:
        m = SortformerEncLabelModel.from_pretrained(args.model, map_location="cpu")
    m.eval()
    m.preprocessor.featurizer.dither = 0.0

    y, sr = sf.read(args.audio, dtype="float32")
    if y.ndim != 1 or sr != 16000:
        sys.exit(f"gen_diar_baseline: {args.audio} must be 16 kHz mono (got sr={sr}, shape={y.shape})")
    x = torch.from_numpy(y)[None]
    n = torch.tensor([len(y)])

    results = {}
    for name, streaming in (("offline", False), ("stream", True)):
        m.streaming_mode = streaming
        with torch.no_grad():
            preds = m.forward(x, n)  # [1, T, n_spk]
        results[f"{name}_probs"] = preds[0].numpy().T.copy()  # [n_spk, T]
        results[f"{name}_segs"] = _segments(m, args.audio)
        print(f"{name}: probs {results[name + '_probs'].shape}, "
              f"{len(results[name + '_segs'])} segments")

    w = gguf.GGUFWriter(args.output, "parakeet-diar-baseline")
    w.add_tensor("audio", np.ascontiguousarray(y, dtype=np.float32))
    for k, v in results.items():
        w.add_tensor(k, np.ascontiguousarray(v, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

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
* ``stream_probs_<mode>`` / ``stream_segs_<mode>``
                                       the same for each low-latency streaming
                                       mode of the model card (LATENCY_MODES)
                                       listed in --modes (default: all). NeMo
                                       runs these slowly on CPU.

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


# Streaming configurations from the Nemotron-3-Diarization model card, in 80 ms
# encoder frames: (spkcache_len, fifo_len, chunk_len, chunk_right_context,
# spkcache_update_period). Latency = (chunk_len + right_context) * 80 ms.
LATENCY_MODES = {
    "low": (264, 264, 9, 4, 222),         # 1.04 s
    "very_low": (264, 264, 6, 2, 222),    # 0.64 s
    "ultra_low": (264, 264, 3, 1, 222),   # 0.32 s
}


def _set_streaming(model, spkcache, fifo, chunk, right, update):
    sm = model.sortformer_modules
    sm.spkcache_len = spkcache
    sm.fifo_len = fifo
    sm.chunk_len = chunk
    sm.chunk_right_context = right
    sm.spkcache_update_period = update
    model._check_streaming_parameters()


def _diarize(model, path):
    """One diarize() run: (probs [n_spk, T], segments [N, 3]) as NeMo returns them."""
    lines, preds = model.diarize(audio=[path], batch_size=1, include_tensor_outputs=True)
    rows = []
    for s in lines[0]:
        start, end, spk = s.split()
        rows.append([float(spk.split("_")[-1]), float(start), float(end)])
    probs = preds[0][0].detach().cpu().numpy().T.copy()
    return probs, np.asarray(rows, dtype=np.float32).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".nemo path or HF id")
    ap.add_argument("--audio", required=True, help="16 kHz mono wav")
    ap.add_argument("--output", required=True)
    ap.add_argument("--modes", default=",".join(LATENCY_MODES),
                    help="comma-separated low-latency modes to capture "
                         f"({', '.join(LATENCY_MODES)}); empty for none")
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

    sm = m.sortformer_modules
    saved = (sm.spkcache_len, sm.fifo_len, sm.chunk_len, sm.chunk_right_context,
             sm.spkcache_update_period)
    runs = [("offline", False, None), ("stream", True, saved)]
    for k in filter(None, args.modes.split(",")):
        runs.append((f"stream_{k}", True, LATENCY_MODES[k]))

    results = {}
    for name, streaming, cfg in runs:
        m.streaming_mode = streaming
        if cfg is not None:
            _set_streaming(m, *cfg)
        # Keys: offline_probs, stream_probs, stream_probs_low, ...
        pk, sk = (f"{name}_probs", f"{name}_segs") if "_" not in name else \
            (name.replace("stream_", "stream_probs_"), name.replace("stream_", "stream_segs_"))
        with torch.no_grad():
            results[pk], results[sk] = _diarize(m, args.audio)
        print(f"{name}: probs {results[pk].shape}, {len(results[sk])} segments")

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

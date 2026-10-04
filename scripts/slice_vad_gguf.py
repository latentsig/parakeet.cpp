#!/usr/bin/env python3
"""Cut a VAD-only GGUF out of an Ultra or Redux parakeet GGUF.

moondream/parakeet-ultra and -redux carry a voice-activity head on the
encoder's subsampler output. The VAD path reads only the log-mel filterbank and
window, the subsampler (`encoder.pre_encode.*`) and the head (`vad_head.*`).
This script copies exactly those tensors, byte for byte, plus the config keys
that path reads. Nothing is requantized, so the slice produces the same VAD
output as its parent.

The output sets `parakeet.arch` to `vad`. The loader needs that marker: a
normal file has an encoder, a decoder and a vocabulary, and the slice has none
of them. With the marker, `parakeet-cli vad` and the C-API VAD calls load the
file, and every other entry point refuses it with a clear message.
`general.architecture` stays `parakeet`.

Provenance keys written (all strings or integers):
  parakeet.vad_only.parent_name     general.name of the parent
  parakeet.vad_only.parent_file     file name of the parent
  parakeet.vad_only.parent_sha256   SHA-256 of the parent file
  parakeet.vad_only.parent_bytes    size of the parent file
  parakeet.vad_only.parent_arch     parakeet.arch of the parent (for example tdt)

    python scripts/slice_vad_gguf.py ultra-q8_0.gguf ultra-vad.gguf
"""
import argparse
import hashlib
import pathlib
import sys

try:
    import gguf
except ImportError as e:  # pragma: no cover - env guard
    print(f"slice_vad_gguf: missing dependency: {e}", file=sys.stderr)
    sys.exit(2)

TENSOR_PREFIXES = ("vad_head.", "encoder.pre_encode.", "preprocessor.featurizer.")
KEY_PREFIXES = ("parakeet.encoder.", "parakeet.preprocessor.", "parakeet.vad.")
# Packed ternary tensors cannot appear on the VAD path; refuse them loudly.
PACKED_SUFFIXES = (".weight_packed", ".weight_scales")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def field_value(f):
    return f.contents()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="Ultra or Redux GGUF with a VAD head")
    ap.add_argument("dst", help="output VAD-only GGUF")
    a = ap.parse_args()

    r = gguf.GGUFReader(a.src)
    flds = r.fields
    present = flds.get("parakeet.vad.present")
    if present is None or not bool(field_value(present)):
        sys.exit("slice_vad_gguf: source has no VAD head (parakeet.vad.present is not true)")
    arch = flds.get("parakeet.arch")
    if arch is not None and str(field_value(arch)) == "vad":
        sys.exit("slice_vad_gguf: source is already a VAD-only file")

    tensors = [t for t in r.tensors if t.name.startswith(TENSOR_PREFIXES)]
    names = {t.name for t in tensors}
    for need in ("vad_head.proj.weight", "vad_head.proj.bias", "vad_head.ctx.weight", "vad_head.ctx.bias",
                 "vad_head.out.weight", "vad_head.out.bias", "preprocessor.featurizer.fb",
                 "preprocessor.featurizer.window", "encoder.pre_encode.out.weight"):
        if need not in names:
            sys.exit(f"slice_vad_gguf: source lacks tensor {need}")
    for t in tensors:
        if t.name.endswith(PACKED_SUFFIXES):
            sys.exit(f"slice_vad_gguf: packed tensor {t.name} on the VAD path; not supported")

    parent_name = str(field_value(flds["general.name"])) if "general.name" in flds else pathlib.Path(a.src).name
    w = gguf.GGUFWriter(a.dst, "parakeet")
    w.add_name(parent_name + " (VAD only)")
    w.add_string("parakeet.arch", "vad")
    for k, f in flds.items():
        if not k.startswith(KEY_PREFIXES):
            continue
        # Copy with the original GGUF value type so the C++ readers see the same kind.
        vt = f.types[0]
        sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(k, field_value(f), vt, sub_type=sub)
    w.add_string("parakeet.vad_only.parent_name", parent_name)
    w.add_string("parakeet.vad_only.parent_file", pathlib.Path(a.src).name)
    w.add_string("parakeet.vad_only.parent_sha256", sha256_file(a.src))
    w.add_uint64("parakeet.vad_only.parent_bytes", pathlib.Path(a.src).stat().st_size)
    w.add_string("parakeet.vad_only.parent_arch", str(field_value(arch)) if arch is not None else "")
    for t in tensors:
        w.add_tensor(t.name, t.data, raw_shape=t.data.shape, raw_dtype=t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    n = sum(t.n_bytes for t in tensors)
    print(f"wrote {a.dst}: {len(tensors)} tensors, {n} tensor bytes, from {a.src}")


if __name__ == "__main__":
    main()

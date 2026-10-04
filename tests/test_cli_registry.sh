#!/bin/sh
# parakeet-cli enroll / registry / scene against real voice-detect GGUFs.
# usage: test_cli_registry.sh <parakeet-cli> <wav>
#   PARAKEET_TEST_VD_GGUF      speaker encoder (required, else skip 77)
#   PARAKEET_TEST_VD_ALT_GGUF  optional: other family, same embedding size
set -u
CLI=$1; WAV=$2
[ -n "${PARAKEET_TEST_VD_GGUF:-}" ] || exit 77
T=$(mktemp -d) || exit 1
trap 'rm -rf "$T"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

"$CLI" enroll --model "$PARAKEET_TEST_VD_GGUF" --name ada --input "$WAV" --registry "$T/r.bin" >/dev/null || fail enroll
"$CLI" registry "$T/r.bin" | grep -q '^format version: 2$' || fail "enroll did not write v2"
"$CLI" registry "$T/r.bin" | grep -q '^encoder family: voicedetect:' || fail "no family"
"$CLI" registry "$T/r.bin" | grep -q '^encoder weights: sha256:' || fail "no weights"
if [ -n "${PARAKEET_TEST_VD_ALT_GGUF:-}" ]; then
  "$CLI" enroll --model "$PARAKEET_TEST_VD_ALT_GGUF" --name bob --input "$WAV" --registry "$T/r.bin" 2>"$T/err" \
    && fail "enroll with another family was accepted"
  grep -q 'encoder family' "$T/err" || fail "no family message"
  # a v1 registry (no fingerprint) is refused for enrolment and is never stamped silently
  python3 - "$T/v1.bin" <<'PY'
import struct, sys
d = 192
open(sys.argv[1], "wb").write(b"PKSR" + struct.pack("<IiI", 1, d, 1) + struct.pack("<I", 3) + b"ada" +
                              struct.pack("<i", 1) + struct.pack("<%df" % d, *([1.0] + [0.0] * (d - 1))))
PY
  "$CLI" registry "$T/v1.bin" | grep -q '^format version: 1$' || fail "v1 not read"
  "$CLI" enroll --model "$PARAKEET_TEST_VD_GGUF" --name bob --input "$WAV" --registry "$T/v1.bin" 2>/dev/null \
    && fail "enroll into an unfingerprinted registry was accepted"
  "$CLI" registry "$T/v1.bin" | grep -q '^format version: 1$' || fail "v1 changed by a refused enroll"
  "$CLI" registry "$T/v1.bin" --restamp --encoder "$PARAKEET_TEST_VD_GGUF" >/dev/null || fail restamp
  "$CLI" registry "$T/v1.bin" | grep -q '^format version: 2$' || fail "restamp did not write v2"
  "$CLI" registry "$T/v1.bin" --restamp --encoder "$PARAKEET_TEST_VD_GGUF" >/dev/null 2>&1 && fail "second restamp accepted"
fi
echo "test_cli_registry: PASS"

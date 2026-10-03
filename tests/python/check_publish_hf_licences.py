#!/usr/bin/env python3
"""Check scripts/publish_hf.py licence handling.

* every model id the script can publish has an explicit licence (no default);
* an unknown id raises instead of getting a default licence;
* the script's LICENSES table equals the table in docs/licenses.md;
* the generated collection card keeps the per-model licences, the required
  notices and the sections for files that the script does not convert.

No network, no conversion. Exit codes (ctest convention): 0 = pass, 1 = fail.
"""
import importlib.util
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location("publish_hf", root / "scripts" / "publish_hf.py")
ph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ph)

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


# 1. Explicit licence for every known id.
for problem in ph.check_licences():
    failures.append(problem)

# 2. No silent default.
try:
    ph._license("someone/unknown-model")
    failures.append("unknown model id did not raise")
except KeyError:
    pass

# 3. Script table == docs table.
doc_rows = {}
for line in (root / "docs" / "licenses.md").read_text(encoding="utf-8").splitlines():
    m = re.match(r"\| `([^`]+)` \| ([^|]+?) \| <([^>]+)> \| ([^|]*?) \|$", line)
    if m:
        doc_rows[m.group(1)] = (m.group(2), m.group(3), m.group(4))
script_rows = {k: (v[1], v[2], ph.LICENSE_NOTICES.get(k, "")) for k, v in ph.LICENSES.items()}
check(doc_rows == script_rows, "LICENSES differs from docs/licenses.md: "
      f"{sorted(set(doc_rows.items()) ^ set(script_rows.items()))}")

# 4. Generated card content.
card = ph.build_collection_card(ph.ALL_MODELS, ph.DEFAULT_VARIANTS, {}, ph.DEFAULT_COLLECTION_REPO)
front = card.split("---")[1]
check("license: other" in front, "front matter is not `license: other`")
check("license_name: mixed-per-model-see-license-section" in front, "license_name missing")
for needle in (
    "Licensed by NVIDIA Corporation under the NVIDIA Open Model License",
    "Copyright (c) 2020-present Silero Team",
    "https://openmdw.ai/license/1-1/",
    "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
    "### nemotron-3-diarization", "### ultra and redux (Moondream)", "### silero-vad",
    "### nemotron-3.5-asr-streaming-0.6b",
    "snakers4/silero-vad", "moondream/parakeet-redux", "nvidia/Nemotron-3-Diarization",
):
    check(needle in card, f"card is missing: {needle}")
section = re.search(r"^### realtime_eou_120m-v1\n\n(.*?)\n\n", card, re.S | re.M)
check(section is not None and "NVIDIA Open Model License" in section.group(1),
      "realtime_eou_120m-v1 section lacks the NVIDIA Open Model License")
check("`realtime_eou_120m-v1-*`: [NVIDIA Open Model License]" in card,
      "License section lacks the realtime_eou_120m-v1 bullet")
# CC-BY-4.0 must not be claimed for the non-CC models.
for bad in ("`realtime_eou_120m-v1-*`: [CC-BY-4.0]", "`nemotron-3.5-asr-streaming-0.6b-*`: [CC-BY-4.0]"):
    check(bad not in card, f"wrong licence in card: {bad}")

if failures:
    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)
    sys.exit(1)
print("check_publish_hf_licences: PASS")

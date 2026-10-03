### nemotron-3-diarization

Source: [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) · Speaker diarization (Sortformer, up to 8 speakers, streaming speaker cache) · License: [OpenMDW 1.1](https://openmdw.ai/license/1-1/)

| File | Variant | Size | Segments vs NeMo |
|---|---|---:|---:|
| `nemotron-3-diarization-f16.gguf` ← **recommended** | F16 | 200.7 MB | identical |
| `nemotron-3-diarization-q8_0.gguf` | Q8_0 | 108.7 MB | identical on 2 of 3 clips, 99.5% of frames on the third |

> Checked against NeMo on a 23.6 s and a 68.5 s two-speaker clip (offline and streaming: every segment identical, to the 10 ms frame) and a 12.3 min three-speaker clip (F16 100%, Q8_0 99.9% of speech frames). Q8_0 can flip a frame whose probability sits right at the 0.5 threshold, splitting a segment (99.5% of frames on a 31.5 s clip); use F16 when segment-exact output matters. Answers "who spoke when"; pair it with any ASR model above for speaker-attributed transcripts. See [diarization.md](https://github.com/mudler/parakeet.cpp/blob/master/docs/diarization.md).

```bash
build/examples/cli/diarize models/nemotron-3-diarization-f16.gguf meeting.wav
```

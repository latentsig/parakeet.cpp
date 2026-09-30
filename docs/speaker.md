# Speaker identification

parakeet.cpp can put a name on a diarized speaker. You enroll a few people
from short clips, and the scene stream and the speaker-attributed ASR output
then say `Ada:` where they would otherwise say `Speaker 0:`.

It runs a speaker-embedding model from
[voice-detect.cpp](https://github.com/localai-org/voice-detect.cpp), built in as a
static library (`PARAKEET_WITH_VOICEDETECT`, on by default, the same way
ced.cpp is built in for sound events). `pk::SpeakerEncoder`
(`src/speaker_encoder.hpp`) is the only parakeet code that talks to it.

## What it does and does not do

It does:

- Turn a clip of speech into an L2-normalized embedding, and keep enrolled
  voices in a registry (one centroid per name).
- Give each diarization slot a name by embedding the slot's clean audio and
  matching it against the registry.
- Leave a slot unnamed when nobody in the registry matches well enough.

It does not:

- Find or diarize speakers by itself. It names the slots that the diarization
  model (Sortformer) produces, so it needs `--diar`.
- Resolve overlapped speech. Time where two speakers overlap is skipped when a
  slot's voice is built. It is not attributed to anyone.
- Learn new voices on the fly. The registry is only changed by enrolling.

## Which GGUFs work

Use a speaker-encoder GGUF from
[`mudler/voice-detect-gguf`](https://huggingface.co/mudler/voice-detect-gguf).
The four speaker encoders are:

| Model | Embedding size | f32 GGUF size | Starting threshold |
| --- | --- | --- | --- |
| WeSpeaker ResNet34 | 256 | 26.5 MB | 0.5 |
| CAM++ (3D-Speaker, zh-cn) | 192 | 27.7 MB | 0.5 |
| ECAPA-TDNN (SpeechBrain, VoxCeleb) | 192 | 83.2 MB | 0.7 |
| ERes2Net (3D-Speaker, base) | 512 | 39.5 MB | not measured |

Start with WeSpeaker ResNet34: it kept the two voices furthest apart in the
measurements below. The starting threshold is the `accept_threshold` to begin
with (`--speaker-threshold` on the command line). The default is 0.5, which is
right for WeSpeaker and CAM++ here, but ECAPA scored a voice that was not
enrolled at 0.566, so it needs about 0.7 (0.13 above that impostor and 0.26
below the lowest genuine ECAPA score). These numbers come from one fixture,
where the enrollment clips and the test audio share a recording and genuine
scores were 0.92 to 0.98. Expect lower genuine scores when enrollment and test
audio come from different sessions or microphones, and check the threshold on
your own audio.

The repository also holds age, gender and emotion models. They are not
speaker encoders and cannot be used here (`SpeakerEncoder::load` returns null
for a GGUF with no speaker embedding). f16 and q8_0 files are published as
well. Sizes are for the f32 files. ERes2Net has not been run through any of
the tests here.

A registry belongs to the encoder that made it. The embedding sizes differ, and
even two encoders with the same size do not share a space, so enroll again if
you switch models. `scene` checks the size and stops if it does not match; it
cannot tell two encoders of the same size apart.

The speaker-model weights have their own licences (WeSpeaker, 3D-Speaker and
SpeechBrain each publish theirs). voice-detect.cpp's own licence does
not cover the weights. Read the licence of the checkpoint you
ship.

## Enroll

```
parakeet-cli enroll --model <speaker.gguf> --name <name> \
    --input <wav> [--input <wav> ...] --registry <file>
```

Each `--input` is one clip. The registry file is created when it is missing
and added to when it exists. Nothing is written unless every clip embedded.
The number printed is the clips enrolled by that command, not the total for
that name.

The example below uses three clips cut out of
`tests/fixtures/two_speakers.wav` (voice A at 0.6 to 4.6 s and 14.9 to 18.5 s,
voice B at 6.9 to 10.9 s), enrolled with WeSpeaker ResNet34. Real output:

```
$ parakeet-cli enroll --model wespeaker_resnet34_f32.gguf --name Ada --input a.wav --registry reg.bin
enrolled Ada (1 clip(s)), registry has 1 speaker(s)
$ parakeet-cli enroll --model wespeaker_resnet34_f32.gguf --name Ben --input b.wav --registry reg.bin
enrolled Ben (1 clip(s)), registry has 2 speaker(s)
$ parakeet-cli enroll --model wespeaker_resnet34_f32.gguf --name Ada --input a2.wav --input a.wav --registry reg.bin
enrolled Ada (2 clip(s)), registry has 2 speaker(s)
```

Enrolling a name again refines that voice (the centroid moves) and does not
add a second speaker. The same voice enrolled under two names comes out
unknown: both names match about equally well, so neither beats the other by
the margin. Names are compared exactly, so near-duplicate names (`Ada` and
`ada`, or a trailing space) count as two speakers.

## Scene with names

```
parakeet-cli scene --model <asr.gguf> --diar <diar.gguf> \
    --speakers <speaker.gguf> --registry <file> [--speaker-threshold F] \
    --input <wav>
```

`--speakers` needs `--diar` and `--registry`. Real output on the same fixture
(110m TDT, Sortformer, WeSpeaker; the enrollment clips come from the same
recording):

```
[00:00.4 - 00:05.4]  Ada: mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.
[00:06.8 - 00:10.8]  Ben: Well, I don't wish to see it any more, observed Phoebe, turning away her eyes.
[00:11.4 - 00:13.6]  Ben: It is certainly very like the old portrait.
[00:14.8 - 00:18.5]  Ada: Nor is mister Quilter's manner less interesting than his matter.
[00:19.9 - 00:20.0]  Ben: Well,
[00:20.4 - 00:23.3]  Ben: I don't wish to see it any more, observed Phoebe, turning away her
```

With `--json` each update carries a `"names"` map (empty, `{}`, until a slot
is seen), for example at the end of the file:

```
"names":{"0":{"name":"Ada","score":0.9752},"1":{"name":"Ben","score":0.9681}}
```

The scores in this sample come from enrolling with the whole clips used in the
example above (Ada from `a.wav` and `a2.wav`, Ben from `b.wav`), so they differ
a little from the numbers in the measured section, where each voice is enrolled
from one clip.

An unnamed slot still renders as `Speaker N:`.

Errors exit with a one-line message: 2 for a usage problem (`--speakers`
without `--diar` or without `--registry`, a bad `--speaker-threshold`), 1 for a
runtime problem (missing or invalid registry file, registry from a model with a
different embedding size, model that fails to load).

## The timing rule

A slot needs some clean audio before it can be named (2 s by default). That
audio is collected while the slot is still talking, not only after it pauses,
so a speaker who talks without a break is named during that first turn. Still,
a word can be committed before its slot is identified. Such a word keeps the
label it had when it was committed (empty name, rendered as `Speaker N`), and
it is not rewritten later. The `names` map in each update, and `active`, carry
the current identity of each slot. In the run above every utterance was named
from its first word, but that is one fixture and it depends on how early the
speakers start talking.

## Defaults

| Option | Default | Meaning |
| --- | --- | --- |
| `min_voice_sec` | 2.0 | clean audio a slot needs before it is embedded |
| `refresh_sec` | 3.0 | new clean audio that triggers another embedding |
| `max_voice_sec` | 10.0 | the newest audio kept per slot for embedding |
| `accept_threshold` | 0.5 | minimum cosine to take a name |
| `margin` | 0.05 | best match must beat the runner-up by this much |

Change them in C++ through `SceneParts::speaker_opts` (`pk::SpeakerIdOpts`), in
the C-API through the `speaker_*` fields of `parakeet_scene_opts`, and on the
command line with `--speaker-threshold` (only `accept_threshold`).

`accept_threshold` is a starting point, not a tuned value. It depends on the
encoder: see the starting threshold column in "Which GGUFs work" and the
numbers below.

## C-API (ABI v9)

Additive: no earlier signature changed, LocalAI does not use these yet. A
speaker GGUF loads through `parakeet_capi_load` into a context of kind
`PARAKEET_MODEL_KIND_SPEAKER` (4, from `parakeet_capi_model_kind`).

```
parakeet_capi_speaker_dim                        # embedding size, -1 if not a speaker ctx
parakeet_capi_speaker_registry_new / _free / _size / _last_error
parakeet_capi_speaker_enroll                     # embed PCM and add it under a name
parakeet_capi_speaker_registry_save / _load      # binary file
parakeet_capi_speaker_identify_pcm_json          # {"name":"alice","score":0.71}
parakeet_capi_scene_stream_begin_speaker         # scene stream with a speaker ctx + registry
parakeet_capi_transcribe_and_diarize_named_json  # offline speaker-attributed ASR with names
```

`parakeet_capi_scene_stream_begin_speaker` takes the same arguments as
`parakeet_capi_scene_stream_begin` plus a speaker ctx and a registry. The
registry is borrowed: keep it alive and unchanged while the stream runs.

JSON fields: each utterance, word and speaker segment gets `"name"` and
`"name_score"` (empty name and 0.0000 mean unknown), and the top level gets
`"names"`, a map from slot to `{"name","score"}`. In a scene stream with a
speaker part these fields are there from the first document on: `"names"` is
`{}` until diarization has seen a slot, so the shape of the document does not
change during the stream. Without a speaker part none of them appear. The
offline named document is the SAS document plus those fields.

## Devices and threads

voice-detect keeps its own backend and device selection, like ced.cpp does with
`CED_DEVICE`: `VOICEDETECT_DEVICE` picks the device and `VOICEDETECT_THREADS`
the CPU thread count. They are read separately from `PARAKEET_DEVICE`. An
embedded voice-detect build does not apply voice-detect's own CUDA and cuDNN
ggml patch. That does not matter on CPU.

## The registry file

A small binary blob (version 1) written by `enroll` and by
`parakeet_capi_speaker_registry_save`. It is not a stable interchange format
yet: do not depend on it outside parakeet.cpp.

`SpeakerIdentifier::update` has an internal contract about which segments are
listed as still open (see `src/speaker_identifier.hpp`). Callers of the scene
stream do not need to care about it.

## What has been measured

Everything here is one fixture: `tests/fixtures/two_speakers.wav`, two
read-speech LibriSpeech voices (1272 and 2086) alternating A-B-A-B. Nothing
else has been run.

Clip-to-clip cosine between two clips of one voice, and between clips of two
different voices. Each clip is a whole turn, as in
`tests/test_speaker_encoder.cpp`: voice A 0.6 to 5.4 s and 14.9 to 18.7 s,
voice B 6.9 to 10.7 s and 20.2 to 23.5 s. "Same voice" averages the A pair and
the B pair, "different voices" averages the four A-B pairs. The design spike
measured 2 s windows of the same file instead, and shorter windows give lower
numbers (for example WeSpeaker 0.585 same voice), so the two sets differ but
do not disagree.

| Encoder | same voice | different voices |
| --- | --- | --- |
| WeSpeaker ResNet34 | 0.869 | -0.012 |
| CAM++ | 0.894 | 0.383 |
| ECAPA-TDNN | 0.934 | 0.558 |

In `tests/test_speaker_identify.cpp` the scene stream names both voices right
with all three encoders and the default thresholds, with the two voices
enrolled in the reverse of the order they speak, so slot numbers cannot be
matched to registry order. The enrollment clips are cut from the same
recording that is then streamed, so these scores are optimistic. Genuine slot
scores:

| Encoder | genuine slot scores |
| --- | --- |
| WeSpeaker ResNet34 | 0.922 to 0.968 |
| CAM++ | 0.940 to 0.983 |
| ECAPA-TDNN | 0.959 to 0.985 |

An impostor voice (the voice that is not in the registry) scored -0.019 with
WeSpeaker, 0.374 with CAM++ and 0.546 to 0.566 with ECAPA. The default
threshold of 0.5 is fine for WeSpeaker and CAM++ on this fixture, but ECAPA
would admit that impostor. That is why the test uses `accept_threshold` 0.7 for
its unenrolled-voice check, and why the right threshold is encoder specific.

With ASR on as well (`PARAKEET_TEST_GGUF`), the same test streams the fixture
and checks that utterances for both slots carry the right name and that no
utterance ever carries the other voice's name.

### Not measured yet (open work)

The accuracy beyond this fixture is unknown. Still to do:

- a third voice, and more than two speakers in one recording;
- noisy audio, and audio with overlapping speech;
- telephone-band or other far-from-read-speech audio;
- enrollment from a different session or microphone than the test audio
  (here enrollment and test share a recording);
- a threshold sweep per encoder against a labelled set, so the defaults come
  from data.

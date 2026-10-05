# Trim regression follow-up: tables

Tables behind the section "Trimming: word error rate, enlarged measurement" of
[docs/vad-benchmarks.md](../../../../docs/vad-benchmarks.md). Every table is copied from a file in
[../../trim_regression/results](../../trim_regression/results) (named in each heading); the scripts
and the way to regenerate them are in [../../trim_regression](../../trim_regression/README.md).
`results/tables.txt` in this directory is the first run (small sets) and is not changed. Its Redux
rows (talks 4.39 to 4.51, pink noise 0 dB 12.35 to 13.29) were noise.

Deltas are trim 0.3 minus old cuts (`--vad-trim 0`) in WER points, with a 95 percent paired
bootstrap interval. T0 is the old cuts, Tx is trim x seconds.

## Trim 0.3 against the old cuts (sdi.txt, grid_redux_*.md, cand_*_heldout.md, cand_*_tune.md, grid_*_talks_all.txt)

| Set | Redux head | Ultra head | v3 + Silero |
| --- | ---: | ---: | ---: |
| 9 talks (21,540 words) | +0.08 (-0.03..+0.20) | +0.01 (-0.06..+0.08) | +0.01 (-0.06..+0.10) |
| 5 held-out talks (11,490 words) | +0.16 (-0.01..+0.36) | +0.02 (-0.06..+0.11) | +0.03 (-0.08..+0.20) |
| 4 tune talks (10,050 words) | -0.01 (-0.14..+0.13) | +0.00 (-0.10..+0.12) | -0.02 (-0.09..+0.04) |
| noisy speech: white 5, pink 5, pink 0 (Redux 17,946 words, others 15,642) | -0.12 (-0.57..+0.45) | -0.27 (-0.57..+0.01) | -0.26 (-0.57..+0.02) |
| pink 0 dB alone, Redux (5,982 words) | -0.15 (-0.90..+0.71) | | |
| clean speech, held out (1,350 words) | -0.37 (-0.80..-0.07) | -0.22 (-0.76..+0.28) | -3.19 (-7.76..-0.14) |

Held-out clean WER for v3 + Silero: 6.37 with the old cuts, 3.19 with trim 0.3.

## Redux, trim variants against the old cuts (grid_redux_all.md, 9 talks and all noisy files)

noisy-all is white and pink noise at 20, 10, 5 and 0 dB (47,856 words).

| Variant | talks | noisy-all |
| --- | ---: | ---: |
| T0.1 | +0.03 (-0.12..+0.18) | -0.01 (-0.22..+0.22) |
| T0.2 | +0.02 (-0.13..+0.19) | -0.11 (-0.40..+0.16) |
| T0.3 | +0.08 (-0.03..+0.20) | -0.12 (-0.42..+0.17) |
| T0.5 | -0.02 (-0.08..+0.04) | +0.06 (-0.13..+0.25) |
| T1.0 | -0.00 (-0.01..+0.00) | -0.01 (-0.13..+0.12) |

## Text changes when the trim changes (segchange.txt)

Share of segments whose text changes, and the net change in errors.

| Detector | T0 to T0.3, talks | T0 to T0.3, noisy | T0.3 to T0.5, talks |
| --- | ---: | ---: | ---: |
| Redux | 46 of 296 (15.5%), net +17 | 439 of 860 (51.0%), net -56 | 49 of 296 (16.6%), net -22 |
| Ultra | 30 of 299 (10.0%), net +1 | 149 of 291 (51.2%), net -40 | 29 of 299 (9.7%), net +0 |
| v3 + Silero | 21 of 276 (7.6%), net +2 | 151 of 279 (54.1%), net -32 | 20 of 276 (7.2%), net -3 |

Redux, pad 0.30 to 0.32 s: 40 of 296 segments (13.5%) change text; 0.30 to 0.35 s: 49 of 296
(16.6%). On the 9 talks (weighted from the tune and held-out sets of cand_redux_*.md) the talk WER
is 4.87 with 0.30, 4.82 with 0.32 and 4.80 with 0.35, that is -0.05 and -0.07.

## Where the errors move, Redux, T0 to T0.3 (boundary_T0_T0.3.txt)

Errors by zone of the segment (substitutions, deletions, insertions).

| Set | cut away | start edge | end edge | interior |
| --- | ---: | ---: | ---: | ---: |
| talks | 2 to 1 | 18 to 18 | 21 to 21 | 992 to 1010 (+18) |
| noisy speech | 4 to 5 | 21 to 15 | 31 to 13 | 3208 to 3169 (-39) |

Words cut away on noisy speech that were correct, by trim 0.3 (clip_count.txt): Redux 14 of 47,791,
Ultra 14 of 16,133, v3 19 of 15,506.

## Noise block (noise_seconds_90_inserts.txt, noise_words_20_inserts.md)

Seconds of a 60 s noise block that the decoder gets (90 insert files), and invented words in the
block (20 insert files).

| Detector | T0 | T0.3 | T0.5 | T1.0 | words, T0 | words, T0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Redux | 33.2 | 4.6 | 4.9 | 5.7 | 0 | 0 |
| Ultra | 37.2 | 12.1 | 12.4 | 13.3 | 18 | 0 |
| v3 + Silero | 27.0 | 0.0 | 0.2 | 0.6 | 14 | 0 |

## Detected speech start against the true start (vad_edges.txt, all noisy files)

Median lateness: Redux -5 ms, Ultra 79 ms, Silero 237 ms.

## Other padding rules (per_detector_default_table.md)

Held-out WER (talks, clean and the three noisy conditions together) and the delta against T0.3:
no rule is better than 0.3 outside the intervals.

| Detector | no trim | 0.3 / 0.3 | 0.5 / 0.5 | 0.5 / 0.3 | 0.3 / 0.3, edges of at least 0.5 s only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Redux | 6.34 | 6.40 | 6.32 | 6.33 | 6.31 |
| Ultra | 5.10 | 5.01 | 5.05 | 5.05 | 5.04 |
| v3 + Silero | 5.70 | 5.40 | 5.52 | 5.51 | 5.41 |

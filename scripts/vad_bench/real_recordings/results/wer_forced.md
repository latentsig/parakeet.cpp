# WER (forced), decoding the segments of each system

## forced: ultra ASR model (VAD segmentation from ultra head / Silero / fusion / gate)

| system | WER % [95% CI] | S | D | I | decoded s | per-talk WER % | dWER vs pause20 (pp) | dWER vs forced20 (pp) |
|---|---|---|---|---|---|---|---|---|
| pause20 | 4.19 [3.46, 4.94] | 195 | 103 | 72 | 2885 | 3.3, 2.1, 5.4, 5.5, 3.4 | +0.00 [+0.00, +0.00] | -0.29 [-0.61, +0.00] |
| forced20 | 4.48 [3.76, 5.24] | 202 | 125 | 69 | 2894 | 3.7, 3.2, 4.6, 6.3, 3.6 | +0.29 [-0.00, +0.61] | +0.00 [+0.00, +0.00] |

## forced: redux ASR model (VAD segmentation from redux head / Silero / fusion / gate)

| system | WER % [95% CI] | S | D | I | decoded s | per-talk WER % | dWER vs pause20 (pp) | dWER vs forced20 (pp) |
|---|---|---|---|---|---|---|---|---|
| pause20 | 5.15 [4.39, 5.92] | 255 | 123 | 77 | 2885 | 3.7, 4.8, 5.8, 6.8, 4.2 | +0.00 [+0.00, +0.00] | -0.32 [-0.65, +0.05] |
| forced20 | 5.47 [4.66, 6.28] | 264 | 145 | 74 | 2894 | 3.9, 5.4, 5.8, 7.6, 4.2 | +0.32 [-0.05, +0.65] | +0.00 [+0.00, +0.00] |

talk order: GaryFlake-merged, RobertGupta-merged, EricMead_2009P_EricMead-merged, DanBarber-merged, MichaelSpecter-merged

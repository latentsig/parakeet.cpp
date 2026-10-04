
### redux
| rule | speech F1 clean | speech F1 noisy (white+pink 20..0 dB) | noise-only FA frames % | embedded-stretch FA frames % | noise-only FA regions per clip |
|---|---|---|---|---|---|
| head thr 0.5 | 95.4 (P 98.6, R 92.3) | 94.7 | 97.8 | 55.2 | 2.1 |
| head thr 0.7 | 94.8 (P 98.9, R 91.0) | 93.9 | 76.5 | 10.1 | 13.8 |
| head thr 0.9 | 93.9 (P 99.3, R 89.1) | 92.0 | 16.3 | 0.1 | 12.9 |
| head thr 0.97 | 92.5 (P 99.5, R 86.4) | 89.4 | 1.0 | 0.0 | 1.1 |
| head thr 0.5, min_speech 0.25 s | 95.4 (P 98.7, R 92.3) | 94.7 | 97.7 | 51.7 | 2.2 |
| head thr 0.5, min_speech 0.5 s | 95.4 (P 98.7, R 92.3) | 94.9 | 97.4 | 43.3 | 2.2 |
| head thr 0.5, min_speech 1.0 s | 94.6 (P 98.7, R 90.8) | 94.1 | 96.6 | 29.0 | 2.2 |
| head thr 0.5 + energy gate (frame >= file P95 - 15 dB) | 86.1 (P 100.0, R 75.6) | 91.7 | 97.8 | 40.5 | 2.1 |
| head thr 0.5 + energy gate (frame >= file P95 - 25 dB) | 92.1 (P 99.9, R 85.4) | 94.3 | 97.8 | 53.2 | 2.1 |
| head thr 0.5 + energy gate (frame >= file P95 - 35 dB) | 94.6 (P 99.7, R 90.1) | 94.7 | 97.8 | 55.2 | 2.1 |
| head thr 0.9 + energy gate (frame >= file P95 - 25 dB) | 91.4 (P 99.9, R 84.1) | 91.8 | 16.3 | 0.1 | 12.9 |
| head thr 0.5, keep run only if median logit >= 2.5 | 95.4 (P 98.7, R 92.3) | 94.7 | 0.1 | 0.3 | 0.0 |
| head thr 0.5, keep run only if median logit >= 3.5 | 94.9 (P 98.7, R 91.4) | 93.7 | 0.0 | 0.2 | 0.0 |
| head thr 0.5, keep run only if median logit >= 5.0 | 93.5 (P 98.7, R 88.8) | 90.2 | 0.0 | 0.1 | 0.0 |
| Silero 0.5 AND head 0.5 | 93.4 (P 99.4, R 88.2) | 91.8 | 0.0 | 0.0 | 0.0 |
| Silero 0.5 AND head 0.9 | 93.0 (P 99.5, R 87.4) | 90.8 | 0.0 | 0.0 | 0.0 |
| two-stage (Silero + head extend/fill, fusion rule 6) | 95.4 (P 98.6, R 92.4) | 94.8 | 0.0 | 0.0 | 0.0 |
| Silero 0.5 alone (reference, its own post) | 93.9 (P 99.1, R 89.2) | 92.4 | 0.0 | 0.0 | 0.0 |

### ultra
| rule | speech F1 clean | speech F1 noisy (white+pink 20..0 dB) | noise-only FA frames % | embedded-stretch FA frames % | noise-only FA regions per clip |
|---|---|---|---|---|---|
| head thr 0.5 | 95.0 (P 98.5, R 91.7) | 93.6 | 99.4 | 17.3 | 1.1 |
| head thr 0.7 | 94.2 (P 98.8, R 90.0) | 91.8 | 94.2 | 3.6 | 5.1 |
| head thr 0.9 | 92.4 (P 99.2, R 86.5) | 87.5 | 25.8 | 0.0 | 18.2 |
| head thr 0.97 | 90.0 (P 99.5, R 82.1) | 80.6 | 0.5 | 0.0 | 0.7 |
| head thr 0.5, min_speech 0.25 s | 95.0 (P 98.5, R 91.7) | 93.6 | 99.4 | 15.4 | 1.1 |
| head thr 0.5, min_speech 0.5 s | 95.0 (P 98.5, R 91.7) | 93.5 | 99.4 | 11.9 | 1.1 |
| head thr 0.5, min_speech 1.0 s | 93.8 (P 98.6, R 89.6) | 91.9 | 99.4 | 6.9 | 1.1 |
| head thr 0.5 + energy gate (frame >= file P95 - 15 dB) | 86.0 (P 100.0, R 75.4) | 90.9 | 99.4 | 17.3 | 1.1 |
| head thr 0.5 + energy gate (frame >= file P95 - 25 dB) | 92.0 (P 99.9, R 85.2) | 93.3 | 99.4 | 17.3 | 1.1 |
| head thr 0.5 + energy gate (frame >= file P95 - 35 dB) | 94.4 (P 99.7, R 89.6) | 93.6 | 99.4 | 17.3 | 1.1 |
| head thr 0.9 + energy gate (frame >= file P95 - 25 dB) | 90.4 (P 99.9, R 82.5) | 87.3 | 25.8 | 0.0 | 18.2 |
| head thr 0.5, keep run only if median logit >= 2.5 | 94.5 (P 98.5, R 90.8) | 90.3 | 0.0 | 0.0 | 0.0 |
| head thr 0.5, keep run only if median logit >= 3.5 | 93.6 (P 98.6, R 89.0) | 85.3 | 0.0 | 0.0 | 0.0 |
| head thr 0.5, keep run only if median logit >= 5.0 | 89.5 (P 98.6, R 82.0) | 67.2 | 0.0 | 0.0 | 0.0 |
| Silero 0.5 AND head 0.5 | 93.3 (P 99.4, R 87.8) | 91.2 | 0.0 | 0.0 | 0.0 |
| Silero 0.5 AND head 0.9 | 91.7 (P 99.5, R 85.0) | 86.7 | 0.0 | 0.0 | 0.0 |
| two-stage (Silero + head extend/fill, fusion rule 6) | 95.1 (P 98.5, R 92.0) | 94.2 | 0.0 | 0.0 | 0.0 |
| Silero 0.5 alone (reference, its own post) | 93.9 (P 99.1, R 89.2) | 92.4 | 0.0 | 0.0 | 0.0 |

import numpy as np
SR=16000
def rms(x): return float(np.sqrt((x ** 2).mean()) + 1e-12)
def at_db(x, db): return x * (10 ** (db / 20) / rms(x))
def pink(n, r):
    X = np.fft.rfft(r.standard_normal(n)); f = np.arange(len(X)); f[0] = 1
    y = np.fft.irfft(X / np.sqrt(f), n); return y / y.std()
def clicks(n, r):
    x = np.zeros(n); t = 0
    while t < n - 400:
        k = np.arange(400); x[t:t + 400] += r.uniform(0.3, 1) * r.choice([-1, 1]) * np.exp(-k / 40) * np.sin(k * 0.8)
        t += int(r.uniform(0.15, 0.5) * SR)
    return x
def music(n, r):
    t = np.arange(n) / SR; x = np.zeros(n)
    for f0 in (220.0, 261.63, 329.63, 440.0):
        f = f0 * (1 + 0.003 * np.sin(2 * np.pi * r.uniform(4, 6) * t + r.uniform(0, 6)))
        ph = 2 * np.pi * np.cumsum(f) / SR
        for h, a in ((1, 1.0), (2, 0.5), (3, 0.25)): x += a * np.sin(h * ph)
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t + r.uniform(0, 6)) ** 2
    return x * env
def hum(n, r):
    t = np.arange(n) / SR; f = r.choice([50.0, 60.0])
    return sum(np.sin(2 * np.pi * f * h * t + r.uniform(0, 6)) / h for h in (1, 2, 3, 5))
def tone(n, r): return np.sin(2 * np.pi * float(r.choice([440, 1000, 2000])) * np.arange(n) / SR)
def sweep(n, r):
    t = np.arange(n) / SR; T = t[-1]; return np.sin(2 * np.pi * (50 * t + (7500 - 50) * t * t / (2 * T)))
GEN = {"white": lambda n, r: r.standard_normal(n), "pink": pink, "clicks": clicks, "music": music,
       "hum": hum, "tone": tone, "sweep": sweep}

"""Fourier analysis of the SHD input after smoothing the spike trains.

Companion to make_input_spectrum.py (F4.13), which FFTs the raw 1 ms binned population count. Here we first
smooth each trial's population spike train with a Gaussian kernel (a standard firing-rate estimate), then take
the power spectrum, averaged over the whole test set. Smoothing is a low-pass step, so it suppresses the flat
high-frequency (Poisson-like) floor and leaves the slow signal structure that a leaky membrane can integrate.

Population signal per trial: spike count per 1 ms step summed over the 700 input channels (smoothing each channel
then summing is identical to smoothing the sum, since both operations are linear).

Run: python code/analysis_suite/make_input_spectrum_smoothed.py
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S

RROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "code" / "model_perez_nieves"))
from data_gen import open_file

FIGS = REPO / "docs" / "analysis_suite" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
S.apply()

DT = 1e-3
NB_STEPS = 1000
SIGMAS_MS = [2, 5, 10]                      # Gaussian smoothing widths to show
WIN = np.hanning(NB_STEPS)
freqs = np.fft.rfftfreq(NB_STEPS, DT)
band = (freqs > 0) & (freqs <= 200)


def psd(sig):
    sig = sig - sig.mean()
    return np.abs(np.fft.rfft(sig * WIN)) ** 2


def centroid(P):
    return float((freqs[band] * P[band]).sum() / P[band].sum())


print("loading SHD test set ...")
u, t, l = open_file(str(RROOT / "data" / "shd" / "shd_test.h5"))
times = list(t[:]); u._v_file.close()
print(f"  {len(times)} trials")

# build the population signal per trial, then its raw and smoothed PSDs
raw, smoothed = [], {s: [] for s in SIGMAS_MS}
for ti in times:
    idx = np.round(np.asarray(ti) / DT).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < NB_STEPS)]
    sig = np.bincount(idx, minlength=NB_STEPS).astype(np.float64)
    raw.append(psd(sig))
    for s in SIGMAS_MS:
        smoothed[s].append(psd(gaussian_filter1d(sig, sigma=s)))   # sigma in samples = sigma in ms (dt = 1 ms)

P_raw = np.mean(raw, axis=0)
P_sm = {s: np.mean(v, axis=0) for s, v in smoothed.items()}

fig, ax = plt.subplots(figsize=(8.6, 5))
ax.loglog(freqs[band], P_raw[band] / P_raw[band].max(), color="#b8b8b8", lw=1.3,
          label=f"raw spike count (centroid {centroid(P_raw):.1f} Hz)")
ramp = {2: "#f4a3ae", 5: S.HET, 10: "#7a1f2b"}
for s in SIGMAS_MS:
    ax.loglog(freqs[band], P_sm[s][band] / P_sm[s][band].max(), color=ramp[s], lw=1.8,
              label=f"Gaussian smoothed, sigma = {s} ms (centroid {centroid(P_sm[s]):.1f} Hz)")
tau_c = 1.0 / (2 * np.pi * centroid(P_sm[5])) * 1e3
ax.axvline(centroid(P_sm[5]), color=S.CHANCE, ls="--", lw=1.3, alpha=0.8)
ax.set_xlim(1, 200)
ax.set_xlabel("frequency (Hz, log scale)")
ax.set_ylabel("normalised input power (log scale)")
ax.set_title("Input power spectrum of the smoothed spike trains, SHD test set")
ax.legend(loc="lower left", fontsize=8)
S.save(fig, FIGS / "F4_15_input_spectrum_smoothed.png")

out = {"n_trials": len(times), "sigmas_ms": SIGMAS_MS,
       "centroid_raw_hz": centroid(P_raw),
       "centroid_smoothed_hz": {str(s): centroid(P_sm[s]) for s in SIGMAS_MS},
       "tau_at_smoothed5_centroid_ms": tau_c}
json.dump(out, open(REPO / "docs" / "analysis_suite" / "input_spectrum_smoothed.json", "w"), indent=1)
print("centroids:", {"raw": round(centroid(P_raw), 1),
                     **{f"sm{s}": round(centroid(P_sm[s]), 1) for s in SIGMAS_MS}})
print("wrote F4_15_input_spectrum_smoothed.png and input_spectrum_smoothed.json")

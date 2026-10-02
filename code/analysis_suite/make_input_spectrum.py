"""F4.13: the SHD input spectrum against the learned time-constant solutions.

The premise of the temporal-basis argument: the task's input carries power over a band, a leaky membrane with
time constant tau is a low-pass filter with corner frequency f = 1/(2 pi tau), and so a population of time
constants covers a band of the input spectrum while a single constant samples one point in it.

Computes the power spectral density of the SHD test input (population activity summed over the 700 channels,
averaged over trials), its spectral centroid, and draws beneath it the learned heterogeneous and homogeneous
time constants mapped to corner frequencies. Two stacked panels on one frequency axis, no twin axis.

Run: python code/analysis_suite/make_input_spectrum.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S

RROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "code" / "model_perez_nieves"))
from data_gen import open_file

CACHE = REPO / "data_cache" / "network_json"
FIGS = REPO / "docs" / "analysis_suite" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
S.apply()

DT = 1e-3
NB_STEPS = 1000
SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))

print("loading SHD test set ...")
u, t, l = open_file(str(RROOT / "data" / "shd" / "shd_test.h5"))
units, times = list(u[:]), list(t[:])
u._v_file.close()
print(f"  {len(times)} trials")

# population input signal per trial: spike count per 1 ms step, summed over the 700 channels
psds = []
for ti in times:
    idx = np.round(np.asarray(ti) / DT).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < NB_STEPS)]
    sig = np.bincount(idx, minlength=NB_STEPS).astype(np.float64)
    sig -= sig.mean()                       # drop DC so the centroid is about the dynamics
    F = np.fft.rfft(sig * np.hanning(NB_STEPS))
    psds.append(np.abs(F) ** 2)
P = np.mean(psds, axis=0)
freqs = np.fft.rfftfreq(NB_STEPS, DT)

band = (freqs > 0) & (freqs <= 200)
centroid = float((freqs[band] * P[band]).sum() / P[band].sum())
tau_centroid = 1.0 / (2 * np.pi * centroid) * 1e3
cum = np.cumsum(P[band]) / P[band].sum()
f10 = float(freqs[band][np.searchsorted(cum, 0.10)])
f90 = float(freqs[band][np.searchsorted(cum, 0.90)])
print(f"spectral centroid = {centroid:.2f} Hz  -> tau = 1/(2 pi f) = {tau_centroid:.1f} ms")
print(f"10-90% power band = {f10:.1f} to {f90:.1f} Hz -> tau {1/(2*np.pi*f90)*1e3:.1f} to {1/(2*np.pi*f10)*1e3:.1f} ms")

het_tau = np.concatenate([np.array(json.load(open(CACHE / f"4class_seed{s}_het_lognormal.json"))["tau_all_ms"])
                          for s in SEEDS])
_CONFOUNDED = [202, 210, 340, 440, 550, 660, 710, 820, 930]


def _hom_tau(s):
    p = CACHE / f"4class_seed{s}_hom_geommean_nowd_lognormal.json"   # clean (no-WD) retrain for the 9
    src = p if (s in _CONFOUNDED and p.exists()) else CACHE / f"4class_seed{s}_hom_geommean_lognormal.json"
    return json.load(open(src))["tau_median_ms"]


hom_tau = np.array([_hom_tau(s) for s in SEEDS])
het_f = 1.0 / (2 * np.pi * het_tau * 1e-3)
hom_f = 1.0 / (2 * np.pi * hom_tau * 1e-3)

XLIM = (0.7, 100)
fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.15]})
fig.subplots_adjust(hspace=0.32)

ax = axes[0]
ax.axvspan(f10, f90, color=S.CHANCE, alpha=0.10, lw=0, label="Band holding 10% to 90% of the input power")
ax.semilogx(freqs[band], P[band] / P[band].max(), color=S.ACCENT, lw=2.0,
            label="Input power spectral density, normalised")
S.ref_line(ax, x=centroid, label=f"Spectral centroid, {centroid:.1f} Hz ($\\tau$ = {tau_centroid:.0f} ms)",
           ls=(0, (4, 2)))
ax.set_ylabel("Normalised input power")
ax.set_ylim(0, 1.05)
S.panel_title(ax, "a", "Power spectrum of the spiking input")
S.legend(ax, loc="upper right")

ax = axes[1]
ax.axvspan(f10, f90, color=S.CHANCE, alpha=0.10, lw=0)
ax.hist(het_f, bins=np.logspace(np.log10(XLIM[0]), np.log10(60), 34), color=S.HET, alpha=0.85,
        edgecolor="white", linewidth=0.5, label="Heterogeneous corner frequencies, one per neuron")
for f in hom_f:
    ax.axvline(f, color=S.HOM, alpha=0.75, lw=1.4, zorder=3)
ax.axvline(np.nan, color=S.HOM, lw=1.4, label="Homogeneous corner frequency, one per seed")
ax.set_xscale("log")
ax.set_xlim(*XLIM)
S.plain_log_ticks(ax, [1, 2, 5, 10, 20, 50, 100])
ax.set_xlabel("Corner frequency $1/(2\\pi\\tau)$ (Hz, log scale)")
ax.set_ylabel("Number of hidden units")
S.panel_title(ax, "b", "Corner frequencies of the learned time constants")
ax.legend(loc="upper right", frameon=True, framealpha=1.0, facecolor="white", edgecolor=S.GRID, fancybox=False)
S.save(fig, FIGS / "F4_13_input_spectrum.png")

json.dump({"centroid_hz": centroid, "tau_centroid_ms": tau_centroid, "f10_hz": f10, "f90_hz": f90,
           "tau_at_f90_ms": 1 / (2 * np.pi * f90) * 1e3, "tau_at_f10_ms": 1 / (2 * np.pi * f10) * 1e3,
           "het_frac_in_band": float(((het_f >= f10) & (het_f <= f90)).mean()),
           "hom_frac_in_band": float(((hom_f >= f10) & (hom_f <= f90)).mean())},
          open(REPO / "docs" / "analysis_suite" / "input_spectrum.json", "w"), indent=1)
print(f"het units with corner inside the 10-90% band: {((het_f >= f10) & (het_f <= f90)).mean():.1%}")
print(f"hom nets  with corner inside the 10-90% band: {((hom_f >= f10) & (hom_f <= f90)).mean():.1%}")

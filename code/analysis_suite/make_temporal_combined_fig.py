"""The temporal structure of the two codes, in one figure. Top row: the instantaneous four-class information
through the trial, raw and normalised to each network's own peak, from the pairwise mutual information
profiles. Bottom row: language and parity decoded from the hidden state at each step, one panel per type of
network, with the late window used for the sustained-decodability statistic shaded.

Reads data_cache/supervisor_extension/temporal_lockin_n20.json and
data_cache/analysis_suite/instantaneous_decode_n20.json, writes docs/analysis_suite/figs/temporal_combined.png.

Run: python code/analysis_suite/make_temporal_combined_fig.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S

REPO = HERE.parent.parent
TL = REPO / "data_cache" / "supervisor_extension" / "temporal_lockin_n20.json"
ID = REPO / "data_cache" / "analysis_suite" / "instantaneous_decode_n20.json"
FIGS = REPO / "docs" / "analysis_suite" / "figs"

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
COL_LANG = "#1b7837"
COL_PAR = "#762a83"


def smooth(p, w=7):
    return np.convolve(np.asarray(p, float), np.ones(w) / w, mode="same")


def ci95(M):
    M = np.asarray(M, float)
    return np.nanmean(M, 0), 1.96 * np.nanstd(M, 0, ddof=1) / np.sqrt(M.shape[0])


def main():
    S.apply()
    tl = json.load(open(TL))["profiles"]
    idc = json.load(open(ID))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)

    tsteps = len(tl[f"{SEEDS[0]}_het"]["inst"]["4-class"])
    xt = np.arange(tsteps) / (tsteps - 1)

    # (a) instantaneous four-class information
    ax = axes[0]
    for arch in ("het", "hom"):
        P = np.array([smooth(tl[f"{s}_{arch}"]["inst"]["4-class"]) for s in SEEDS])
        m, h = ci95(P)
        S.band(ax, xt, m, h, S.COL[arch], label=S.LABEL[arch])
    ax.set_ylabel("Instantaneous class information (bits)")
    S.panel_title(ax, "a", "Class information through the trial")
    S.legend(ax, loc="upper right")

    # (b) normalised to each network's own peak
    ax = axes[1]
    for arch in ("het", "hom"):
        P = np.array([smooth(tl[f"{s}_{arch}"]["inst"]["4-class"]) for s in SEEDS])
        P = P / P.max(1, keepdims=True)
        m, h = ci95(P)
        S.band(ax, xt, m, h, S.COL[arch], label=S.LABEL[arch])
    ax.set_ylabel("Class information, fraction of own peak")
    S.panel_title(ax, "b", "Class information, normalised to peak")

    # The language and parity decoding panels moved to F4_15_class_resolution.png (make_info_fig.py), which
    # sits beside the sequential-processing paragraph in sec:division.
    for ax in axes.flat:
        ax.set_xlabel("Time (fraction of trial)")
        ax.set_xlim(0, 1)
    FIGS.mkdir(parents=True, exist_ok=True)
    S.save(fig, FIGS / "temporal_combined.png")


if __name__ == "__main__":
    main()

"""F4_14: the network memory function, redrawn from the representation cache.

run_geometry_n20.py computes the memory function alongside the other geometry measures but does not cache
the per-lag curves, only the memory capacity MC (their sum). This script recomputes the curves alone with the
same estimator (ridge readout of the past input drive from the hidden state, 70/30 trial split, R^2 per lag),
writes them to docs/analysis_suite/memory_curves.json so the figure can be restyled without recomputing, and
draws F4_14_memory.png.

Run: python code/analysis_suite/make_memory_fig.py [--redraw]
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S

REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "representation_clean"
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
CURVES = OUT / "memory_curves.json"
S.apply()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
BIN_MS, MAXLAG = 20, 25


def memory_function(hidden, u, rng, maxlag=MAXLAG):
    n, NB, D = hidden.shape
    idx = rng.permutation(n); cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    r2 = []
    for tau in range(maxlag + 1):
        Xtr = hidden[tr, tau:, :].reshape(-1, D); ytr = u[tr, :NB - tau].reshape(-1)
        Xte = hidden[te, tau:, :].reshape(-1, D); yte = u[te, :NB - tau].reshape(-1)
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=10.0).fit(sc.transform(Xtr), ytr)
        pred = m.predict(sc.transform(Xte))
        ss_res = np.sum((yte - pred) ** 2); ss_tot = np.sum((yte - yte.mean()) ** 2) + 1e-12
        r2.append(max(0.0, 1 - ss_res / ss_tot))
    return np.array(r2)


def compute():
    rng = np.random.default_rng(0)
    curves = {"het": {}, "hom": {}}
    for seed in SEEDS:
        for arch in ("het", "hom"):
            d = np.load(CACHE / f"repr_seed{seed}_{arch}.npz")
            mc = memory_function(d["hidden"].astype(float), d["u"].astype(float), rng)
            curves[arch][str(seed)] = mc.tolist()
            print(f"  {seed} {arch}: MC {mc.sum():.2f}", flush=True)
    json.dump({"bin_ms": BIN_MS, "maxlag": MAXLAG, "curves": curves}, open(CURVES, "w"), indent=1)
    return curves


def draw(curves):
    lags = np.arange(MAXLAG + 1) * BIN_MS
    geom = json.load(open(OUT / "geometry_n20_summary.json"))["paired"]["MC"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for arch in ("het", "hom"):
        M = np.stack([curves[arch][str(s)] for s in SEEDS])
        m = M.mean(0)
        h = np.array([S.ci95_mean(M[:, j]) for j in range(M.shape[1])])
        S.band(ax, lags, m, h, S.COL[arch], label=S.LABEL[arch], marker="o", ms=3.5)
    ax.set_xlabel("Lag into the past (ms)")
    ax.set_ylabel("$R^2$ of reconstructing the past input drive")
    ax.set_ylim(0, 1.0)
    S.panel_title(ax, "c", "Network memory function")   # (c): it sits beside the two filter-bank panels
    S.legend(ax, loc="upper right")
    S.note(ax, f"Memory capacity {geom['het_mean']:.2f} against {geom['hom_mean']:.2f}, "
               f"paired $t$ $p$ = {S.fmt_p(geom['t_p'])}, {geom['wins']}/{geom['n']} seeds", loc="lower left")
    S.save(fig, FIGS / "F4_14_memory.png")


if __name__ == "__main__":
    if "--redraw" in sys.argv and CURVES.exists():
        curves = json.load(open(CURVES))["curves"]
    else:
        curves = compute()
    draw(curves)

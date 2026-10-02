"""Cross-validated signal dimensionality at the clean n = 20, het vs hom.

Raw n_90 and the participation ratio count every direction that carries variance, including directions that
carry only trial-to-trial noise. This separates the two by cross-validation, following the cvPCA logic of
Stringer et al. 2019: split the trials of each condition into two disjoint halves, take the two independent
condition-mean matrices A and B, derive principal components from A, project both, and read the cross-validated
variance of component i as the covariance between the two halves along it,

    cv_i = < a_i , b_i >_conditions .

Noise is independent between halves, so a noise direction has cv_i near zero in expectation while a signal
direction stays positive. The signal spectrum is the positive part of cv, and its participation ratio is the
number of dimensions that carry reliable stimulus-locked signal.

Conditions are digit x language x speaker cells (speaker comes from the SHD file's extra/speaker field) with at
least MIN_TRIALS trials, so each half holds at least two trials per condition. That gives 73 conditions, above
the 64 neurons, so the estimate is not capped by the number of conditions the way a 20-class split would be.
The same cell list, the same trial split and the same repeat seeds are used for all forty networks.

Run: python code/analysis_suite/run_signal_dimension_n20.py
"""
import json, sys
from pathlib import Path
import numpy as np
import h5py

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple
import matplotlib.pyplot as plt

RROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "representation_clean"
SHD_TEST = RROOT / "data" / "shd" / "shd_test.h5"
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
S.apply()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
MIN_TRIALS = 4
N_REPS = 20          # split-half repeats, averaged

with h5py.File(SHD_TEST, "r") as _f:
    RAW = np.asarray(_f["labels"][:]).astype(int)
    SPK = np.asarray(_f["extra"]["speaker"][:]).astype(int)

CELL = RAW * 100 + SPK
_u, _c = np.unique(CELL, return_counts=True)
KEEP_CELLS = _u[_c >= MIN_TRIALS]
CELL_IDX = [np.where(CELL == c)[0] for c in KEEP_CELLS]


def pr_positive(v):
    v = np.asarray(v, float)
    v = v[v > 0]
    return float((v.sum() ** 2) / np.square(v).sum()) if v.size else 0.0


def cv_spectrum(X, rng, shuffle=False):
    """One split-half cvPCA pass. Returns the cross-validated variance per component.

    With shuffle=True the trials are reassigned across cells at random, keeping the cell sizes, which destroys
    the stimulus-locked signal and gives the null spectrum a purely noisy code would produce.
    """
    if shuffle:
        pool = rng.permutation(np.concatenate(CELL_IDX))
        cells, k = [], 0
        for idx in CELL_IDX:
            cells.append(pool[k: k + idx.size]); k += idx.size
    else:
        cells = CELL_IDX
    A, B = [], []
    for idx in cells:
        p = rng.permutation(idx)
        h = p.size // 2
        A.append(X[p[:h]].mean(0))
        B.append(X[p[h: 2 * h]].mean(0))
    A = np.stack(A); B = np.stack(B)
    mu = 0.5 * (A.mean(0) + B.mean(0))
    A = A - mu; B = B - mu
    # components from A alone, so the projection is not fitted on B
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    a = A @ Vt.T
    b = B @ Vt.T
    return (a * b).mean(0)                     # cross-validated variance per component


def analyse(X):
    """Cross-validated signal spectrum plus a shuffled null.

    A simple count of components with cv > 0 is not usable: the expectation of cv_i is the signal variance
    along component i, which is non-negative, so that count saturates at the neuron number for every network.
    The number of reliable dimensions is therefore taken against the shuffled null instead.
    """
    n_comp = min(len(CELL_IDX), X.shape[1])
    cv = np.zeros(n_comp)
    null = np.zeros((N_REPS, n_comp))
    for r in range(N_REPS):
        rng = np.random.default_rng(1000 + r)        # identical splits for every network
        cv += cv_spectrum(X, rng)[:n_comp]
        rng_n = np.random.default_rng(5000 + r)
        null[r] = cv_spectrum(X, rng_n, shuffle=True)[:n_comp]
    cv /= N_REPS
    null_hi = np.percentile(null, 95, axis=0)
    return {"signal_pr": pr_positive(cv),
            "n_reliable": int((cv > null_hi).sum()),
            "cv": cv.tolist(), "null_hi": null_hi.tolist()}


def main():
    print(f"conditions: {len(CELL_IDX)} cells (digit x language x speaker, >= {MIN_TRIALS} trials), "
          f"{sum(len(i) for i in CELL_IDX)} trials, {N_REPS} split-half repeats\n")
    per = {"het": {}, "hom": {}}
    curves = {"het": [], "hom": []}
    nulls = {"het": [], "hom": []}
    print(f"{'seed':>5} {'arch':>4} | {'signalPR':>8} {'n_reliab':>8}")
    for seed in SEEDS:
        for arch in ("het", "hom"):
            X = np.load(CACHE / f"repr_seed{seed}_{arch}.npz")["rate"].astype(float)
            r = analyse(X)
            per[arch][seed] = {k: r[k] for k in ("signal_pr", "n_reliable")}
            curves[arch].append(np.asarray(r["cv"]))
            nulls[arch].append(np.asarray(r["null_hi"]))
            print(f"{seed:>5} {arch:>4} | {r['signal_pr']:>8.2f} {r['n_reliable']:>8d}", flush=True)

    V = lambda arch, k: np.array([per[arch][s][k] for s in SEEDS], float)
    METRICS = [("signal_pr", "signal dimensionality (PR)", True),
               ("n_reliable", "dimensions above shuffled null", True)]
    triples = {}
    print("\n--- paired het vs hom (n = 20) ---")
    for k, lbl, hib in METRICS:
        r = paired_triple(V("het", k), V("hom", k), lbl, higher_is_better=hib)
        triples[k] = r
        print(f"{lbl:>30}: het {r['het_mean']:.3f}  hom {r['hom_mean']:.3f}  d {r['delta']:+.3f}  "
              f"dz {r['dz']:+.2f}  t {S.fmt_p(r['t_p'])}  W {S.fmt_p(r['w_p'], r['w_floor'])}  "
              f"perm {S.fmt_p(r['perm_p'], r['perm_floor'])}  wins {r['wins']}/{r['n']}", flush=True)

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    r = triples["signal_pr"]
    ax.bar([0, 1], [r["het_mean"], r["hom_mean"]], 0.6, yerr=[r["het_sd"], r["hom_sd"]],
           color=[S.HET, S.HOM], capsize=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels([S.LABEL["het"], S.LABEL["hom"]])
    ax.set_ylabel("cross-validated signal dimensionality (PR)")
    ax.set_title("Signal dimensionality\n(noise directions removed by cross-validation)", fontsize=10)
    ax.text(0.5, 0.97, f"$\\Delta$ {r['delta']:+.2f}   $d_z$ {r['dz']:+.2f}\n"
                       f"perm {S.fmt_p(r['perm_p'], r['perm_floor'])}   wins {r['wins']}/{r['n']}",
            ha="center", va="top", transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
    ax = axes[1]
    for arch, col in [("het", S.HET), ("hom", S.HOM)]:
        M = np.stack(curves[arch])
        m = M.mean(0); sd = M.std(0)
        x = np.arange(1, m.size + 1)
        S.band(ax, x, m, sd, col, label=S.LABEL[arch])
    nl = np.stack(nulls["het"] + nulls["hom"]).mean(0)
    ax.plot(np.arange(1, nl.size + 1), nl, color=S.CHANCE, ls=":", lw=1.4, label="shuffled null (95th pct)")
    ax.set_xlabel("component"); ax.set_ylabel("cross-validated variance")
    ax.set_yscale("log")
    # the last few components collapse to numerical zero; clip the axis to the informative range
    top = float(np.stack(curves["het"] + curves["hom"]).max())
    ax.set_ylim(top * 1e-7, top * 3)
    ax.set_title("Cross-validated spectrum", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("Cross-validated signal dimensionality, het vs hom (n = 20)", fontweight="bold", y=1.02)
    S.save(fig, FIGS / "F4_21_signal_dimension.png")

    summary = {
        "analysis": "cv_signal_dimensionality", "n_pairs": len(SEEDS),
        "conditions": {"definition": "digit x language x speaker", "n_cells": len(CELL_IDX),
                       "min_trials": MIN_TRIALS, "n_trials": int(sum(len(i) for i in CELL_IDX))},
        "n_split_half_repeats": N_REPS,
        "by_arch": {arch: {k: {"mean": float(V(arch, k).mean()), "sd": float(V(arch, k).std(ddof=1))}
                           for k, _, _ in METRICS} for arch in ("het", "hom")},
        "paired": triples,
        "per_network": {arch: {int(s): per[arch][s] for s in SEEDS} for arch in ("het", "hom")},
    }
    json.dump(summary, open(OUT / "signal_dimension_n20_summary.json", "w"), indent=1)
    print(f"\nwrote {OUT/'signal_dimension_n20_summary.json'} and F4_21_signal_dimension.png")


if __name__ == "__main__":
    main()

"""Temporal generalisation matrix at the clean n = 20, het vs hom.

Train a linear decoder on the population state at one time bin and test it at every other time bin (King and
Dehaene 2014). The resulting square matrix says whether the code is stationary or dynamic: a decoder that only
works at the moment it was trained gives a narrow diagonal, while a code that holds the same task information in
the same directions over time generalises off the diagonal.

This is the analysis most directly tied to the manipulated variable. A spread of membrane time constants gives
the population a bank of integration windows, so the prediction is that the heterogeneous network holds
decodable class information in a stable subspace for longer, and shows a wider off-diagonal band.

Bins are the 50 x 20 ms bins of the cached hidden state. The decoder is the house logistic. Trials are split by
stratified cross-validation; the scaler and the decoder are both fitted at the train bin and applied unchanged
at the test bin, so the matrix measures transfer of a fixed readout across time rather than refitting.

Run: python code/analysis_suite/run_temporal_generalisation_n20.py
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple
import matplotlib.pyplot as plt

REPO = HERE.parent.parent
CACHE = REPO / "data_cache" / "representation_clean"
OUT = REPO / "docs" / "analysis_suite"
FIGS = OUT / "figs"
FIGS.mkdir(parents=True, exist_ok=True)
S.apply()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930] + list(range(1001, 1011))
BIN_MS = 20
N_FOLDS = 3
CHANCE = 0.25          # 4-class task
N_JOBS = -1


def tg_matrix(seed, arch):
    d = np.load(CACHE / f"repr_seed{seed}_{arch}.npz")
    H = d["hidden"].astype(float)          # (N, 50, 64)
    y = d["y"].astype(int)
    n_bins = H.shape[1]
    M = np.zeros((n_bins, n_bins))
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
    for tr, te in cv.split(H[:, 0, :], y):
        for t in range(n_bins):
            sc = StandardScaler().fit(H[tr, t, :])
            clf = LogisticRegression(max_iter=200, C=1.0).fit(sc.transform(H[tr, t, :]), y[tr])
            for tp in range(n_bins):
                M[t, tp] += clf.score(sc.transform(H[te, tp, :]), y[te])
    M /= N_FOLDS
    return seed, arch, M


def lag_profile(M):
    """Mean decoding accuracy as a function of the train-test lag |t - t'|, in bins."""
    n = M.shape[0]
    i, j = np.indices((n, n))
    lag = np.abs(i - j)
    return np.array([M[lag == L].mean() for L in range(n)])


def summarise(M):
    n = M.shape[0]
    diag = np.diag(M)
    off = M[~np.eye(n, dtype=bool)]
    prof = lag_profile(M)
    long_lag = float(prof[10:].mean())          # transfer beyond 200 ms
    # generalisation width: per train bin, how many test bins stay above 75% of that bin's own peak,
    # counted only for bins that actually decode above chance
    peak = diag.copy()
    usable = peak > CHANCE + 0.05
    widths = []
    for t in np.where(usable)[0]:
        thresh = CHANCE + 0.75 * (peak[t] - CHANCE)
        widths.append(float((M[t] >= thresh).sum()))
    return {"diag_mean": float(diag.mean()),
            "diag_peak": float(diag.max()),
            "offdiag_mean": float(off.mean()),
            "gen_ratio": float((off.mean() - CHANCE) / max(diag.mean() - CHANCE, 1e-9)),
            "gen_width_bins": float(np.mean(widths)) if widths else float("nan"),
            "long_lag_acc": long_lag,
            "long_lag_ratio": float((long_lag - CHANCE) / max(diag.mean() - CHANCE, 1e-9))}


def main():
    jobs = [(s, a) for s in SEEDS for a in ("het", "hom")]
    print(f"temporal generalisation: {len(jobs)} nets x 50 train bins x {N_FOLDS} folds ...", flush=True)
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(delayed(tg_matrix)(s, a) for s, a in jobs)

    mats = {"het": {}, "hom": {}}
    for seed, arch, M in res:
        mats[arch][seed] = M
    per = {arch: {s: summarise(mats[arch][s]) for s in SEEDS} for arch in ("het", "hom")}

    V = lambda arch, k: np.array([per[arch][s][k] for s in SEEDS], float)
    METRICS = [("diag_mean", "instantaneous decoding (diagonal)", True),
               ("offdiag_mean", "cross-time decoding (off-diagonal)", True),
               ("gen_ratio", "generalisation ratio (off / diag)", True),
               ("gen_width_bins", "generalisation width (bins)", True),
               ("long_lag_acc", "long-lag transfer (> 200 ms)", True),
               ("long_lag_ratio", "long-lag transfer / diagonal", True)]
    triples = {}
    print("\n--- paired het vs hom (n = 20) ---")
    for k, lbl, hib in METRICS:
        r = paired_triple(V("het", k), V("hom", k), lbl, higher_is_better=hib)
        triples[k] = r
        print(f"{lbl:>36}: het {r['het_mean']:.3f}  hom {r['hom_mean']:.3f}  d {r['delta']:+.3f}  "
              f"dz {r['dz']:+.2f}  perm {S.fmt_p(r['perm_p'], r['perm_floor'])}  wins {r['wins']}/{r['n']}",
              flush=True)

    Mh = np.mean([mats["het"][s] for s in SEEDS], axis=0)
    Mo = np.mean([mats["hom"][s] for s in SEEDS], axis=0)
    diff = Mh - Mo
    n_bins = Mh.shape[0]
    ext = [0, n_bins * BIN_MS, n_bins * BIN_MS, 0]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 10.0), constrained_layout=True)
    vmin, vmax = CHANCE, max(Mh.max(), Mo.max())
    ims = []
    for ax, (M, title) in zip(axes.flat[:2], [(Mh, "heterogeneous"), (Mo, "homogeneous")]):
        im = ax.imshow(M, origin="upper", extent=ext, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        ims.append(im)
        ax.plot([0, n_bins * BIN_MS], [0, n_bins * BIN_MS], color="white", lw=0.7, ls=":", alpha=0.7)
        ax.set_xlabel("test time (ms)"); ax.set_ylabel("train time (ms)")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
    fig.colorbar(ims[0], ax=list(axes.flat[:2]), fraction=0.046, pad=0.02, label="decoding accuracy")

    ax = axes.flat[2]
    lim = float(np.abs(diff).max())
    imd = ax.imshow(diff, origin="upper", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
    ax.plot([0, n_bins * BIN_MS], [0, n_bins * BIN_MS], color="#333", lw=0.7, ls=":", alpha=0.7)
    ax.set_xlabel("test time (ms)"); ax.set_ylabel("train time (ms)")
    ax.set_title("het $-$ hom", fontsize=10)
    ax.grid(False)
    fig.colorbar(imd, ax=ax, fraction=0.046, pad=0.02, label="accuracy difference")

    ax = axes.flat[3]
    lags = np.arange(n_bins) * BIN_MS
    for arch, col in [("het", S.HET), ("hom", S.HOM)]:
        P = np.stack([lag_profile(mats[arch][s]) for s in SEEDS])
        S.band(ax, lags, P.mean(0), P.std(0), col, label=S.LABEL[arch])
    ax.axhline(CHANCE, color=S.CHANCE, ls=":", lw=1.2, label="chance (4-class)")
    ax.set_xlabel("train-test lag (ms)"); ax.set_ylabel("decoding accuracy")
    ax.set_title("Transfer against lag", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("Temporal generalisation of the class code (mean over 20 seeds)", fontweight="bold")
    S.save(fig, FIGS / "F4_22_temporal_generalisation.png")

    summary = {
        "analysis": "temporal_generalisation_matrix", "n_pairs": len(SEEDS),
        "bin_ms": BIN_MS, "n_bins": int(n_bins), "n_folds": N_FOLDS, "chance": CHANCE,
        "by_arch": {arch: {k: {"mean": float(V(arch, k).mean()), "sd": float(V(arch, k).std(ddof=1))}
                           for k, _, _ in METRICS} for arch in ("het", "hom")},
        "paired": triples,
        "per_network": {arch: {int(s): per[arch][s] for s in SEEDS} for arch in ("het", "hom")},
    }
    json.dump(summary, open(OUT / "temporal_generalisation_n20_summary.json", "w"), indent=1)
    np.savez_compressed(OUT / "temporal_generalisation_matrices.npz", het=Mh, hom=Mo)
    print(f"\nwrote {OUT/'temporal_generalisation_n20_summary.json'} and F4_22_temporal_generalisation.png")


if __name__ == "__main__":
    main()

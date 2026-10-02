"""Shattering dimensionality as a function of retained principal components, het vs hom, n = 20.

The reconstruction analysis shows that the heterogeneous network's task information keeps accumulating past the
rank at which the homogeneous network saturates. The shattering analysis shows that the heterogeneous code
separates more arbitrary groupings of the stimuli. This asks whether those are the same fact: if the extra
expressiveness lives in the higher principal components, the two shattering curves should sit together at low
rank and separate above the reconstruction crossover near m = 8.

Method. For each network the rate code is restricted to the shared balanced trial sample. Within each
cross-validation fold the scaler and the PCA are fitted on the training trials only and applied to the held-out
trials, so no test information leaks into the projection. The PCA is unsupervised and therefore label-free, which
means one fit per fold is reused for every dichotomy and every rank. For each rank m the decoder sees the first m
principal component scores, which for a linear readout is equivalent to the rank-m reconstruction in the original
space. Shattering dimensionality at rank m is the accuracy averaged over all 35 balanced dichotomies of the
eight-condition set.

Run: python code/analysis_suite/run_shattering_vs_rank_n20.py
"""
import json, sys, itertools
from pathlib import Path
import numpy as np
import h5py
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed

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
RANKS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
CAP_PER_COND = 100
N_FOLDS = 5
N_JOBS = -1

with h5py.File(SHD_TEST, "r") as _f:
    RAW = np.asarray(_f["labels"][:]).astype(int)
DIGIT, LANG = RAW % 10, RAW // 10
COND = 4 * LANG + 2 * (DIGIT % 2) + (DIGIT >= 5).astype(int)     # 8 conditions
N_COND = 8


def balanced_index(cond, n_cond, cap, rng):
    k = min(cap, min(int((cond == c).sum()) for c in range(n_cond)))
    idx = []
    for c in range(n_cond):
        ci = np.where(cond == c)[0]
        rng.shuffle(ci)
        idx.append(ci[:k])
    return np.sort(np.concatenate(idx)), k


def all_balanced_dichotomies(n):
    return [frozenset(p) for p in itertools.combinations(range(n), n // 2) if 0 in p]


IDX, K_PER = balanced_index(COND, N_COND, CAP_PER_COND, np.random.default_rng(0))
COND_SUB = COND[IDX]
DICH = all_balanced_dichotomies(N_COND)
LABELS = [np.where(np.isin(COND_SUB, list(p)), 1, -1) for p in DICH]


def net_curve(seed, arch):
    """Shattering dimensionality at each rank for one network."""
    X = np.load(CACHE / f"repr_seed{seed}_{arch}.npz")["rate"].astype(float)[IDX]
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=0)
    acc = np.zeros((len(RANKS), len(DICH)))
    # folds are defined on the condition labels so every fold is condition-balanced
    for tr, te in cv.split(X, COND_SUB):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        p = PCA(n_components=max(RANKS), svd_solver="full").fit(Xtr)   # label-free, one fit per fold
        Ztr, Zte = p.transform(Xtr), p.transform(Xte)
        for di, lab in enumerate(LABELS):
            ytr, yte = lab[tr], lab[te]
            for ri, m in enumerate(RANKS):
                clf = LogisticRegression(max_iter=1000, C=1.0).fit(Ztr[:, :m], ytr)
                acc[ri, di] += clf.score(Zte[:, :m], yte)
    acc /= N_FOLDS
    return seed, arch, acc.mean(1)          # mean over dichotomies -> one value per rank


def main():
    print(f"{N_COND} conditions, {K_PER} trials/cond ({IDX.size} trials), {len(DICH)} dichotomies, "
          f"{len(RANKS)} ranks\n", flush=True)
    jobs = [(s, a) for s in SEEDS for a in ("het", "hom")]
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(delayed(net_curve)(s, a) for s, a in jobs)

    curves = {"het": {}, "hom": {}}
    for seed, arch, c in res:
        curves[arch][seed] = c
    M = {a: np.stack([curves[a][s] for s in SEEDS]) for a in ("het", "hom")}   # (20, n_ranks)

    print(f"\n{'rank':>5} | {'het':>7} {'hom':>7} {'delta':>7} {'dz':>6} {'perm':>10} {'wins':>6}")
    per_rank = {}
    for ri, m in enumerate(RANKS):
        r = paired_triple(M["het"][:, ri], M["hom"][:, ri], f"shatter@m={m}")
        per_rank[m] = r
        print(f"{m:>5} | {r['het_mean']:>7.4f} {r['hom_mean']:>7.4f} {r['delta']:>+7.4f} "
              f"{r['dz']:>+6.2f} {S.fmt_p(r['perm_p'], r['perm_floor']):>10} {r['wins']:>3}/{r['n']}",
              flush=True)

    # first rank at which the heterogeneous advantage becomes significant and stays so
    sig = [m for m in RANKS if per_rank[m]["perm_p"] < 0.05 and per_rank[m]["delta"] > 0]
    onset = None
    for m in RANKS:
        if m in sig and all((mm in sig) for mm in RANKS if mm >= m):
            onset = m
            break
    print(f"\nheterogeneous advantage becomes significant and stays so from rank m = {onset}")

    # ================================================================= figure
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax = axes[0]
    for arch, col in [("het", S.HET), ("hom", S.HOM)]:
        S.band(ax, RANKS, M[arch].mean(0), M[arch].std(0), col, label=S.LABEL[arch], marker="o", ms=3.5)
    ax.axhline(0.5, color=S.CHANCE, ls=":", lw=1.2, label="chance")
    if onset is not None:
        ax.axvline(onset, color=S.ACCENT, ls="--", lw=1.0, alpha=0.7)
        ax.annotate(f"advantage opens\nat m = {onset}", xy=(onset, 0.62), xytext=(onset * 1.5, 0.57),
                    fontsize=8, color=S.ACCENT,
                    arrowprops=dict(arrowstyle="->", color=S.ACCENT, lw=0.9))
    ax.set_xscale("log"); ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
    ax.set_xlabel("principal components retained ($m$)")
    ax.set_ylabel("shattering dimensionality")
    ax.set_title("Shattering against retained rank", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    d = M["het"] - M["hom"]
    S.band(ax, RANKS, d.mean(0), d.std(0), S.ACCENT, marker="o", ms=3.5)
    ax.axhline(0, color=S.CHANCE, ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_xticks(RANKS); ax.set_xticklabels(RANKS)
    ax.set_xlabel("principal components retained ($m$)")
    ax.set_ylabel("heterogeneous $-$ homogeneous")
    ax.set_title("The advantage against retained rank", fontsize=10)
    fig.suptitle("Where in the spectrum the shattering advantage lives (n = 20)", fontweight="bold", y=1.02)
    S.save(fig, FIGS / "F4_25_shattering_vs_rank.png")

    summary = {
        "analysis": "shattering_vs_retained_rank", "n_pairs": len(SEEDS),
        "condition_set": {"n_conditions": N_COND, "trials_per_condition": K_PER,
                          "n_dichotomies": len(DICH), "exhaustive": True},
        "ranks": RANKS, "onset_rank": onset,
        "by_arch": {a: {"mean": M[a].mean(0).tolist(), "sd": M[a].std(0, ddof=1).tolist()}
                    for a in ("het", "hom")},
        "paired_by_rank": {str(m): per_rank[m] for m in RANKS},
        "per_network": {a: {int(s): curves[a][s].tolist() for s in SEEDS} for a in ("het", "hom")},
    }
    json.dump(summary, open(OUT / "shattering_vs_rank_n20_summary.json", "w"), indent=1)
    print(f"\nwrote {OUT/'shattering_vs_rank_n20_summary.json'} and F4_25_shattering_vs_rank.png")


if __name__ == "__main__":
    main()

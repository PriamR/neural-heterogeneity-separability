"""Correlation-corrected Cohen 2020 manifold capacity at the clean n = 20, het vs hom.

Same design as run_capacity_n20.py (class manifolds in the 64-D rate code, two condition sets, shared balanced
trials, identical t-vector seed, paired within seed) but using the FULL reference manifold_analysis_corr via
mftma_capacity_corr, which includes the low-rank center-correlation subtraction. This is the theory-faithful
alpha_c that folds the het center-decorrelation advantage into capacity. Slower (Stiefel optimisation per net).

Run: python code/analysis_suite/run_capacity_corr_n20.py
"""
import json, sys
from pathlib import Path
import numpy as np
import h5py
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple
from mftma_capacity_corr import manifold_capacity_corr
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
CAP_PER_COND = 60
N_T = 200
KAPPA = 0
N_REPS = 3
T_SEED = 0
N_JOBS = -1

with h5py.File(SHD_TEST, "r") as _f:
    RAW_LABEL = np.asarray(_f["labels"][:]).astype(int)
DIGIT, LANG = RAW_LABEL % 10, RAW_LABEL // 10


def condition_sets():
    condA = 4 * LANG + 2 * (DIGIT % 2) + (DIGIT >= 5).astype(int)
    return {"A_8cond": (condA, 8), "B_20cond": (RAW_LABEL.copy(), 20)}


def balanced_index(cond, n_cond, cap, rng):
    k = min(cap, min(int((cond == c).sum()) for c in range(n_cond)))
    idx = []
    for c in range(n_cond):
        ci = np.where(cond == c)[0]
        rng.shuffle(ci)
        idx.append(ci[:k])
    return np.sort(np.concatenate(idx)), k


def net_capacity(seed, arch, plans):
    d = np.load(CACHE / f"repr_seed{seed}_{arch}.npz")
    X = d["rate"].astype(float)
    y = d["y"].astype(int)
    assert np.array_equal(2 * (RAW_LABEL // 10) + (RAW_LABEL % 2), y), f"{seed}/{arch} label join off"
    out = {}
    for name, pl in plans.items():
        cond, idx = pl["cond"], pl["idx"]
        manifolds = [X[idx][cond[idx] == c].T for c in range(pl["n_cond"])]
        out[name] = manifold_capacity_corr(manifolds, kappa=KAPPA, n_t=N_T, n_reps=N_REPS, seed=T_SEED)
    return seed, arch, out


def main():
    sets = condition_sets()
    rng_t = np.random.default_rng(0)
    plans = {}
    for name, (cond, n_cond) in sets.items():
        idx, k = balanced_index(cond, n_cond, CAP_PER_COND, rng_t)
        plans[name] = dict(cond=cond, n_cond=n_cond, idx=idx, k=k)
        print(f"[{name}] {n_cond} manifolds, {k} points/manifold", flush=True)

    jobs = [(s, a) for s in SEEDS for a in ("het", "hom")]
    print(f"\ncorrelation-corrected capacity, {len(jobs)} nets (n_reps={N_REPS}) ...", flush=True)
    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(net_capacity)(s, a, plans) for s, a in jobs)

    per = {name: {"het": {}, "hom": {}} for name in plans}
    for seed, arch, out in results:
        for name in plans:
            per[name][arch][seed] = out[name]

    METRICS = [("alpha_M", "capacity alpha_M", True), ("R_M", "radius R_M", False),
               ("D_M", "dimension D_M", False), ("center_corr", "center correlation", False)]
    V = lambda name, arch, k: np.array([per[name][arch][s][k] for s in SEEDS], float)
    triples = {name: {} for name in plans}
    for name in plans:
        print(f"\n--- {name}: paired het vs hom (n = 20), correlation-corrected ---")
        for k, lbl, hib in METRICS:
            r = paired_triple(V(name, "het", k), V(name, "hom", k), lbl, higher_is_better=hib)
            triples[name][k] = r
            arrow = "het>hom" if hib else "het<hom"
            print(f"{lbl:>20} ({arrow}): het {r['het_mean']:.3f}  hom {r['hom_mean']:.3f}  d {r['delta']:+.3f}  "
                  f"dz {r['dz']:+.2f}  perm {S.fmt_p(r['perm_p'], r['perm_floor'])}  wins {r['wins']}/{r['n']}",
                  flush=True)

    for name in plans:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        panels = [("alpha_M", "capacity $\\alpha_M$ (corrected)", True), ("R_M", "radius $R_M$", False),
                  ("D_M", "dimension $D_M$", False), ("center_corr", "center correlation", False)]
        for ax, (k, ylab, hib) in zip(axes, panels):
            r = triples[name][k]
            ax.bar([0, 1], [r["het_mean"], r["hom_mean"]], 0.6, yerr=[r["het_sd"], r["hom_sd"]],
                   color=[S.HET, S.HOM], capsize=4)
            ax.set_xticks([0, 1]); ax.set_xticklabels([S.LABEL["het"], S.LABEL["hom"]])
            ax.set_ylabel(ylab)
            ax.set_title(f"{ylab}\n({'higher = more separable' if hib else 'lower = more separable'})", fontsize=10)
            ax.text(0.5, 0.97, f"$\\Delta$ {r['delta']:+.3f}  $d_z$ {r['dz']:+.2f}\n"
                               f"perm {S.fmt_p(r['perm_p'], r['perm_floor'])}  {r['wins']}/{r['n']}",
                    ha="center", va="top", transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))
        nice = "8 conditions" if name == "A_8cond" else "20 conditions"
        fig.suptitle(f"Correlation-corrected manifold capacity, het vs hom ({nice}, n = 20)",
                     fontweight="bold", y=1.03)
        S.save(fig, FIGS / f"F4_16b_capacity_corr_{name}.png")

    summary = {
        "analysis": "manifold_capacity_corrected_cohen2020_mftma", "n_pairs": len(SEEDS),
        "kappa": KAPPA, "n_t": N_T, "n_reps": N_REPS, "points_per_manifold": CAP_PER_COND,
        "note": "full reference manifold_analysis_corr incl. low-rank center-correlation subtraction (pymanopt-2.x port)",
        "by_arch": {name: {arch: {k: {"mean": float(V(name, arch, k).mean()),
                                      "sd": float(V(name, arch, k).std(ddof=1))}
                                  for k, _, _ in METRICS} for arch in ("het", "hom")} for name in plans},
        "paired": {name: triples[name] for name in plans},
        "per_network": {name: {arch: {int(s): {k: per[name][arch][s][k] for k, _, _ in METRICS}
                                      for s in SEEDS} for arch in ("het", "hom")} for name in plans},
    }
    json.dump(summary, open(OUT / "capacity_corr_n20_summary.json", "w"), indent=1)
    print(f"\nwrote {OUT/'capacity_corr_n20_summary.json'} and F4_16b_capacity_corr_*.png")


if __name__ == "__main__":
    main()

"""Manifold capacity on the four task classes, the partition the readout is actually trained on.

run_capacity_corr_n20.py measures capacity on two stimulus partitions, eight conditions and twenty, both of
them finer than the task. Neither is the partition the network was built to separate. The four task classes,
language crossed with parity, are what the readout actually has to pull apart, so the geometry of those four
manifolds is the one with a direct behavioural reading: it should be the set where capacity tracks accuracy.

Two things had to change for four manifolds, and both are limitations of the estimator rather than choices.

The correlation-corrected estimator cannot run here. Its low-rank centre-correlation subtraction estimates a
subspace from the manifold centres, and with four centres that subspace is rank three at most; the reference
implementation then divides by centre norms that have gone to zero, and the run fails outright with
divide-by-zero throughout `manifold_analysis_correlation`. The uncorrected estimator is used instead, which
computes the same alpha_M, R_M and D_M and reports the centre correlation directly rather than subtracting
it. So this set is comparable with the uncorrected analysis, not with the corrected numbers for the eight-
and twenty-manifold sets, and the table says so.

Replica mean-field capacity is also an asymptotic theory in the number of manifolds P and the dimension N,
and at P = 4 against N = 64 the load is 0.06, far below the capacity limit the theory describes. alpha_M here
is an extrapolation and should not be read in absolute terms. What carries no asymptotic assumption is the
geometry the theory decomposes capacity into, the radius, the manifold dimension and the centroid
correlation, all measured on the manifolds as they are. The paired comparison between arms is valid at any P,
since both arms are measured identically.

Everything else matches run_capacity_corr_n20.py, including the trial cap, the t-vector seed and the balanced
sampling. Results are merged into capacity_corr_n20_summary.json under the key T_4class.

Run: python code/analysis_suite/run_capacity_corr_task_n20.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import style as S
from stats_core import paired_triple
import matplotlib.pyplot as plt
import run_capacity_corr_n20 as C
from mftma_capacity import manifold_capacity

OUT = C.OUT
FIGS = C.FIGS
S.apply()

SET_NAME = "T_4class"
CLASSES = ["Eng-even", "Eng-odd", "Ger-even", "Ger-odd"]


def task_condition():
    """The trained partition: 2*language + parity, matching the y stored in the representation cache."""
    return 2 * C.LANG + (C.DIGIT % 2), 4


def net_capacity(seed, arch, plan):
    d = np.load(C.CACHE / f"repr_seed{seed}_{arch}.npz")
    X = d["rate"].astype(float)
    y = d["y"].astype(int)
    cond, idx = plan["cond"], plan["idx"]
    # the condition vector must be the label the network was trained against, not merely a partition of it
    assert np.array_equal(cond, y), f"{seed}/{arch}: task condition does not match the stored label"
    manifolds = [X[idx][cond[idx] == c].T for c in range(plan["n_cond"])]
    # uncorrected estimator: the corrected one divides by zero at P = 4, see the module docstring
    return seed, arch, manifold_capacity(manifolds, kappa=C.KAPPA, n_t=C.N_T, seed=C.T_SEED)


def main():
    cond, n_cond = task_condition()
    rng_t = np.random.default_rng(0)
    idx, k = C.balanced_index(cond, n_cond, C.CAP_PER_COND, rng_t)
    plan = dict(cond=cond, n_cond=n_cond, idx=idx, k=k)
    print(f"[{SET_NAME}] {n_cond} task manifolds, {k} points/manifold "
          f"(load P/N = {n_cond/64:.3f}, well below the capacity limit)\n", flush=True)

    jobs = [(s, a) for s in C.SEEDS for a in ("het", "hom")]
    res = Parallel(n_jobs=C.N_JOBS, backend="loky", verbose=5)(
        delayed(net_capacity)(s, a, plan) for s, a in jobs)

    per = {"het": {}, "hom": {}}
    for seed, arch, out in res:
        per[arch][seed] = out

    METRICS = [("alpha_M", "capacity alpha_M", True), ("R_M", "radius R_M", False),
               ("D_M", "dimension D_M", False), ("center_corr", "center correlation", False)]
    V = lambda arch, key: np.array([per[arch][s][key] for s in C.SEEDS], float)
    triples = {}
    print(f"\n--- {SET_NAME}: paired het vs hom (n = 20), uncorrected estimator ---")
    for key, lbl, hib in METRICS:
        r = paired_triple(V("het", key), V("hom", key), lbl, higher_is_better=hib)
        triples[key] = r
        arrow = "het>hom" if hib else "het<hom"
        print(f"{lbl:>20} ({arrow}): het {r['het_mean']:.3f}  hom {r['hom_mean']:.3f}  d {r['delta']:+.3f}  "
              f"dz {r['dz']:+.2f}  perm {S.fmt_p(r['perm_p'], r['perm_floor'])}  wins {r['wins']}/{r['n']}",
              flush=True)

    # does capacity on the trained partition track the behaviour it is supposed to explain
    sb = json.load(open(OUT / "summary.json"))["scoreboard"]
    acc = {a: np.array([sb["Task accuracy"][f"{a}_per_seed"][str(s)] for s in C.SEEDS], float)
           for a in ("het", "hom")} if "het_per_seed" in sb["Task accuracy"] else None
    rho = None
    if acc is not None:
        d_alpha = V("het", "alpha_M") - V("hom", "alpha_M")
        d_acc = acc["het"] - acc["hom"]
        rho = float(np.corrcoef(d_alpha, d_acc)[0, 1])
        print(f"\n  corr(per-seed capacity gain, per-seed accuracy gain) = {rho:+.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [("alpha_M", "capacity $\\alpha_M$ (corrected)", True), ("R_M", "radius $R_M$", False),
              ("D_M", "dimension $D_M$", False), ("center_corr", "center correlation", False)]
    for ax, (key, ylab, hib) in zip(axes, panels):
        r = triples[key]
        S.paired_slopes(ax, V("het", key), V("hom", key))
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab}\n({'higher = more separable' if hib else 'lower = more separable'})", fontsize=10)
        ax.text(0.5, 0.03, f"$\\Delta$ {r['delta']:+.3f}  $d_z$ {r['dz']:+.2f}\n"
                           f"perm {S.fmt_p(r['perm_p'], r['perm_floor'])}  wins {r['wins']}/{r['n']}",
                ha="center", va="bottom", transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92))
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Manifold capacity on the four trained task classes, uncorrected estimator "
                 "(language $\\times$ parity, n = 20)", fontweight="bold", y=1.03)
    S.save(fig, FIGS / "F4_16c_capacity_corr_T_4class.png")

    # merge into the existing summary so all three sets are read from one file
    p = OUT / "capacity_corr_n20_summary.json"
    summ = json.load(open(p))
    summ["paired"][SET_NAME] = triples
    summ.setdefault("by_arch", {})[SET_NAME] = {
        arch: {key: {"mean": float(V(arch, key).mean()), "sd": float(V(arch, key).std(ddof=1))}
               for key, _, _ in METRICS} for arch in ("het", "hom")}
    summ.setdefault("per_network", {})[SET_NAME] = {
        arch: {int(s): {key: per[arch][s][key] for key, _, _ in METRICS} for s in C.SEEDS}
        for arch in ("het", "hom")}
    summ.setdefault("notes", {})[SET_NAME] = (
        "four trained task classes (language x parity), UNCORRECTED estimator: the correlation-corrected "
        "one divides by zero at P = 4 because the centre subspace is degenerate. load P/N = 0.0625, so "
        "alpha_M is an extrapolation of an asymptotic theory and is not comparable in absolute terms "
        "with the corrected 8- and 20-manifold numbers. The geometry terms carry no such assumption and "
        "the paired comparison between arms is valid at any P.")
    if rho is not None:
        summ["notes"][SET_NAME + "_acc_corr"] = rho
    json.dump(summ, open(p, "w"), indent=1)
    print(f"\nmerged {SET_NAME} into {p.name} and wrote F4_16c_capacity_corr_T_4class.png")


if __name__ == "__main__":
    main()

"""Fidelity audit: re-run the two places this extension departs from the package's settings.

Two deviations were introduced deliberately and both need checking, because a deviation that changes a
conclusion is a bug regardless of whether the new choice is defensible.

  Active-unit rule. network_analysis.ipynb keeps units firing in at least 0.5% of samples; RSI_test.ipynb
  keeps any unit that spikes at least once. This extension used the 0.5% rule everywhere, for consistency
  across analyses. That drops the near-silent units from the RSI estimate, which could inflate it, so the RSI
  comparison is re-run here under the package's own looser rule.

  Degree-preserving null count. run_modularity_n20.py averages Q_rand over 10 randomisations; the package
  uses 20. Fewer draws make Q_rand noisier, and Q_rand is already the unreliable half of Q/Q_rand, so the
  sensitivity is measured rather than argued.

A conclusion that survives both is a conclusion that does not depend on either choice.

Run: python code/supervisor_extension/audit_fidelity.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import networkx as nx
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sup_common as SC
from stats_core import paired_triple
import run_rsi_n20 as RS
from run_modularity_n20 import mi_matrix, circ_shift, full_graph, mod_sw, N_SURR

N_JOBS = 14
N_PAIRS_AUDIT = 120          # the package's own setting


def counts_any_spike(seed, arch):
    """Per-trial counts under the package's RSI rule: keep every unit that spikes at least once."""
    a = np.load(SC.cache_path(seed, arch))
    active = np.where(a.sum(1) > 0)[0]
    return a[active].reshape(len(active), SC.N_TRIALS, SC.T_STEPS).sum(2).astype(int), active


def rsi_under(seed, arch, rule):
    C, active = counts_any_spike(seed, arch) if rule == "any" else SC.load_counts(seed, arch)
    n = C.shape[0]
    Yd = SC.targets()["4-class"][None, :].astype(int)
    rng = np.random.default_rng(seed)
    allp = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pairs = [allp[p] for p in rng.choice(len(allp), min(N_PAIRS_AUDIT, len(allp)), replace=False)]
    rows = []
    for (i, j) in pairs:
        aI1 = aI2 = aIj = 0.0
        for _ in range(RS.N_DITHER):
            X1 = C[[i]] + rng.uniform(0, RS.DITHER, size=(1, C.shape[1]))
            X2 = C[[j]] + rng.uniform(0, RS.DITHER, size=(1, C.shape[1]))
            aI1 += RS.mixed_mi_knn(X1, Yd, rng)
            aI2 += RS.mixed_mi_knn(X2, Yd, rng)
            aIj += RS.mixed_mi_knn(np.vstack([X1, X2]), Yd, rng)
        I1, I2, Ij = aI1 / RS.N_DITHER, aI2 / RS.N_DITHER, aIj / RS.N_DITHER
        rows.append((I1, I2, Ij, min(I1, I2), Ij - max(I1, I2), Ij - I1 - I2))
    A = np.array(rows, float)
    keep = A[:, 2] > RS.IJ_MIN
    out = {"n_units": n, "redundancy": A[:, 3].mean(), "synergy": A[:, 4].mean(), "RSI": A[:, 5].mean(),
           "I_joint": A[:, 2].mean()}
    out["RSI_n"] = float((A[keep, 5] / A[keep, 2]).mean()) if keep.sum() >= 10 else np.nan
    out["redundancy_n"] = float((A[keep, 3] / A[keep, 2]).mean()) if keep.sum() >= 10 else np.nan
    return seed, arch, rule, out


def qnorm_with_nulls(seed, arch, n_null):
    """Full-graph Q/Q_rand under a given number of degree-preserving randomisations."""
    Z = np.load(SC.DATA / "structure_matrices_n20.npz")
    W = Z[f"{seed}_{arch}__W"]
    G = full_graph(W)
    m, _ = mod_sw(G, seed, weighted=True, n_null=n_null)
    return seed, arch, n_null, m["Qnorm"]


def main():
    print("=" * 78)
    print("A. RSI under the package's own active-unit rule (any spike) against the 0.5% rule")
    print("=" * 78, flush=True)
    jobs = [(s, a, r) for s in SC.SEEDS for a in ("het", "hom") for r in ("any", "pct")]
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(rsi_under)(s, a, r) for s, a, r in jobs)
    by = {(s, a, r): o for s, a, r, o in res}
    print(f"  ({time.time()-t0:.0f}s)\n")
    print(f"  mean units kept: any-spike {np.mean([by[(s,a,'any')]['n_units'] for s in SC.SEEDS for a in ('het','hom')]):.1f}"
          f"   0.5% rule {np.mean([by[(s,a,'pct')]['n_units'] for s in SC.SEEDS for a in ('het','hom')]):.1f}\n")
    import style as S
    for rule, lbl in (("any", "package rule (any spike)"), ("pct", "this extension (0.5%)")):
        print(f"  --- {lbl} ---")
        for key in ("I_joint", "redundancy", "synergy", "RSI", "redundancy_n", "RSI_n"):
            h = np.array([by[(s, "het", rule)][key] for s in SC.SEEDS], float)
            o = np.array([by[(s, "hom", rule)][key] for s in SC.SEEDS], float)
            r = paired_triple(h, o)
            print(f"    {key:>14}: het {r['het_mean']:>+8.4f} hom {r['hom_mean']:>+8.4f} "
                  f"d {r['delta']:>+8.4f} dz {r['dz']:>+5.2f} p {S.fmt_p(r['perm_p'], r['perm_floor']):>12} "
                  f"wins {r['wins']}/{r['n']}")
        print()

    print("=" * 78)
    print("B. Q/Q_rand sensitivity to the number of degree-preserving randomisations")
    print("=" * 78, flush=True)
    jobs = [(s, a, k) for s in SC.SEEDS for a in ("het", "hom") for k in (10, 20)]
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(qnorm_with_nulls)(s, a, k) for s, a, k in jobs)
    q = {(s, a, k): v for s, a, k, v in res}
    print(f"  ({time.time()-t0:.0f}s)\n")
    for k in (10, 20):
        h = np.array([q[(s, "het", k)] for s in SC.SEEDS])
        o = np.array([q[(s, "hom", k)] for s in SC.SEEDS])
        r = paired_triple(h, o)
        print(f"  n_null={k:>2}: het {r['het_mean']:.3f} hom {r['hom_mean']:.3f} d {r['delta']:+.3f} "
              f"dz {r['dz']:+.2f} p {r['perm_p']:.3f} wins {r['wins']}/{r['n']}")
    d10 = np.array([q[(s, a, 10)] for s in SC.SEEDS for a in ("het", "hom")])
    d20 = np.array([q[(s, a, 20)] for s in SC.SEEDS for a in ("het", "hom")])
    print(f"\n  per-network agreement between 10 and 20 nulls: r = {np.corrcoef(d10, d20)[0,1]:.3f}, "
          f"mean |diff| = {np.abs(d10-d20).mean():.3f}")

    json.dump({"rsi_by_rule": {f"{s}_{a}_{r}": by[(s, a, r)] for s in SC.SEEDS
                               for a in ("het", "hom") for r in ("any", "pct")},
               "qnorm_by_nnull": {f"{s}_{a}_{k}": q[(s, a, k)] for s in SC.SEEDS
                                  for a in ("het", "hom") for k in (10, 20)}},
              open(SC.DATA / "audit_fidelity.json", "w"), indent=1)
    print(f"\nwrote {SC.DATA / 'audit_fidelity.json'}")


if __name__ == "__main__":
    main()

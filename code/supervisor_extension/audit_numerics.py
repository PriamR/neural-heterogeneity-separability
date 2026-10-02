"""Numerical audit of the modularity kernels, and of one claim the write-up makes about partitions.

Three checks, each aimed at a specific way this could be silently wrong.

  1. Finiteness. The instantaneous-JMI bug was a NaN that passed a threshold guard and destroyed a summary
     statistic without raising anything. mi_matrix has the same shape of guard, so every stored matrix is
     checked rather than assumed clean.

  2. Precision. mi_matrix runs in float32. That is safe in principle, because the co-occurrence counts are
     integers below 2^24 and so exactly representable, and because the marginal products cannot cancel the
     way the class-conditional sums did. "Safe in principle" is what was believed about the other kernel too,
     so it is recomputed in float64 and compared.

  3. Partition identity. The write-up says the division-of-labour analysis uses the same full-graph Louvain
     partition as the modularity analysis. The two scripts build the surrogate-corrected weights by different
     accumulation routes, one averaging a float32 stack and one summing into a float64 accumulator, so the
     partitions could in principle differ. If they do, the sentence is wrong and has to change.

Run: python code/supervisor_extension/audit_numerics.py
"""
import sys
from pathlib import Path

import numpy as np
import networkx as nx
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sup_common as SC
from run_modularity_n20 import mi_matrix, circ_shift, full_graph, N_SURR

N_JOBS = 14


def check_one(seed, arch):
    Z = np.load(SC.DATA / "structure_matrices_n20.npz")
    out = {"seed": seed, "arch": arch}

    # 1. finiteness of every stored matrix
    nonfinite = 0
    for f in ("MI_obs", "null_mean", "null_std", "W", "cW"):
        A = Z[f"{seed}_{arch}__{f}"]
        nonfinite += int((~np.isfinite(A)).sum())
    out["nonfinite"] = nonfinite

    # 2. float32 against float64 on the observed MI matrix
    R, _ = SC.load_spikes(seed, arch)
    n = R.shape[0]
    S32 = R.reshape(n, -1).astype(np.float32)
    S64 = R.reshape(n, -1).astype(np.float64)
    M32, M64 = mi_matrix(S32), mi_matrix(S64)
    out["mi_max_abs_diff"] = float(np.abs(M32 - M64).max())
    out["mi_max_rel_diff"] = float((np.abs(M32 - M64) / np.maximum(M64, 1e-12)).max())
    out["mi_finite"] = bool(np.isfinite(M32).all() and np.isfinite(M64).all())

    # 3. does the division-of-labour accumulation give the same partition as the modularity one
    rng = np.random.default_rng(seed)
    stack = np.empty((N_SURR, n, n), np.float32)
    for s in range(N_SURR):
        stack[s] = mi_matrix(circ_shift(R, rng))
    W_stack = np.clip(M32 - stack.mean(0), 0, None)
    np.fill_diagonal(W_stack, 0)

    rng2 = np.random.default_rng(seed)
    acc = np.zeros((n, n), np.float64)
    for _ in range(N_SURR):
        acc += mi_matrix(circ_shift(R, rng2))
    W_acc = np.clip(M32 - acc / N_SURR, 0, None)
    np.fill_diagonal(W_acc, 0)

    out["W_max_abs_diff"] = float(np.abs(W_stack - W_acc).max())
    p1 = sorted(nx.community.louvain_communities(full_graph(W_stack), weight="weight", seed=seed),
                key=len, reverse=True)
    p2 = sorted(nx.community.louvain_communities(full_graph(W_acc), weight="weight", seed=seed),
                key=len, reverse=True)
    lab = lambda part: np.array([next(i for i, c in enumerate(part) if v in c) for v in range(n)])
    a, b = lab(p1), lab(p2)
    from sklearn.metrics import adjusted_rand_score
    out["partition_ari"] = float(adjusted_rand_score(a, b))
    out["partition_identical"] = bool(np.array_equal(a, b))
    # and against what is actually stored
    out["ari_vs_stored"] = float(adjusted_rand_score(a, Z[f"{seed}_{arch}__modid"]))
    print(f"  seed {seed:>4} {arch:<3}  nonfinite {nonfinite}  mi_diff {out['mi_max_abs_diff']:.2e}  "
          f"W_diff {out['W_max_abs_diff']:.2e}  ARI(dol,mod) {out['partition_ari']:.3f}  "
          f"ARI(vs stored) {out['ari_vs_stored']:.3f}", flush=True)
    return out


def main():
    jobs = [(s, a) for s in SC.SEEDS for a in ("het", "hom")]
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(delayed(check_one)(s, a) for s, a in jobs)

    nf = sum(r["nonfinite"] for r in res)
    mid = max(r["mi_max_abs_diff"] for r in res)
    mrd = max(r["mi_max_rel_diff"] for r in res)
    wd = max(r["W_max_abs_diff"] for r in res)
    ari = np.array([r["partition_ari"] for r in res])
    aris = np.array([r["ari_vs_stored"] for r in res])
    ident = sum(r["partition_identical"] for r in res)

    print("\n" + "=" * 72)
    print(f"1. finiteness           : {nf} non-finite entries across all stored matrices")
    print(f"2. float32 vs float64   : max abs diff {mid:.3e} bits, max rel diff {mrd:.3e}")
    print(f"                          all matrices finite: {all(r['mi_finite'] for r in res)}")
    print(f"3. partition identity   : identical on {ident}/40 networks, "
          f"ARI min {ari.min():.3f} mean {ari.mean():.3f}")
    print(f"   against stored modid : ARI min {aris.min():.3f} mean {aris.mean():.3f}")
    print("=" * 72)
    verdict = (nf == 0 and mid < 1e-6 and ari.min() > 0.999 and aris.min() > 0.999)
    print("VERDICT:", "clean" if verdict else "SOMETHING TO LOOK AT")


if __name__ == "__main__":
    main()

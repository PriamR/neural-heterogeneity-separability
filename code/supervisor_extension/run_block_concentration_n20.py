"""How concentrated is functional coupling within communities, measured without a randomisation null.

This exists because the statistic the package headlines, Q/Q_rand on the full graph, turns out to be
unreliable in exactly the regime it is applied to, and the cross-seed null it produces is a property of the
statistic rather than of the networks.

Two things go wrong with it at once. Q_rand is obtained by running Louvain on a randomised graph and taking
the modularity it maximises, and maximised modularity on random data is substantially positive, a long-known
problem (Guimera et al. 2004). On a 47-node graph with about 950 edges that value is large and noisy. On top
of that, at density 0.89 the degree-preserving swap cannot find legal moves, so the randomisation reduces to
permuting weights over a fixed topology, which still leaves Louvain free to find a high-modularity partition.
The denominator therefore carries more variance than the numerator, and the ratio stops tracking the thing it
is supposed to measure: across the forty networks Q/Q_rand correlates with actual within-community weight
concentration at only r = 0.21.

The measure here needs no null at all. Take the partition, and compare the mean edge weight inside
communities against the mean edge weight between them. That ratio is what the block matrix shows: bright
blocks on the diagonal against a dark background. It is reported two ways, on the Louvain partition and on a
spectral partition into a fixed three communities, because the Louvain partition is chosen to maximise
modularity and the two arms do not find the same number of communities; fixing the count removes that route
to a difference.

Writes data_cache/supervisor_extension/block_concentration_n20.json.

Run: python code/supervisor_extension/run_block_concentration_n20.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sup_common as SC
from stats_core import paired_triple

warnings.filterwarnings("ignore")
K_FIXED = 3
ARCHS = ("het", "hom")


def ratio(W, modid):
    """Mean within-community edge weight over mean between-community edge weight.

    Above one means coupling is concentrated inside communities, which is what a visible block structure is.
    No randomisation enters, so nothing here depends on how a null behaves on a dense graph.
    """
    same = modid[:, None] == modid[None, :]
    iu = np.triu_indices(W.shape[0], 1)
    w, s = W[iu], same[iu]
    if not s.any() or not (~s).any():
        return float("nan")
    between = w[~s].mean()
    return float(w[s].mean() / between) if between > 0 else float("nan")


def spectral_partition(W, K, seed):
    from sklearn.cluster import SpectralClustering
    A = np.array(W, dtype=float)
    np.fill_diagonal(A, 0.0)
    A = (A + A.T) / 2.0
    sc = SpectralClustering(n_clusters=K, affinity="precomputed", random_state=seed,
                            assign_labels="kmeans")
    return sc.fit_predict(A)


def main():
    Z = np.load(SC.DATA / "structure_matrices_n20.npz")
    SCAL = json.load(open(SC.DATA / "structure_scalars_n20.json"))
    MOD = json.load(open(SC.DATA / "modularity_n20.json"))

    out = {a: {} for a in ARCHS}
    for a in ARCHS:
        for key in ("louvain", "fixedK", "louvain_cum"):
            out[a][key] = []
    for a in ARCHS:
        for s in SC.SEEDS:
            W = Z[f"{s}_{a}__W"]
            cW = Z[f"{s}_{a}__cW"]
            out[a]["louvain"].append(ratio(W, Z[f"{s}_{a}__modid"]))
            out[a]["fixedK"].append(ratio(W, spectral_partition(W, K_FIXED, s)))
            out[a]["louvain_cum"].append(ratio(cW, Z[f"{s}_{a}__modid_c"]))
            print(f"  seed {s:>4} {a:<3}  within/between {out[a]['louvain'][-1]:6.2f}  "
                  f"(fixed K={K_FIXED}: {out[a]['fixedK'][-1]:6.2f})", flush=True)

    import style as S
    paired = {}
    LBL = {"louvain": "Within/between weight ratio (Louvain partition)",
           "fixedK": f"Within/between weight ratio (fixed K = {K_FIXED})",
           "louvain_cum": "Within/between weight ratio (cumulative graph)"}
    print("\n--- paired het vs hom (n = 20); lower means less concentrated, so less modular ---")
    for key, lbl in LBL.items():
        r = paired_triple(np.array(out["het"][key]), np.array(out["hom"][key]), lbl)
        paired[key] = r
        print(f"{lbl:>52}: het {r['het_mean']:>6.2f}  hom {r['hom_mean']:>6.2f}  d {r['delta']:>+6.2f}  "
              f"dz {r['dz']:>+5.2f}  perm {S.fmt_p(r['perm_p'], r['perm_floor']):>12}  "
              f"het higher {r['wins']}/{r['n']}")

    # how well does the package's headline statistic track the thing the block matrices actually show
    allr = np.concatenate([out["het"]["louvain"], out["hom"]["louvain"]])
    allq = np.concatenate([[MOD["per_network"][f"{s}_{a}"]["full"]["Qnorm"] for s in SC.SEEDS]
                           for a in ARCHS])
    rho = float(np.corrcoef(allr, allq)[0, 1])
    print(f"\n  corr(within/between ratio, full-graph Q/Q_rand) over 40 networks = {rho:+.3f}")
    print("  the headline statistic barely tracks the structure its own figures display")

    json.dump({"analysis": "within_between_weight_concentration", "n_pairs": len(SC.SEEDS),
               "seeds": SC.SEEDS, "k_fixed": K_FIXED, "paired": paired,
               "corr_ratio_vs_Qnorm": rho,
               "per_network": {a: {str(s): {k: out[a][k][i] for k in LBL}
                                   for i, s in enumerate(SC.SEEDS)} for a in ARCHS}},
              open(SC.DATA / "block_concentration_n20.json", "w"), indent=1)
    print(f"\nwrote {SC.DATA / 'block_concentration_n20.json'}")


if __name__ == "__main__":
    main()

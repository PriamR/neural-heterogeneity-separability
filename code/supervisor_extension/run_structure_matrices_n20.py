"""Recompute and store the edge matrices and partitions behind the modularity result, for all twenty seeds.

run_modularity_n20.py saved only scalars, which is enough for the paired tests but not for the structural
figures the package leads with: the community-ordered adjacency block matrix, the spring layout, the MI
heatmaps and the community-by-class tuning grids. Those need the matrices themselves, so this script rebuilds
them once and writes them to disk, and every structural figure is drawn from the stored arrays afterwards.

Everything here matches run_modularity_n20.py exactly, down to the random seeds, so the Q/Q_rand values
recomputed from these matrices reproduce the ones already reported. That is asserted at the end rather than
assumed.

Writes data_cache/supervisor_extension/structure_matrices_n20.npz, roughly 40 networks of n-by-n float32.

Run: python code/supervisor_extension/run_structure_matrices_n20.py
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
from run_modularity_n20 import (mi_matrix, circ_shift, knn_mi_matrix, full_graph, mod_sw,
                                N_SURR, N_NULL)

N_JOBS = 14


def analyse(seed, arch):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    R, active = SC.load_spikes(seed, arch)
    n = R.shape[0]

    MI_obs = mi_matrix(R.reshape(n, -1).astype(np.float32))
    stack = np.empty((N_SURR, n, n), np.float32)
    for s in range(N_SURR):
        stack[s] = mi_matrix(circ_shift(R, rng))
    null_mean, null_std = stack.mean(0), stack.std(0)
    W = np.clip(MI_obs - null_mean, 0, None)
    np.fill_diagonal(W, 0)
    sig = MI_obs > np.percentile(stack, 99, axis=0)
    np.fill_diagonal(sig, False)

    G = full_graph(W)
    m, part = mod_sw(G, seed, weighted=True)
    modid = np.zeros(n, int)
    for ci, com in enumerate(sorted(part, key=len, reverse=True)):
        for v in com:
            modid[v] = ci

    CNT = R.sum(2)
    _, _, cW = knn_mi_matrix(CNT.astype(float), np.random.default_rng(seed))
    Gc = full_graph(cW)
    mc, partc = mod_sw(Gc, seed, weighted=True)
    modid_c = np.zeros(n, int)
    for ci, com in enumerate(sorted(partc, key=len, reverse=True)):
        for v in com:
            modid_c[v] = ci

    # community-by-class tuning, for the heatmap grids
    T = SC.targets()
    Y, lang, par = T["4-class"], T["language"], T["parity"]
    net = CNT.sum(0) + 1e-9
    tun = {}
    for key, pp, mid in (("binned", part, modid), ("cumulative", partc, modid_c)):
        pp = sorted(pp, key=len, reverse=True)
        mat = np.zeros((len(pp), 4))
        frac = np.zeros((len(pp), 4))
        e4, el, ep, e4f = [], [], [], []
        for ci, com in enumerate(pp):
            a = CNT[sorted(com)].mean(0).astype(float)
            fr = CNT[sorted(com)].sum(0) / net
            for k in range(4):
                mat[ci, k] = a[Y == k].mean()
                frac[ci, k] = fr[Y == k].mean()
            e4.append(SC.eta2(a, Y, 4))
            el.append(SC.eta2(a, lang, 2))
            ep.append(SC.eta2(a, par, 2))
            e4f.append(SC.eta2(fr, Y, 4))
        tun[key] = dict(mat=mat, frac=frac, eta2_4c=np.array(e4), eta2_lang=np.array(el),
                        eta2_par=np.array(ep), eta2_4c_frac=np.array(e4f),
                        sizes=np.array([len(c) for c in pp]))

    print(f"  seed {seed:>4} {arch:<3}  n={n:<3} Qn {m['Qnorm']:.3f}  cumQn {mc['Qnorm']:.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return dict(seed=seed, arch=arch, n=n, active=active, MI_obs=MI_obs, null_mean=null_mean,
                null_std=null_std, W=W, sig=sig, modid=modid, cW=cW, modid_c=modid_c,
                Qnorm=m["Qnorm"], Q=m["Q"], sigma=m["sigma"], ncomm=m["ncomm"], density=m["density"],
                cQnorm=mc["Qnorm"], cQ=mc["Q"], csigma=mc["sigma"], cncomm=mc["ncomm"],
                cdensity=mc["density"], tun=tun)


def main():
    jobs = [(s, a) for s in SC.SEEDS for a in ("het", "hom")]
    print(f"structure matrices for {len(jobs)} networks\n", flush=True)
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(delayed(analyse)(s, a) for s, a in jobs)

    store = {}
    scal = {}
    for r in res:
        k = f"{r['seed']}_{r['arch']}"
        for f in ("MI_obs", "null_mean", "null_std", "W", "cW"):
            store[f"{k}__{f}"] = r[f].astype(np.float32)
        store[f"{k}__sig"] = r["sig"]
        for f in ("modid", "modid_c", "active"):
            store[f"{k}__{f}"] = r[f].astype(np.int16)
        for key in ("binned", "cumulative"):
            for f, v in r["tun"][key].items():
                store[f"{k}__tun_{key}_{f}"] = np.asarray(v, np.float32)
        scal[k] = {f: float(r[f]) for f in ("Qnorm", "Q", "sigma", "ncomm", "density",
                                            "cQnorm", "cQ", "csigma", "cncomm", "cdensity")}
        scal[k]["n"] = int(r["n"])

    p = SC.DATA / "structure_matrices_n20.npz"
    np.savez_compressed(p, **store)
    json.dump(scal, open(SC.DATA / "structure_scalars_n20.json", "w"), indent=1)

    # the stored matrices must reproduce the already-reported metrics, or the figures would describe
    # a different computation than the tables do
    M = json.load(open(SC.DATA / "modularity_n20.json"))
    bad = []
    for k, v in scal.items():
        s, a = k.rsplit("_", 1)
        ref = M["per_network"][k]
        for got, want in ((v["Qnorm"], ref["full"]["Qnorm"]), (v["cQnorm"], ref["cumulative"]["Qnorm"])):
            if abs(got - want) > 1e-9:
                bad.append((k, got, want))
    print(f"\nwrote {p} ({p.stat().st_size/1e6:.1f} MB)")
    if bad:
        print(f"  MISMATCH against modularity_n20.json on {len(bad)} entries: {bad[:4]}")
    else:
        print("  all 40 networks reproduce their reported Q/Q_rand exactly")


if __name__ == "__main__":
    main()

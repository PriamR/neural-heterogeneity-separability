"""Standalone NuMIT (TDMI-matched discrete null) z-scoring for ONE (seed, arch) network.

Same robust per-network-process pattern as the shuffle runner. The null here replaces time-shuffling
with NuMIT: random logic-gate transitions + flip-noise calibrated to the observed TDMI (bivariate for
k=2, multivariate for k>=4). Tests whether observed M exceeds the *pairwise* baseline. CPU-only,
resumable. Spawn one per network via run_numit_driver.py.
"""
import argparse, sys, json, time, hashlib, io, math, importlib
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
import torch

PR = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
PF = PR / "Project Files"
PIPE = PF / "Discrete Zscore Pipeline"
PAPER = PR / "neural_heterogeneity" / "SuGD_code"
SHD = PR / "data" / "shd" / "shd_test.h5"
CKPT = PF / "Checkpoints" / "SeededRuns" / "seeded_run_1_64n" / "4class_lognormal"
CACHE = PIPE / "outputs" / "spk_cache"
OUT = PIPE / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
RECORDS = OUT / "numit_zscore_records.jsonl"
ARCH_TMPL = {"het": "4class_seed{seed}_het_lognormal.pt", "hom": "4class_seed{seed}_hom_geommean_lognormal.pt"}
for p in (str(PIPE), str(PAPER)):
    if p not in sys.path:
        sys.path.insert(0, p)
from wimfo.W_M_Info import W_M_calculator
from numit_discrete_optimized import find_transition_for_target, find_transition_for_target_multivariate
RSNN = importlib.import_module("model").RSNN
from data_gen import open_file
DEV = torch.device("cpu")  # scoring needs no GPU; spikes are cached. Avoids CUDA multiproc issues.


def _wm(*a, **k):
    with io.StringIO() as b, redirect_stdout(b):
        return W_M_calculator(*a, **k)

def _ss(*p):
    return int(hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()[:8], 16)

def _enc(rows):
    return (2 ** np.arange(rows.shape[0])).dot(rows)

def observed_wm(sub_spk, tau):
    k = sub_spk.shape[0]; half = k // 2; A = 2 ** half
    g1, g2 = sub_spk[:half], sub_spk[half:]
    data = np.array([_enc(g1[:, :-tau]), _enc(g2[:, :-tau]), _enc(g1[:, tau:]), _enc(g2[:, tau:])]).astype(int)
    W, M = _wm(data, type="discrete", option="data", alphabet_size=A)
    return float(W), float(M), A

def numit_null_wm(target_tdmi, k, A, max_tries=500, max_retries=50, smooth=1e-4):
    """One TDMI-matched discrete null draw -> (W, M)."""
    for _ in range(max_retries):
        try:
            if k == 2:
                _, _, _, P = find_transition_for_target(target_tdmi, max_tries=max_tries)
                pmf = P.reshape((2, 2, 2, 2))
            else:
                _, _, _, P = find_transition_for_target_multivariate(target_tdmi, n_vars=k, max_tries=max_tries)
                pmf = P.reshape((A, A, A, A))
            pmf = pmf + smooth; pmf = pmf / pmf.sum()
            W, M = _wm(input=pmf, type="discrete", option="distr", alphabet_size=A)
            if np.isfinite(W) and np.isfinite(M):
                return float(W), float(M)
        except Exception:
            continue
    return float("nan"), float("nan")

def _z(obs, arr):
    arr = np.asarray(arr)
    if arr.size < 2:
        return float("nan")
    s = arr.std()
    return float((obs - arr.mean()) / s) if s > 1e-12 else float("nan")


@torch.no_grad()
def extract_and_cache(seed, arch):
    path = CACHE / f"4class_seed{seed}_{arch}_spk.npy"
    if path.exists():
        return np.load(path)
    ck = torch.load(CKPT / ARCH_TMPL[arch].format(seed=seed), map_location=DEV, weights_only=False)
    prms = dict(ck["prms"]); prms["dtype"] = torch.float; prms["device"] = DEV; prms["cuda"] = False
    model = RSNN(prms, rec=True).to(DEV); model.load_state_dict(ck["model_state_dict"]); model.eval()
    u, t, l = open_file(str(SHD)); units = list(u[:]); times = list(t[:]); labels = np.array(l[:]); u._v_file.close()
    nbh = int(model.network[0].output_size); ns = int(prms["nb_steps"]); ni = int(prms["nb_inputs"])
    bs = int(prms["batch_size"]); inv = 1.0 / float(prms["time_step"]); stride = 4
    si = np.where(np.isin(labels, prms["class_list"]))[0]; nb = min(8, -(-len(si) // bs)); ch = []
    for b in range(nb):
        bi = si[bs*b: min(len(si), bs*(b+1))]; ab = len(bi)
        ta = [np.round(times[i]*inv).astype(np.int64) for i in bi]; ua = [units[i] for i in bi]
        ln = np.array([len(a) for a in ta]); ts = np.concatenate(ta); us = np.concatenate(ua)
        bc = np.repeat(np.arange(ab, dtype=np.int64), ln); v = ts < ns; ts, us, bc = ts[v], us[v], bc[v]
        x = torch.sparse_coo_tensor(torch.from_numpy(np.stack([bc, ts, us])),
                                    torch.ones(ts.size, dtype=torch.float32),
                                    torch.Size([ab, ns, ni])).to_dense().clamp(max=1.0)
        ch.append(np.transpose(model(0, 0, x)[0][2][:, ::stride, :].numpy(), (2, 0, 1)).reshape(nbh, -1))
    spk = np.concatenate(ch, axis=1); np.save(path, spk.astype(np.uint8))
    return spk.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arch", choices=["het", "hom"], required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--n-subsets", type=int, default=50)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--tau", type=int, default=1)
    ap.add_argument("--t-limit", type=int, default=128000)
    ap.add_argument("--records", default=None, help="override records file path")
    args = ap.parse_args()
    seed, arch = args.seed, args.arch
    records_path = Path(args.records) if args.records else RECORDS

    done = set()
    if records_path.exists():
        for line in open(records_path):
            try:
                r = json.loads(line); done.add((r["seed"], r["arch"], r["k"], r["subset_draw"]))
            except Exception:
                pass

    spk = extract_and_cache(seed, arch).astype(int)
    if args.t_limit and spk.shape[1] > args.t_limit:
        spk = spk[:, :args.t_limit]
    active = np.where(spk.sum(1) > 0)[0]; spk_active = spk[active]
    n_active, T = spk_active.shape
    print(f"[{seed}/{arch}] active={n_active}/{spk.shape[0]} T={T}", flush=True)

    rec_file = open(records_path, "a")
    try:
        for k in args.ks:
            if k > n_active:
                print(f"[{seed}/{arch}] k={k}: only {n_active} active, skip", flush=True); continue
            rng = np.random.default_rng(_ss(seed, arch, k)); seen, subsets = set(), []
            cap = min(args.n_subsets, math.comb(n_active, k))
            while len(subsets) < cap:
                s = tuple(sorted(rng.choice(n_active, size=k, replace=False).tolist()))
                if s not in seen: seen.add(s); subsets.append(s)
            todo = [(i, s) for i, s in enumerate(subsets) if (seed, arch, k, i) not in done]
            if not todo:
                print(f"[{seed}/{arch}] k={k}: done ({len(subsets)})", flush=True); continue
            t0 = time.time()
            for i, sub in todo:
                A = 2 ** (k // 2)
                try:
                    W, M, A = observed_wm(spk_active[list(sub)], args.tau)
                    rec = {"W_obs": W, "M_obs": M, "tdmi": W + M, "alphabet": A}
                    if np.isfinite(W) and np.isfinite(M):
                        np.random.seed(_ss(seed, arch, k, i) % (2**32)); import random; random.seed(_ss(seed, arch, k, i))
                        tdmi = W + M; nW, nM = [], []
                        for _ in range(args.n_null):
                            w, m = numit_null_wm(tdmi, k, A)
                            if np.isfinite(w) and np.isfinite(m): nW.append(w); nM.append(m)
                        nW, nM = np.asarray(nW), np.asarray(nM)
                        rec.update(status="ok", n_null_valid=int(nM.size),
                                   nullW_mean=float(nW.mean()) if nW.size else float("nan"),
                                   nullM_mean=float(nM.mean()) if nM.size else float("nan"),
                                   nullM_std=float(nM.std()) if nM.size else float("nan"),
                                   zW=_z(W, nW), zM=_z(M, nM))
                    else:
                        rec.update(status="observed_failed", zW=float("nan"), zM=float("nan"), n_null_valid=0)
                except Exception as exc:
                    rec = {"status": f"error:{type(exc).__name__}", "zM": float("nan"), "zW": float("nan")}
                rec.update(seed=seed, arch=arch, k=k, subset_draw=i, n_active=int(n_active), neurons=[int(x) for x in sub])
                rec_file.write(json.dumps(rec) + "\n"); rec_file.flush()
            print(f"[{seed}/{arch}] k={k}: {len(todo)} subsets in {time.time()-t0:.1f}s", flush=True)
    finally:
        rec_file.close()
    print(f"[{seed}/{arch}] DONE", flush=True)


if __name__ == "__main__":
    main()

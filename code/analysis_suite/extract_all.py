"""One forward pass per network over the full SHD test set, from which every metric in the suite is computed.

Covers all 91 nets in checkpoints/all_networks_4class_64n: 20 learned pairs, 10 scaled-init homs, the 40-net
fixed-tau ladder, and the extra fixed-200 ms net.

Per network it computes, from that single pass:
  acc            task accuracy, readout = argmax of the max-over-time output membrane (the real readout)
  active, rate   recruitment (units with >= 1 spike) and mean firing rate
  PCA            eigen-spectrum of the time-resolved [(trials x time) x 64] matrix, at native 1 ms and at a
                 4 ms stride (the 4 ms one only for continuity with the earlier numbers). Gives n90, PR,
                 top-5 and top-10 variance.
  pc_curve       PC-reconstruction accuracy vs number of PCs, rank-m reconstruction fed through the network's
                 OWN output membrane at native decays. At m = 64 this must reproduce acc exactly; the script
                 asserts that, which is the validation that the probe is faithful.
  gain           spectral radius of the recurrent weight matrix
  tau            hidden membrane time constants

Everything is numpy/torch only. sklearn is deliberately not imported: it pulls a second OpenMP runtime into
the same process as torch and segfaults on this machine.

Naming note: the rung whose files say `fixed2ms` actually has tau = 3.259 ms, because the inherited clipper in
code/model_perez_nieves/clipper.py clamps beta to [2/e, 0.995]. We label it "2 ms" throughout at the user's
request; `tau_actual_ms` in every record carries the true value. See the MANIFEST in the checkpoint folder.

Run: python code/analysis_suite/extract_all.py            # all nets, resumable
     python code/analysis_suite/extract_all.py --limit 2  # benchmark on two nets
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

RROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
REPO = RROOT / "Paper Repo (Clean)"
sys.path.insert(0, str(REPO / "code" / "training"))
sys.path.insert(0, str(REPO / "code" / "model_perez_nieves"))
import seeded_runs_common as src
from data_gen import open_file

CK = REPO / "checkpoints" / "all_networks_4class_64n"
OUT = REPO / "data_cache" / "analysis_suite"
OUT.mkdir(parents=True, exist_ok=True)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src.DEVICE = DEV
src.TAU_ARTIFACT_PATH = REPO / "code" / "training" / "Tau from 15 epoch run.json"
src.SHD_TEST = RROOT / "data" / "shd" / "shd_test.h5"
for k, v in {"nb_recurrent": 64, "batch_size": 256}.items():
    src.BASE_PRMS[k] = v

# binary spike counts stay exact in fp32 only if TF32 is off (TF32 truncates the mantissa to 10 bits)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

M_GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
SEEDS_ORIG = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
SEEDS_ALL = SEEDS_ORIG + list(range(1001, 1011))
TAU_ACTUAL = {2.0: 3.259, 3.5: 3.5, 15.0: 15.0, 35.0: 35.0, 200.0: 199.5}


def net_list():
    nets = []
    for s in SEEDS_ALL:
        nets.append(dict(name=f"4class_seed{s}_het_lognormal", group="learned", arch="het", seed=s, tau_nominal=None))
        nets.append(dict(name=f"4class_seed{s}_hom_geommean_lognormal", group="learned", arch="hom", seed=s, tau_nominal=None))
    # no-weight-decay hom retrains for the 9 originally-confounded seeds (202..930). These become the clean
    # homogeneous arm for the main comparison; the WD versions above stay as the control-family baseline.
    for s in [202, 210, 340, 440, 550, 660, 710, 820, 930]:
        nets.append(dict(name=f"4class_seed{s}_hom_geommean_nowd_lognormal", group="learned_nowd", arch="hom_nowd",
                         seed=s, tau_nominal=None))
    for s in SEEDS_ORIG:
        nets.append(dict(name=f"4class_seed{s}_hom_scaledinit_inputrec_lognormal", group="scaled", arch="scaled", seed=s, tau_nominal=None))
    for tag, tau in [("fixed2ms", 2.0), ("fixed3p5ms", 3.5), ("fixed15ms", 15.0), ("fixed35ms", 35.0)]:
        for s in SEEDS_ORIG:
            nets.append(dict(name=f"4class_seed{s}_hom_{tag}_lognormal", group=f"fixed_{tau:g}ms",
                             arch="fixed", seed=s, tau_nominal=tau))
    nets.append(dict(name="4class_seed202_hom_fixed200ms_lognormal", group="fixed_200ms", arch="fixed",
                     seed=202, tau_nominal=200.0))
    return nets


print("loading SHD test set ...", flush=True)
_u, _t, _l = open_file(str(src.SHD_TEST))
UNITS, TIMES, LABELS = list(_u[:]), list(_t[:]), np.array(_l[:])
_u._v_file.close()
print(f"  {len(LABELS)} trials", flush=True)


def eig_stats(C):
    """Eigen-spectrum of a covariance matrix -> explained-variance ratio, n90, PR, top-k mass."""
    ev = np.linalg.eigvalsh(C)[::-1].clip(min=0)
    tot = ev.sum()
    if tot <= 0:
        return dict(evr=[0.0] * 64, n90=float("nan"), pr=float("nan"), top5=float("nan"), top10=float("nan"))
    evr = ev / tot
    return dict(
        evr=[float(v) for v in evr],
        n90=int(np.searchsorted(np.cumsum(evr), 0.90) + 1),
        pr=float((ev.sum() ** 2) / np.square(ev).sum()),
        top5=float(evr[:5].sum()),
        top10=float(evr[:10].sum()),
    )


def membrane_max(drive, alpha, beta):
    """Run the output membrane at native decays and take the max over time. drive [B,T,4] on device.
    Mirrors run_pc_reconstruction_1ms.membrane_logits exactly (loop to T-1, mem starts at 0)."""
    B, T, C = drive.shape
    syn = torch.zeros(B, C, device=drive.device)
    mem = torch.zeros(B, C, device=drive.device)
    best = torch.full((B, C), -1e30, device=drive.device)
    for ti in range(T - 1):
        syn = alpha * syn + drive[:, ti]
        mem = beta * mem + (1 - beta) * syn
        best = torch.maximum(best, mem)
    return best


def decode_cv(X, y, folds=5, seed=0):
    """5-fold linear decode (least squares to one-hot). numpy only."""
    rng = np.random.default_rng(seed)
    n = len(y)
    order = rng.permutation(n)
    oh = np.eye(4)[y]
    accs = []
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        A = np.c_[X[tr], np.ones(len(tr))]
        B = np.c_[X[te], np.ones(len(te))]
        W = np.linalg.lstsq(A, oh[tr], rcond=None)[0]
        accs.append(float(((B @ W).argmax(1) == y[te]).mean()))
    return float(np.mean(accs))


@torch.no_grad()
def analyse(rec):
    path = CK / f"{rec['name']}.pt"
    ck = torch.load(path, map_location=DEV, weights_only=False)
    prms = dict(ck["prms"])
    prms.update({"dtype": torch.float, "device": DEV, "cuda": DEV.type == "cuda"})
    prms["task_label_map"] = src.TASKS["4class"]["task_label_map"]
    prms["class_list"] = list(range(20))
    model = src.RSNN(prms, rec=True).to(DEV)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    sd = ck["model_state_dict"]
    w_out = sd["network.1.w"].to(DEV).float()
    alpha = float(sd["network.1.alpha"].cpu().numpy().ravel()[0])
    beta_out = float(sd["network.1.beta"].cpu().numpy().ravel()[0])

    # ---- forward the whole test set, keep hidden spikes on device
    H, Y, preds = [], [], []
    for x, y in src.shd_generator(UNITS, TIMES, LABELS, prms, shuffle=False, epoch=0, drop_last=False):
        recs = model(0, 0, x.to(DEV))
        spk = recs[0][2]                       # [B,T,64] hidden spikes
        mem_out = recs[-1][1]                  # [B,T,4] output membrane
        H.append(spk)
        preds.append(mem_out.max(1).values.argmax(1).cpu().numpy())
        Y.append(y.numpy())
    H = torch.cat(H)                            # [N,T,64]
    Y = np.concatenate(Y)
    pred = np.concatenate(preds)
    N, T, _ = H.shape
    acc = float((pred == Y).mean())

    # ---- recruitment
    rate_trial = H.mean(1)                                        # [N,64] per-trial rate
    rate_np = rate_trial.cpu().numpy().astype(np.float64)
    active = int((rate_np.sum(0) > 0).sum())
    mean_rate = float(rate_np.mean())

    # ---- PCA on the time-resolved matrix. X^T X on 0/1 data is exact in fp32 (counts << 2^24) with TF32 off.
    X = H.reshape(-1, 64)                                          # [N*T, 64]
    n_rows = X.shape[0]
    mu = X.mean(0)
    C1 = ((X.T @ X) / n_rows - torch.outer(mu, mu)).double().cpu().numpy()
    pca_1ms = eig_stats(C1)
    X4 = H[:, ::4, :].reshape(-1, 64)
    mu4 = X4.mean(0)
    C4 = ((X4.T @ X4) / X4.shape[0] - torch.outer(mu4, mu4)).double().cpu().numpy()
    pca_4ms = eig_stats(C4)
    del X4, C4

    # ---- PC reconstruction -> the network's own readout
    evals, evecs = np.linalg.eigh(C1)
    V = torch.from_numpy(np.ascontiguousarray(evecs[:, np.argsort(evals)[::-1]].T)).float().to(DEV)  # [64,64] rows=PCs
    Vw = V @ w_out                                                  # [64,4]
    mw = mu @ w_out                                                 # [4]
    pc_curve = []
    for m in M_GRID:
        Am = V[:m].T @ Vw[:m]                                       # [64,4]
        correct = 0
        for b0 in range(0, N, 256):
            hb = H[b0:b0 + 256]                                     # [B,T,64]
            drive = (hb - mu) @ Am + mw                             # [B,T,4]
            logits = membrane_max(drive, alpha, beta_out)
            correct += int((logits.argmax(1).cpu().numpy() == Y[b0:b0 + 256]).sum())
        pc_curve.append(correct / N)

    # ---- validation: a rank-64 reconstruction is the identity, so it must reproduce the live accuracy
    recon_err = abs(pc_curve[-1] - acc)

    # ---- weights and taus
    gain = float(np.max(np.abs(np.linalg.eigvals(sd["network.0.v"].cpu().numpy()))))
    beta_h = sd["network.0.beta"].cpu().numpy().ravel().astype(np.float64)
    tau = -1.0 / np.log(np.clip(beta_h, 1e-9, 1 - 1e-9))
    lin_dec = decode_cv(rate_np, Y)

    hist = ck.get("history", {}).get("test_acc", [])
    out = dict(rec)
    out.update({
        "acc": acc,
        "acc_ckpt": float(hist[-1]) if isinstance(hist, list) and hist else None,
        "n_trials": int(N),
        "active": active,
        "mean_rate": mean_rate,
        "gain": gain,
        "lin_decode": lin_dec,
        "tau_median_ms": float(np.median(tau)),
        "tau_mean_ms": float(tau.mean()),
        "tau_sd_ms": float(tau.std(ddof=1)) if tau.size > 1 else 0.0,
        "tau_cv": float(tau.std(ddof=1) / tau.mean()) if tau.size > 1 and tau.mean() > 0 else 0.0,
        "tau_min_ms": float(tau.min()),
        "tau_max_ms": float(tau.max()),
        "tau_all_ms": [float(v) for v in tau] if tau.size > 1 else [float(tau[0])] * 64,
        "tau_actual_ms": TAU_ACTUAL.get(rec["tau_nominal"]) if rec["tau_nominal"] else None,
        "n90_1ms": pca_1ms["n90"], "pr_1ms": pca_1ms["pr"],
        "top5_1ms": pca_1ms["top5"], "top10_1ms": pca_1ms["top10"], "evr_1ms": pca_1ms["evr"],
        "n90_4ms": pca_4ms["n90"], "pr_4ms": pca_4ms["pr"],
        "top5_4ms": pca_4ms["top5"], "top10_4ms": pca_4ms["top10"], "evr_4ms": pca_4ms["evr"],
        "M_GRID": M_GRID, "pc_curve": pc_curve,
        "recon_err_at_64": recon_err,
    })
    del H, X, model
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    nets = net_list()
    if args.limit:
        nets = nets[:args.limit]
    print(f"{len(nets)} networks, device = {DEV}\n", flush=True)

    bad_recon = []
    for i, rec in enumerate(nets, 1):
        f = OUT / f"{rec['name']}.json"
        if f.exists() and not args.force:
            print(f"[{i}/{len(nets)}] {rec['name']}: cached", flush=True)
            continue
        t0 = time.time()
        try:
            out = analyse(rec)
        except Exception as e:
            print(f"[{i}/{len(nets)}] {rec['name']}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        json.dump(out, open(f, "w"), indent=1)
        flag = ""
        if out["recon_err_at_64"] > 0.005:
            flag = f"  <<< rank-64 recon off by {out['recon_err_at_64']:.4f}"
            bad_recon.append(rec["name"])
        print(f"[{i}/{len(nets)}] {rec['name']}: acc={out['acc']:.3f} active={out['active']:>2} "
              f"n90={out['n90_1ms']:>2} PR={out['pr_1ms']:.1f} gain={out['gain']:.2f} "
              f"tau={out['tau_median_ms']:.1f}ms ({time.time()-t0:.0f}s){flag}", flush=True)

    if bad_recon:
        print(f"\n!!! rank-64 reconstruction did not reproduce live accuracy for {len(bad_recon)} nets:")
        for n in bad_recon:
            print("   -", n)
    else:
        print("\nAll rank-64 reconstructions reproduce live accuracy: the PC probe is faithful.")

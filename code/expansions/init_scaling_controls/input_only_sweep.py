"""Input-only init-scaling sweep (ZERO training). Scale ONLY the input weights (network.0.w), not the recurrent,
to avoid the recurrent-gain runaway seen with input+recurrent. Find c that lifts hom's init mean-rate to het's
TRAINED rate, and check the rate curve is smooth (no explosion). Wider grid since input drive is more linear."""
import sys, time, json
from pathlib import Path
import numpy as np, torch

REPO = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\Paper Repo (Clean)")
sys.path.insert(0, str(REPO / "code" / "training")); sys.path.insert(0, str(REPO / "code" / "model_perez_nieves"))
import seeded_runs_common as src
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"); src.DEVICE = DEVICE
src.TAU_ARTIFACT_PATH = REPO / "code" / "training" / "Tau from 15 epoch run.json"
src.SHD_TEST = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\data\shd\shd_test.h5")
CKPTS = REPO / "checkpoints" / "seeded_run_1_64n" / "4class_lognormal"
for k, v in {"nb_recurrent": 64, "batch_size": 256}.items(): src.BASE_PRMS[k] = v

SEEDS = [202, 340, 660]
GRID = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]     # input-only can need larger c to reach het rate
W_HID, V_HID = "network.0.w", "network.0.v"; N_BATCHES = 2

@torch.no_grad()
def fwd(model, prms, U, T, L, n_batches=N_BATCHES):
    model.eval(); nb_h = int(model.network[0].output_size)
    nb_steps, nb_in = int(prms["nb_steps"]), int(prms["nb_inputs"]); bs = int(prms["batch_size"]); inv = 1.0/float(prms["time_step"])
    idx = np.where(np.isin(L, prms["class_list"]))[0]; n_batches = min(n_batches, -(-len(idx)//bs)); ch = []
    for b in range(n_batches):
        bi = idx[bs*b:min(len(idx), bs*(b+1))]; ab = len(bi)
        ta = [np.round(T[i]*inv).astype(np.int64) for i in bi]; ua = [U[i] for i in bi]; ln = np.array([len(a) for a in ta])
        ts = np.concatenate(ta); us = np.concatenate(ua); bc = np.repeat(np.arange(ab), ln); m = ts < nb_steps
        x = torch.sparse_coo_tensor(torch.from_numpy(np.stack([bc[m], ts[m], us[m]])), torch.ones(int(m.sum()), dtype=torch.float32),
                                    torch.Size([ab, nb_steps, nb_in])).to_dense().clamp(max=1.0).to(DEVICE)
        spk = model(0, 0, x)[0][2].detach().cpu().numpy(); ch.append(np.transpose(spk, (2, 0, 1)).reshape(nb_h, -1).astype(np.uint8))
    return np.concatenate(ch, axis=1)

def n90(spk, stride=4):
    X = spk[:, ::stride].T.astype(np.float64); X = X - X.mean(0, keepdims=True)
    if np.allclose(X, 0): return float("nan")
    ev = np.linalg.eigvalsh((X.T @ X)/(X.shape[0]-1))[::-1].clip(min=0); tot = ev.sum()
    return int(np.searchsorted(np.cumsum(ev)/tot, 0.90)+1) if tot > 0 else float("nan")

def act(spk): return {"rate": float(spk.mean()), "active": int((spk.sum(1) > 0).sum()), "n90": n90(spk)}

def load(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False); prms = dict(ck["prms"])
    prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
    m = src.RSNN(prms, rec=True).to(DEVICE); m.load_state_dict(ck["model_state_dict"]); m.eval(); return m, prms

def interp_c(curve, target):
    g = np.array(GRID); y = np.array([curve[c]["rate"] for c in GRID])
    if y[0] >= target: return 1.0
    if np.all(y < target): return float("inf")
    k = int(np.argmax(y >= target)); x0, x1, y0, y1 = g[k-1], g[k], y[k-1], y[k]
    return float(x0 + (target-y0)*(x1-x0)/(y1-y0)) if y1 != y0 else float(g[k])

print("loading SHD test..."); cc = src.SHDCache(src.SHD_TEST); U, T, L = cc.units, cc.times, cc.labels
t0 = time.time(); rows = []
for s in SEEDS:
    mh, ph = load(CKPTS / f"4class_seed{s}_het_lognormal.pt"); het_tr_rate = float(fwd(mh, ph, U, T, L).mean()); del mh
    pair = src.build_seeded_pair(master_seed=s, task_key="4class", mem_distribution_family="lognormal")
    hom = pair["hom_model"]; op = pair["hom_prms"]; base = {k: v.clone() for k, v in hom.state_dict().items()}
    curve = {}
    for c in GRID:
        sd = {k: v.clone() for k, v in base.items()}; sd[W_HID] = sd[W_HID]*c   # INPUT ONLY
        hom.load_state_dict(sd); curve[c] = act(fwd(hom, op, U, T, L))
    c_star = interp_c(curve, het_tr_rate)
    rows.append({"seed": s, "het_trained_rate": het_tr_rate, "c_match_trained_rate": c_star, "curve": curve})
    print(f"\nseed {s}: het trained rate = {het_tr_rate:.4f}  ->  c_match(input-only) = {c_star:.2f}")
    print("  c    :", "  ".join(f"{c:>6.0f}" for c in GRID))
    print("  rate :", "  ".join(f"{curve[c]['rate']:>6.3f}" for c in GRID))
    print("  activ:", "  ".join(f"{curve[c]['active']:>6.0f}" for c in GRID))
    print("  n90  :", "  ".join(f"{curve[c]['n90']:>6.1f}" for c in GRID))
    del hom
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

cs = [r["c_match_trained_rate"] for r in rows if np.isfinite(r["c_match_trained_rate"])]
print(f"\n(elapsed {time.time()-t0:.0f}s)")
print(f"input-only c_match(het trained rate): per-seed {[round(r['c_match_trained_rate'],2) for r in rows]}  median {np.median(cs):.2f}")
print(f"seed 202 c_match = {rows[0]['c_match_trained_rate']:.2f}  (this is the constant for the local 202 training test)")
out = REPO / "data_cache" / "init_scaling" / "input_only_sweep_3seed.json"; out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"seeds": SEEDS, "grid": GRID, "which": "input_only", "rows": rows}, indent=1))
print("saved ->", out)

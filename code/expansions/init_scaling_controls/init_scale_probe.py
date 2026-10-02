"""Local init-scaling probe: measure het-vs-hom activity at INIT and TRAINED, sweep the hom weight-scale c,
and find/verify the constant that lifts hom's activity to het's. Runs on the local GPU + local SHD test set.
No training. This is the 'test the constant before the full run' check."""
import sys, time
from pathlib import Path
import numpy as np, torch

REPO = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\Paper Repo (Clean)")
sys.path.insert(0, str(REPO / "code" / "training"))
sys.path.insert(0, str(REPO / "code" / "model_perez_nieves"))
import seeded_runs_common as src

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src.DEVICE = DEVICE
src.TAU_ARTIFACT_PATH = REPO / "code" / "training" / "Tau from 15 epoch run.json"
src.SHD_TEST = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\data\shd\shd_test.h5")
CKPTS = REPO / "checkpoints" / "seeded_run_1_64n" / "4class_lognormal"

TASK_KEY, MEM = "4class", "lognormal"
SEEDS = [202, 340, 660]                                  # SMALL batch: 3 learned seeds (no training)
SCALE_GRID = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
WHICH = {"input_w": True, "recurrent_v": True, "output_w": False}
W_HID, V_HID, W_OUT = "network.0.w", "network.0.v", "network.1.w"
N_BATCHES = 1

for k, v in {"nb_recurrent": 64, "batch_size": 256, "nb_epochs": 50}.items():
    src.BASE_PRMS[k] = v

@torch.no_grad()
def fwd_spikes(model, prms, U, T, L, n_batches=N_BATCHES):
    model.eval()
    nb_hidden = int(model.network[0].output_size)
    nb_steps, nb_inputs = int(prms["nb_steps"]), int(prms["nb_inputs"])
    bs, inv_dt = int(prms["batch_size"]), 1.0 / float(prms["time_step"])
    idxs = np.where(np.isin(L, prms["class_list"]))[0]
    n_batches = min(n_batches, -(-len(idxs) // bs))
    chunks = []
    for b in range(n_batches):
        bi = idxs[bs*b:min(len(idxs), bs*(b+1))]; abs_ = len(bi)
        t_arr = [np.round(T[i]*inv_dt).astype(np.int64) for i in bi]; u_arr = [U[i] for i in bi]
        lens = np.array([len(a) for a in t_arr]);
        ts = np.concatenate(t_arr); us = np.concatenate(u_arr)
        bc = np.repeat(np.arange(abs_), lens); m = ts < nb_steps; ts, us, bc = ts[m], us[m], bc[m]
        x = torch.sparse_coo_tensor(torch.from_numpy(np.stack([bc, ts, us])),
                                    torch.ones(ts.size, dtype=torch.float32),
                                    torch.Size([abs_, nb_steps, nb_inputs])).to_dense().clamp(max=1.0).to(DEVICE)
        spk = model(0, 0, x)[0][2].detach().cpu().numpy()
        chunks.append(np.transpose(spk, (2, 0, 1)).reshape(nb_hidden, -1).astype(np.uint8))
    return np.concatenate(chunks, axis=1)

def pca_n90(spk, stride=4):
    X = spk[:, ::stride].T.astype(np.float64); X = X - X.mean(0, keepdims=True)
    if X.shape[0] < 2 or np.allclose(X, 0): return float("nan")
    ev = np.linalg.eigvalsh((X.T @ X) / (X.shape[0]-1))[::-1].clip(min=0); tot = ev.sum()
    return int(np.searchsorted(np.cumsum(ev)/tot, 0.90)+1) if tot > 0 else float("nan")

def act(spk):
    return {"rate": float(spk.mean()), "active": int((spk.sum(1) > 0).sum()), "n90": pca_n90(spk)}

def scaled(sd, c):
    s = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sd.items()}
    if WHICH["input_w"]: s[W_HID] = s[W_HID]*c
    if WHICH["recurrent_v"]: s[V_HID] = s[V_HID]*c
    if WHICH["output_w"]: s[W_OUT] = s[W_OUT]*c
    return s

def load_trained(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    prms = dict(ck["prms"]); prms.update({"dtype": torch.float, "device": DEVICE, "cuda": True})
    m = src.RSNN(prms, rec=True).to(DEVICE); m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m, prms

print("loading SHD test..."); cache = src.SHDCache(src.SHD_TEST); U, T, L = cache.units, cache.times, cache.labels
t0 = time.time()
rows = []
print(f"\n{'seed':>5} | {'INIT het':>16} {'INIT hom c=1':>16} | {'TRAINED het':>16} {'TRAINED hom':>16}")
print(f"{'':>5} | {'rate    active':>16} {'rate    active':>16} | {'rate    active':>16} {'rate    active':>16}")
for s in SEEDS:
    pair = src.build_seeded_pair(master_seed=s, task_key=TASK_KEY, mem_distribution_family=MEM)
    het, hp = pair["hetero_model"], pair["hetero_prms"]
    hom, op = pair["hom_model"], pair["hom_prms"]
    het_sd, hom_sd = {k: v.clone() for k, v in het.state_dict().items()}, {k: v.clone() for k, v in hom.state_dict().items()}
    het.load_state_dict(het_sd); ih = act(fwd_spikes(het, hp, U, T, L))
    # hom sweep
    curve = {}
    for c in SCALE_GRID:
        hom.load_state_dict(scaled(hom_sd, c)); curve[c] = act(fwd_spikes(hom, op, U, T, L))
    # trained references
    thet, thp = load_trained(CKPTS / f"4class_seed{s}_het_lognormal.pt")
    thom, thom_p = load_trained(CKPTS / f"4class_seed{s}_hom_geommean_lognormal.pt")
    th = act(fwd_spikes(thet, thp, U, T, L)); to = act(fwd_spikes(thom, thom_p, U, T, L))
    rows.append({"seed": s, "init_het": ih, "hom_curve": curve, "tr_het": th, "tr_hom": to})
    print(f"{s:>5} | {ih['rate']:.4f}  {ih['active']:>5} | {curve[1.0]['rate']:.4f}  {curve[1.0]['active']:>5} | "
          f"{th['rate']:.4f}  {th['active']:>5} | {to['rate']:.4f}  {to['active']:>5}")
    del het, hom, thet, thom
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

# interpolate c where hom-init reaches a target
def interp_c(curve, key, target):
    g = np.array(SCALE_GRID); y = np.array([curve[c][key] for c in SCALE_GRID], float)
    if y[0] >= target: return 1.0
    if np.all(y < target): return float("inf")
    k = int(np.argmax(y >= target)); x0,x1,y0,y1 = g[k-1],g[k],y[k-1],y[k]
    return float(x0 + (target-y0)*(x1-x0)/(y1-y0)) if y1!=y0 else float(g[k])

print(f"\n(elapsed {time.time()-t0:.0f}s)\n")
print("=== c to lift hom INIT activity to various targets (per seed) ===")
print(f"{'seed':>5} | {'c->het INIT rate':>16} {'c->het INIT act':>16} | {'c->het TRAINED rate':>19} {'c->het TRAINED act':>18}")
cm = {k: [] for k in ["ir","ia","tr","ta"]}
for r in rows:
    c_ir = interp_c(r["hom_curve"], "rate",   r["init_het"]["rate"])
    c_ia = interp_c(r["hom_curve"], "active", r["init_het"]["active"])
    c_tr = interp_c(r["hom_curve"], "rate",   r["tr_het"]["rate"])
    c_ta = interp_c(r["hom_curve"], "active", r["tr_het"]["active"])
    for k,val in zip(["ir","ia","tr","ta"],[c_ir,c_ia,c_tr,c_ta]):
        if np.isfinite(val): cm[k].append(val)
    print(f"{r['seed']:>5} | {c_ir:>16.2f} {c_ia:>16.2f} | {c_tr:>19.2f} {c_ta:>18.2f}")
print(f"\nMEDIAN c:  het-INIT rate={np.median(cm['ir']):.2f}  het-INIT active={np.median(cm['ia']):.2f}  "
      f"het-TRAINED rate={np.median(cm['tr']):.2f}  het-TRAINED active={np.median(cm['ta']):.2f}")

# show the full hom sweep (median across seeds): activity + init dimensionality vs scale
print("\n=== hom INIT activity + n90 vs scale c (median across seeds) ===")
print("  c   :", "  ".join(f"{c:>6.1f}" for c in SCALE_GRID))
med_act  = [np.median([r["hom_curve"][c]["active"] for r in rows]) for c in SCALE_GRID]
med_rate = [np.median([r["hom_curve"][c]["rate"]  for r in rows]) for c in SCALE_GRID]
med_n90  = [np.median([r["hom_curve"][c]["n90"]   for r in rows]) for c in SCALE_GRID]
print("active:", "  ".join(f"{a:>6.0f}" for a in med_act))
print("rate  :", "  ".join(f"{a:>6.3f}" for a in med_rate))
print("n90   :", "  ".join(f"{a:>6.1f}" for a in med_n90))
print(f"\nref medians: het INIT active={np.median([r['init_het']['active'] for r in rows]):.0f} "
      f"rate={np.median([r['init_het']['rate'] for r in rows]):.4f} n90={np.median([r['init_het']['n90'] for r in rows]):.0f} | "
      f"het TRAINED active={np.median([r['tr_het']['active'] for r in rows]):.0f} rate={np.median([r['tr_het']['rate'] for r in rows]):.4f} | "
      f"hom TRAINED active={np.median([r['tr_hom']['active'] for r in rows]):.0f} rate={np.median([r['tr_hom']['rate'] for r in rows]):.4f}")

import json as _json
_out = Path(r"C:\Users\Priya\AppData\Local\Temp\claude\C--Users-Priya-Desktop-Jobber-Stuff\91f4fe99-29b4-4f89-b711-e1b357b74852\scratchpad\probe_data.json")
_out.write_text(_json.dumps({"seeds": SEEDS, "scale_grid": SCALE_GRID, "which_weights": WHICH, "rows": rows}, indent=1))
print("saved data ->", _out)

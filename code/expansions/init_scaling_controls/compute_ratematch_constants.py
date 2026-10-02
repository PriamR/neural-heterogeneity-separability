"""Compute per-seed input+recurrent scale c that matches het's TRAINED mean-rate, for the 9 learned seeds.
Zero training (forward-only). Writes data_cache/init_scaling/ratematch_constants_inputrec.json for the Colab
scaled-init cell. This is the constant validated locally on seed 202 (c=6.87)."""
import sys, json, time
from pathlib import Path
import numpy as np, torch
REPO = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\Paper Repo (Clean)")
sys.path.insert(0, str(REPO / "code" / "training")); sys.path.insert(0, str(REPO / "code" / "model_perez_nieves"))
import seeded_runs_common as src
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"); src.DEVICE = DEVICE
src.TAU_ARTIFACT_PATH = REPO / "code" / "training" / "Tau from 15 epoch run.json"
src.SHD_TEST = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\data\shd\shd_test.h5")
CK = REPO / "checkpoints" / "seeded_run_1_64n" / "4class_lognormal"
for k, v in {"nb_recurrent": 64, "batch_size": 256}.items(): src.BASE_PRMS[k] = v

SEEDS = [202, 210, 340, 440, 550, 660, 710, 820, 930]
GRID = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
W_HID, V_HID = "network.0.w", "network.0.v"     # input + recurrent

@torch.no_grad()
def rate(model, prms, U, T, L, n_batches=2):
    model.eval(); ns, ni = int(prms["nb_steps"]), int(prms["nb_inputs"]); bs = int(prms["batch_size"]); inv = 1.0/float(prms["time_step"])
    idx = np.where(np.isin(L, prms["class_list"]))[0]; n_batches = min(n_batches, -(-len(idx)//bs)); tot = 0.0; n = 0
    for b in range(n_batches):
        bi = idx[bs*b:min(len(idx), bs*(b+1))]; ab = len(bi)
        ta = [np.round(T[i]*inv).astype(np.int64) for i in bi]; ua = [U[i] for i in bi]; ln = np.array([len(a) for a in ta])
        ts = np.concatenate(ta); us = np.concatenate(ua); bc = np.repeat(np.arange(ab), ln); m = ts < ns
        x = torch.sparse_coo_tensor(torch.from_numpy(np.stack([bc[m], ts[m], us[m]])), torch.ones(int(m.sum()), dtype=torch.float32),
                                    torch.Size([ab, ns, ni])).to_dense().clamp(max=1.0).to(DEVICE)
        spk = model(0, 0, x)[0][2]; tot += float(spk.sum().item()); n += spk.numel()
    return tot/n

def load(seed, role):
    fn = f"4class_seed{seed}_{'het' if role=='het' else 'hom_geommean'}_lognormal.pt"
    ck = torch.load(CK / fn, map_location=DEVICE, weights_only=False); prms = dict(ck["prms"])
    prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
    m = src.RSNN(prms, rec=True).to(DEVICE); m.load_state_dict(ck["model_state_dict"]); m.eval(); return m, prms

def scale(sd, c):
    s = {k: v.clone() for k, v in sd.items()}; s[W_HID] = s[W_HID]*c; s[V_HID] = s[V_HID]*c; return s

print("loading SHD test..."); c = src.SHDCache(src.SHD_TEST); U, T, L = c.units, c.times, c.labels
per_seed = {}; t0 = time.time()
print(f"\n{'seed':>5} {'het_tr_rate':>11} {'c_ratematch':>12}")
for s in SEEDS:
    mh, ph = load(s, "het"); het_rate = rate(mh, ph, U, T, L); del mh
    pair = src.build_seeded_pair(master_seed=s, task_key="4class", mem_distribution_family="lognormal")
    hom = pair["hom_model"]; op = pair["hom_prms"]; base = {k: v.clone() for k, v in hom.state_dict().items()}
    ys = []
    for cc in GRID:
        hom.load_state_dict(scale(base, cc)); ys.append(rate(hom, op, U, T, L))
    g = np.array(GRID); y = np.array(ys)
    if y[0] >= het_rate: cstar = 1.0
    elif np.all(y < het_rate): cstar = float(g[-1])   # clamp to max grid if never reaches
    else:
        k = int(np.argmax(y >= het_rate)); cstar = float(g[k-1] + (het_rate-y[k-1])*(g[k]-g[k-1])/(y[k]-y[k-1]))
    per_seed[str(s)] = round(cstar, 3)
    print(f"{s:>5} {het_rate:>11.4f} {cstar:>12.2f}")
    del hom
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

cs = list(per_seed.values())
print(f"\n(elapsed {time.time()-t0:.0f}s)  per-seed c: {per_seed}")
print(f"median {np.median(cs):.2f}  range [{min(cs):.2f}, {max(cs):.2f}]")
out = REPO / "data_cache" / "init_scaling" / "ratematch_constants_inputrec.json"; out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"which": "input_recurrent", "target": "het_trained_mean_rate", "grid": GRID,
                           "per_seed": per_seed, "median": float(np.median(cs))}, indent=1))
print("saved ->", out)

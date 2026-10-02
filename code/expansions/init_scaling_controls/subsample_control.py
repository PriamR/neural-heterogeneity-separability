"""Subsample-het dimensionality control (ZERO training).
Tests: at MATCHED active-neuron count, is het still higher-dimensional than hom? If yes, het's dimensionality
advantage is not merely 'more neurons fire'. Uses trained checkpoints only; forward-at-test, no gradients.

For each seed: forward trained het & hom on the SHD test set -> [neurons, trials*T] binary spikes.
  n90 & participation-ratio (PR) on the time-resolved [(time*trials) x neurons] matrix (4 ms stride, matches
  the corrected-PCA method). het-full vs hom-full, and het SUBSAMPLED to hom's active count (R random draws)."""
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

SEEDS = [202, 210, 340, 440, 550, 660, 710, 820, 930]   # 9 learned; 101 excluded (frozen anomaly)
N_BATCHES = 2; STRIDE = 4; R = 25; RNG = np.random.default_rng(0)

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

def dim_stats(spk_cols):
    """n90 and participation ratio on [samples x neurons] (time-resolved, 4 ms)."""
    X = spk_cols[:, ::STRIDE].T.astype(np.float64); X = X - X.mean(0, keepdims=True)
    if X.shape[1] < 1 or np.allclose(X, 0): return float("nan"), float("nan")
    ev = np.linalg.eigvalsh((X.T @ X)/(X.shape[0]-1))[::-1].clip(min=0); tot = ev.sum()
    if tot <= 0: return float("nan"), float("nan")
    n90 = int(np.searchsorted(np.cumsum(ev)/tot, 0.90)+1); pr = (ev.sum()**2)/(np.square(ev).sum())
    return n90, float(pr)

def load(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False); prms = dict(ck["prms"])
    prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
    m = src.RSNN(prms, rec=True).to(DEVICE); m.load_state_dict(ck["model_state_dict"]); m.eval(); return m, prms

print("loading SHD test..."); c = src.SHDCache(src.SHD_TEST); U, T, L = c.units, c.times, c.labels
rows = []; t0 = time.time()
print(f"\n{'seed':>5} | {'het act':>7} {'hom act':>7} | {'het n90':>7} {'hom n90':>7} {'het@hom-k n90':>13} | {'het PR':>7} {'hom PR':>7} {'het@hom-k PR':>13}")
for s in SEEDS:
    mh, ph = load(CKPTS / f"4class_seed{s}_het_lognormal.pt"); sh = fwd(mh, ph, U, T, L)
    mo, po = load(CKPTS / f"4class_seed{s}_hom_geommean_lognormal.pt"); so = fwd(mo, po, U, T, L)
    het_act = np.where(sh.sum(1) > 0)[0]; hom_act = np.where(so.sum(1) > 0)[0]; k = len(hom_act)
    het_n90, het_pr = dim_stats(sh[het_act]); hom_n90, hom_pr = dim_stats(so[hom_act])
    sub_n90, sub_pr = [], []
    for _ in range(R):
        sub = RNG.choice(het_act, size=k, replace=False); n9, pr = dim_stats(sh[sub]); sub_n90.append(n9); sub_pr.append(pr)
    r = {"seed": s, "het_act": len(het_act), "hom_act": k, "het_n90": het_n90, "hom_n90": hom_n90,
         "het_sub_n90_mean": float(np.mean(sub_n90)), "het_sub_n90_std": float(np.std(sub_n90)),
         "het_pr": het_pr, "hom_pr": hom_pr, "het_sub_pr_mean": float(np.mean(sub_pr)), "het_sub_pr_std": float(np.std(sub_pr))}
    rows.append(r)
    print(f"{s:>5} | {len(het_act):>7} {k:>7} | {het_n90:>7} {hom_n90:>7} {np.mean(sub_n90):>10.1f}   | {het_pr:>7.2f} {hom_pr:>7.2f} {np.mean(sub_pr):>10.2f}   ")
    del mh, mo
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

def col(k): return np.array([r[k] for r in rows], float)
print(f"\n(elapsed {time.time()-t0:.0f}s)  MEDIANS across {len(rows)} seeds")
print(f"  active neurons:        het {np.median(col('het_act')):.0f}   hom {np.median(col('hom_act')):.0f}")
print(f"  n90:  het-full {np.median(col('het_n90')):.1f} | hom-full {np.median(col('hom_n90')):.1f} | het SUBSAMPLED to hom-count {np.median(col('het_sub_n90_mean')):.1f}")
print(f"  PR :  het-full {np.median(col('het_pr')):.2f} | hom-full {np.median(col('hom_pr')):.2f} | het SUBSAMPLED to hom-count {np.median(col('het_sub_pr_mean')):.2f}")
# paired: het-subsampled vs hom, both at matched count
d_n90 = col('het_sub_n90_mean') - col('hom_n90'); d_pr = col('het_sub_pr_mean') - col('hom_pr')
print(f"\n  PAIRED (het@matched-count - hom), n={len(rows)}:")
print(f"    n90 diff: mean {d_n90.mean():+.2f}  ({np.sum(d_n90>0)}/{len(rows)} seeds het higher)")
print(f"    PR  diff: mean {d_pr.mean():+.2f}  ({np.sum(d_pr>0)}/{len(rows)} seeds het higher)")
print("\n  READ: if het@matched-count still > hom, the dimensionality edge is NOT just neuron count.")
out = REPO / "data_cache" / "init_scaling" / "subsample_het_control.json"; out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"seeds": SEEDS, "n_batches": N_BATCHES, "stride": STRIDE, "R": R, "rows": rows}, indent=1))
print("saved ->", out)

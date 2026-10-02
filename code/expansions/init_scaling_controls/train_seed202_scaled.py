"""Local single-seed validation: train seed-202 scaled-init hom (INPUT-only scaling by C_SCALE) and test
whether the elevated activity SURVIVES training (vs the original hom that collapses to ~12 active). If it
survives and the n90 gap to het changes meaningfully, we scale to more seeds on Colab.

Recipe = original hom exactly (single learned tau, WD 1e-4, 50 epochs). Only difference: input weights x C_SCALE
at init. Compares the trained scaled-hom against het and the original (unscaled) hom, both from checkpoints."""
import sys, time, json
from pathlib import Path
import numpy as np, torch

# ============ CONFIG ============
SEED = 202
C_SCALE = 6.87          # seed-202 input+recurrent c that matches het-202's TRAINED rate (0.15); 4.7 was the 3-seed median
WHICH = {"input_w": True, "recurrent_v": True, "output_w": False}    # input + recurrent (the scheme that reaches het's rate)
assert C_SCALE is not None, "set C_SCALE first"

REPO = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\Paper Repo (Clean)")
sys.path.insert(0, str(REPO / "code" / "training")); sys.path.insert(0, str(REPO / "code" / "model_perez_nieves"))
import seeded_runs_common as src
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu"); src.DEVICE = DEVICE
src.TAU_ARTIFACT_PATH = REPO / "code" / "training" / "Tau from 15 epoch run.json"
src.SHD_TRAIN = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\data\shd\shd_train.h5")
src.SHD_TEST  = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\data\shd\shd_test.h5")
CKPTS = REPO / "checkpoints" / "seeded_run_1_64n" / "4class_lognormal"
OUT = REPO / "checkpoints" / "seeded_run_1_64n_controls" / "4class_lognormal"; OUT.mkdir(parents=True, exist_ok=True)
W_HID, V_HID, W_OUT = "network.0.w", "network.0.v", "network.1.w"

# recipe (seeded_run_1_64n) before building the model
for k, v in {"nb_recurrent": 64, "batch_size": 256, "nb_epochs": 50, "early_stop_patience": 10,
             "optimizer": "adam", "weight_decay": 1e-4, "lr": 2e-3, "lr_ab": 2e-3, "lr_patience": 5,
             "lr_factor": 0.5, "lr_min": 5e-5, "compile_model": False}.items():
    src.BASE_PRMS[k] = v
if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True

def scaled(sd, c):
    s = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sd.items()}
    if WHICH["input_w"]: s[W_HID] = s[W_HID]*c
    if WHICH["recurrent_v"]: s[V_HID] = s[V_HID]*c
    if WHICH["output_w"]: s[W_OUT] = s[W_OUT]*c
    return s

@torch.no_grad()
def fwd(model, prms, U, T, L, n_batches=2):
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

def dims(spk_cols, stride=4):
    act = np.where(spk_cols.sum(1) > 0)[0]
    X = spk_cols[act][:, ::stride].T.astype(np.float64); X = X - X.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh((X.T @ X)/(X.shape[0]-1))[::-1].clip(min=0); tot = ev.sum()
    n90 = int(np.searchsorted(np.cumsum(ev)/tot, 0.90)+1) if tot > 0 else float("nan")
    pr = (ev.sum()**2)/(np.square(ev).sum()) if tot > 0 else float("nan")
    return {"active": int(len(act)), "rate": float(spk_cols.mean()), "n90": n90, "pr": float(pr)}

def load(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False); prms = dict(ck["prms"])
    prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
    m = src.RSNN(prms, rec=True).to(DEVICE); m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m, prms, float(ck["history"]["test_acc"][-1])

print(f"=== seed {SEED} scaled-init (input+recurrent x{C_SCALE}) — local training ===", flush=True)
train_cache, test_cache = src.load_default_caches()
U, T, L = test_cache.units, test_cache.times, test_cache.labels

# references (already trained)
mh, ph, het_acc = load(CKPTS / f"4class_seed{SEED}_het_lognormal.pt"); het = dims(fwd(mh, ph, U, T, L)); het["acc"] = het_acc; del mh
mo, po, hom_acc = load(CKPTS / f"4class_seed{SEED}_hom_geommean_lognormal.pt"); hom = dims(fwd(mo, po, U, T, L)); hom["acc"] = hom_acc; del mo
print(f"REF het:  acc {het['acc']:.3f}  active {het['active']}  rate {het['rate']:.4f}  n90 {het['n90']}  PR {het['pr']:.2f}", flush=True)
print(f"REF hom:  acc {hom['acc']:.3f}  active {hom['active']}  rate {hom['rate']:.4f}  n90 {hom['n90']}  PR {hom['pr']:.2f}", flush=True)

# build scaled hom, verify init activity, train
pair = src.build_seeded_pair(master_seed=SEED, task_key="4class", mem_distribution_family="lognormal")
assert pair["metadata"]["linear_sync_verified"]
sc = pair["hom_model"]; sp = dict(pair["hom_prms"]); sp["weight_decay"] = 1e-4
sp.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
assert sp.get("train_hom_ab") == 1 and sp.get("het_ab") == 0
sc.load_state_dict(scaled(sc.state_dict(), C_SCALE))
init_act = dims(fwd(sc, sp, U, T, L))
print(f"scaled hom INIT: active {init_act['active']}  rate {init_act['rate']:.4f}  n90 {init_act['n90']}  (target het rate {het['rate']:.4f})", flush=True)

t0 = time.time()
history = src.train_experiment(sc, sp, train_cache, test_cache)
elapsed = time.time() - t0
out_path = OUT / f"4class_seed{SEED}_hom_scaledinit_inputrec_c{C_SCALE:g}_lognormal.pt"
src.save_model_checkpoint(out_path, sc, history, elapsed, sp,
    extra={"label": out_path.stem, "task": "4class", "task_key": "4class", "pair_seed": SEED,
           "pair_role": "homogeneous_scaled_init", "run_label": "scaled_init_local", "control_type": "hom_scaled_init",
           "init_weight_scale": float(C_SCALE), "scale_which_weights": WHICH, "scale_target_metric": "pop_rate"})

trained = dims(fwd(sc, sp, U, T, L)); trained["acc"] = float(history["test_acc"][-1])
print(f"\n=== RESULT (trained {elapsed/60:.1f} min) ===", flush=True)
print(f"scaled hom TRAINED: acc {trained['acc']:.3f}  active {trained['active']}  rate {trained['rate']:.4f}  n90 {trained['n90']}  PR {trained['pr']:.2f}", flush=True)
print(f"\n  {'':>16} {'acc':>6} {'active':>7} {'rate':>7} {'n90':>5} {'PR':>6}")
for nm, d in [("het (ref)", het), ("hom (ref)", hom), (f"hom-scaled c{C_SCALE:g}", trained)]:
    print(f"  {nm:>16} {d['acc']:>6.3f} {d['active']:>7} {d['rate']:>7.4f} {d['n90']:>5} {d['pr']:>6.2f}")
print(f"\n  SURVIVED? scaled active {trained['active']} vs hom {hom['active']} (bigger = collapse avoided); "
      f"target het active {het['active']}.", flush=True)
out_json = OUT / f"seed{SEED}_scaledinit_result.json"
out_json.write_text(json.dumps({"seed": SEED, "c_scale": C_SCALE, "which": WHICH, "het": het, "hom": hom,
                                "scaled_init": init_act, "scaled_trained": trained, "elapsed_s": elapsed}, indent=1))
print("saved ->", out_path, "\nsaved ->", out_json, flush=True)

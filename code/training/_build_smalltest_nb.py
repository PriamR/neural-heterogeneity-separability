"""Builder: emits 'Small Test - 200ms 2ms Scaled (Colab).ipynb'.

A small Colab probe (8 nets) around the long-tau hom finding:
  2 fixed-tau homs at 200 ms  (seeds 202, 210)   -> does a FIXED long tau work, or only a LEARNED one?
  3 fixed-tau homs at 2 ms    (seeds 202, 210, 340) -> short-arm control
  3 scaled-init homs          (seeds 210, 340, 440) -> per-seed rate-match, input+recurrent

The point of the 200 ms rung: seed 1004's LEARNED hom drifted to the 200 ms clip and reached het-level accuracy
with het-like gain (spectral radius ~2.6). A FIXED 200 ms hom does not get to co-adapt tau and weights together.
If fixed-200 stays low-gain and low-accuracy (like the earlier fixed 15/35 ms), the co-adaptation is what mattered;
if it reaches high gain and high accuracy, a long tau helps on its own. The verification therefore reports the
recurrent spectral radius alongside accuracy and dimensionality.

Resumable; reproduces the seeded_run_1_64n recipe (64 neurons, 4-class, 50 epochs, Adam, lr 2e-3, WD 1e-4).

Regenerate:  python code/training/_build_smalltest_nb.py
"""
import json
from pathlib import Path

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""# Small Test: 200 ms + 2 ms + Scaled homs (Colab)

A small probe (8 networks) around the long-tau hom finding. Requires a **GPU** runtime.

- **2 fixed-tau homs at 200 ms** (seeds 202, 210). Seed 1004's *learned* hom drifted to the 200 ms clip and hit
  het-level accuracy (0.728) with het-like recurrent gain. A *fixed* 200 ms hom cannot co-adapt tau and weights
  together. If it stays low-gain and low-accuracy like the earlier fixed 15/35 ms rungs, the co-adaptation is what
  mattered. If it reaches high gain and accuracy, a long tau helps on its own.
- **3 fixed-tau homs at 2 ms** (seeds 202, 210, 340). Short-arm control.
- **3 scaled-init homs** (seeds 210, 340, 440). Per-seed rate-match, input and recurrent.

Verification reports accuracy, active neurons, n90, participation ratio, and the **recurrent spectral radius**
(gain), which is the through-line separating the high-accuracy nets from the collapsed ones.

---
**Runs both locally and on Colab; the paths cell auto-detects which.**
- **Local (VS Code on your GPU):** just Run All from inside the repo. It finds the repo root, uses the repo's own
  `code/training` and `code/model_perez_nieves`, the SHD data, the checkpoints, and `data_cache/init_scaling`.
  No upload. New nets land in `checkpoints/seeded_run_1_64n_controls/4class_lognormal/`.
- **Colab:** put the `Small Test (Colab Upload)/` folder on Drive, set the runtime to **GPU**, upload
  `ratematch_constants_inputrec.json` into its `init_scaling/`, then Run All.

Resumable either way: finished checkpoints are skipped.
""")

md(r"""## 1 - Colab setup""")
code(r"""
# ========================== 1. Colab setup ==========================
import sys, subprocess
# Robust detection: import (not just check sys.modules), because a Colab runtime driven from VS Code has
# google.colab installed but usually not pre-imported. If you connect VS Code to a Colab GPU, this reads True.
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except Exception:
    IN_COLAB = False
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "tables"], check=False)
print("Colab runtime:", IN_COLAB)
""")

md(r"""## 2 - Paths and imports""")
code(r"""
# ========================== 2. Paths + imports ==========================
import sys
from pathlib import Path
import numpy as np, torch, json, time

if IN_COLAB:
    # ---- COLAB: everything under the uploaded folder on Drive ----
    # >>> EDIT RESEARCH_ROOT only if your Drive folder name differs <<<
    RESEARCH_ROOT = Path("/content/drive/MyDrive/research project (SNN Info Theory)")
    PROJECT_ROOT  = RESEARCH_ROOT / "Small Test (Colab Upload)"
    SHD_DIR       = RESEARCH_ROOT / "data" / "shd"
    CODE, PAPER   = PROJECT_ROOT / "code", PROJECT_ROOT / "code" / "SuGD_code"
    REFCK         = PROJECT_ROOT / "reference_checkpoints"
    CKPT_OUT      = PROJECT_ROOT / "outputs" / "checkpoints"
    SCALING_DIR   = PROJECT_ROOT / "init_scaling"
    TAU_ARTIFACT  = PROJECT_ROOT / "Tau from 15 epoch run.json"
else:
    # ---- LOCAL (VS Code on your GPU): run straight from the repo, no upload needed ----
    _here = Path.cwd()
    REPO = next((c for c in [_here, *_here.parents]
                 if (c / "code" / "training" / "seeded_runs_common.py").exists()), None)
    if REPO is None:  # fallback: edit if you launched the kernel outside the repo
        REPO = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\Paper Repo (Clean)")
    PROJECT_ROOT = REPO
    SHD_DIR      = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\data\shd")  # EDIT if elsewhere
    CODE, PAPER  = REPO / "code" / "training", REPO / "code" / "model_perez_nieves"
    REFCK        = REPO / "checkpoints" / "seeded_run_1_64n" / "4class_lognormal"          # het + learned hom
    CKPT_OUT     = REPO / "checkpoints" / "seeded_run_1_64n_controls" / "4class_lognormal"  # new nets land here
    SCALING_DIR  = REPO / "data_cache" / "init_scaling"
    TAU_ARTIFACT = REPO / "code" / "training" / "Tau from 15 epoch run.json"

CKPT_OUT.mkdir(parents=True, exist_ok=True); FIGDIR = CKPT_OUT.parent

for p in (str(CODE), str(PAPER)):
    if p not in sys.path:
        sys.path.insert(0, p)
import seeded_runs_common as src

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src.DEVICE = DEVICE; src.PROJECT_ROOT = PROJECT_ROOT; src.PROJECT_FILES = PROJECT_ROOT
src.PAPER_ROOT = PAPER; src.CHECKPOINT_ROOT = CKPT_OUT
src.TAU_ARTIFACT_PATH = TAU_ARTIFACT
src.SHD_TRAIN = SHD_DIR / "shd_train.h5"; src.SHD_TEST = SHD_DIR / "shd_test.h5"
assert src.SHD_TRAIN.exists() and src.SHD_TEST.exists(), f"SHD .h5 not found under {SHD_DIR}"
assert src.TAU_ARTIFACT_PATH.exists(), f"tau artifact not found at {src.TAU_ARTIFACT_PATH}"
assert REFCK.exists(), f"reference_checkpoints not found: {REFCK}"
print("mode   :", "COLAB" if IN_COLAB else "LOCAL", "| project:", PROJECT_ROOT)
print("device :", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")
print("output ->", CKPT_OUT)
""")

md(r"""## 3 - Configuration""")
code(r"""
# ========================== 3. Config ==========================
TASK_KEY, MEM_FAMILY = "4class", "lognormal"

# the small batch: (fixed tau in ms -> seeds) and the scaled seeds
FIXED_JOBS   = [(200.0, [202, 210]), (2.0, [202, 210, 340])]   # 2 at 200 ms, 3 at 2 ms
SCALED_SEEDS = [210, 340, 440]                                  # 3 scaled-init (per-seed rate-match)
WHICH_WEIGHTS = {"input_w": True, "recurrent_v": True, "output_w": False}
RATEMATCH_JSON = SCALING_DIR / "ratematch_constants_inputrec.json"
MANUAL_SCALE   = 6.87

# reproduce the seeded_run_1_64n recipe
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
for k, v in {"nb_recurrent": 64, "batch_size": 256, "nb_epochs": 50, "early_stop_patience": 10,
             "optimizer": "adam", "weight_decay": 1e-4, "lr": 2e-3, "lr_ab": 2e-3, "lr_patience": 5,
             "lr_factor": 0.5, "lr_min": 5e-5, "compile_model": False}.items():
    src.BASE_PRMS[k] = v

W_HID, V_HID, W_OUT = "network.0.w", "network.0.v", "network.1.w"
def tau_tag(t): return f"{t:g}".replace(".", "p")
OUT_FIXED  = "4class_seed{s}_hom_fixed{tag}ms_lognormal.pt"
OUT_SCALED = "4class_seed{s}_hom_scaledinit_inputrec_lognormal.pt"
HET_NAME   = "4class_seed{s}_het_lognormal.pt"
HOM_NAME   = "4class_seed{s}_hom_geommean_lognormal.pt"

n_fixed = sum(len(ss) for _, ss in FIXED_JOBS)
print(f"fixed-tau nets: {n_fixed}  (200 ms x{len(FIXED_JOBS[0][1])}, 2 ms x{len(FIXED_JOBS[1][1])})")
print(f"scaled-init nets: {len(SCALED_SEEDS)}  seeds {SCALED_SEEDS}")
print(f"total: {n_fixed + len(SCALED_SEEDS)} networks  ->  {CKPT_OUT}")
""")

md(r"""## 4 - Helpers""")
code(r'''
# ========================== 4. Helpers ==========================
def build_fixed_hom(seed, tau_mem_ms):
    """Homogeneous net with hidden tau_mem FROZEN at tau_mem_ms; init weights = the het seed's (paired)."""
    het_init, _ = src.build_heterogeneous_model(TASK_KEY, seed)
    prms = src._base_prms_for_task(TASK_KEY, seed)
    prms.update({"het_ab": 0, "train_ab": 0, "train_hom_ab": 0})
    prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
    src.set_seed(seed)
    model = src.RSNN(prms, rec=True).to(DEVICE)
    src.sync_non_tau_parameters(het_init, model)
    src.set_layer_taus_ms(model.network[0], tau_syn_ms=src.DEFAULT_HIDDEN_TAU_SYN_MS,
                          tau_mem_ms=float(tau_mem_ms), time_step=prms["time_step"])
    src.set_layer_taus_ms(model.network[1], tau_syn_ms=src.DEFAULT_OUTPUT_TAU_SYN_MS,
                          tau_mem_ms=src.DEFAULT_OUTPUT_TAU_MEM_MS, time_step=prms["time_step"])
    assert sum(p.requires_grad for n, p in model.named_parameters() if ("alpha" in n or "beta" in n)) == 0
    got = src.decay_to_tau_ms(model.network[0].beta.detach().cpu().numpy().ravel(), prms["time_step"])
    assert np.allclose(got, tau_mem_ms, atol=1e-6) and np.unique(np.round(got, 6)).size == 1
    return model, prms

def scaled_state(base_state, c, which=WHICH_WEIGHTS):
    s = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base_state.items()}
    if which.get("input_w"):     s[W_HID] = s[W_HID] * float(c)
    if which.get("recurrent_v"): s[V_HID] = s[V_HID] * float(c)
    if which.get("output_w"):    s[W_OUT] = s[W_OUT] * float(c)
    return s

@torch.no_grad()
def fwd_spikes(model, prms, units, times, labels, n_batches=2):
    model.eval(); nb = int(model.network[0].output_size)
    ns, ni = int(prms["nb_steps"]), int(prms["nb_inputs"]); bs = int(prms["batch_size"]); inv = 1.0/float(prms["time_step"])
    idx = np.where(np.isin(labels, prms["class_list"]))[0]; n_batches = min(n_batches, -(-len(idx)//bs)); ch = []
    for b in range(n_batches):
        bi = idx[bs*b:min(len(idx), bs*(b+1))]; ab = len(bi)
        ta = [np.round(times[i]*inv).astype(np.int64) for i in bi]; ua = [units[i] for i in bi]; ln = np.array([len(a) for a in ta])
        ts = np.concatenate(ta); us = np.concatenate(ua); bc = np.repeat(np.arange(ab), ln); m = ts < ns
        x = torch.sparse_coo_tensor(torch.from_numpy(np.stack([bc[m], ts[m], us[m]])), torch.ones(int(m.sum()), dtype=torch.float32),
                                    torch.Size([ab, ns, ni])).to_dense().clamp(max=1.0).to(DEVICE)
        spk = model(0, 0, x)[0][2].detach().cpu().numpy(); ch.append(np.transpose(spk, (2, 0, 1)).reshape(nb, -1).astype(np.uint8))
    return np.concatenate(ch, axis=1)

def geom(spk, stride=4):
    act = np.where(spk.sum(1) > 0)[0]
    X = spk[act][:, ::stride].T.astype(np.float64); X = X - X.mean(0, keepdims=True)
    if X.shape[1] < 1 or np.allclose(X, 0): return len(act), float(spk.mean()), float("nan"), float("nan")
    ev = np.linalg.eigvalsh((X.T @ X)/(X.shape[0]-1))[::-1].clip(min=0); tot = ev.sum()
    n90 = int(np.searchsorted(np.cumsum(ev)/tot, 0.90)+1) if tot > 0 else float("nan")
    pr = (ev.sum()**2)/(np.square(ev).sum()) if tot > 0 else float("nan")
    return len(act), float(spk.mean()), n90, float(pr)

def spectral_radius(state):
    return float(np.max(np.abs(np.linalg.eigvals(state[V_HID].detach().cpu().numpy()))))

print("helpers ready.")
''')

md(r"""## 5 - Load SHD""")
code(r"""
# ========================== 5. Data ==========================
train_cache, test_cache = src.load_default_caches()
UN, TI, LA = test_cache.units, test_cache.times, test_cache.labels
""")

md(r"""## 6 - Part 1: fixed-tau homs (200 ms and 2 ms)""")
code(r"""
# ========================== 6. Train fixed-tau homs ==========================
for tau, seeds in FIXED_JOBS:
    tag = tau_tag(tau)
    for seed in seeds:
        out_path = CKPT_OUT / OUT_FIXED.format(s=seed, tag=tag)
        if out_path.exists():
            ck = torch.load(out_path, map_location="cpu", weights_only=False)
            print(f"fixed {tau}ms seed {seed}: exists, skip (acc={ck['history']['test_acc'][-1]:.3f})"); continue
        print(f"\n=== fixed tau={tau}ms seed {seed}: training (50 epochs) ===")
        model, prms = build_fixed_hom(seed, tau)
        t0 = time.time(); history = src.train_experiment(model, prms, train_cache, test_cache); el = time.time()-t0
        src.save_model_checkpoint(out_path, model, history, el, prms,
            extra={"label": out_path.stem, "task": TASK_KEY, "task_key": TASK_KEY, "pair_seed": seed,
                   "pair_role": "homogeneous_fixed_tau", "run_label": "small_test", "mem_distribution_family": MEM_FAMILY,
                   "control_type": "hom_tau_frozen_fixed", "fixed_tau_mem_ms": float(tau)})
        print(f"   saved {out_path.name} | acc={history['test_acc'][-1]:.3f} ({el/60:.1f} min)")
print("\nPart 1 complete.")
""")

md(r"""## 7 - Part 2: scaled-init homs""")
code(r"""
# ========================== 7. Train scaled-init homs ==========================
ratematch = json.loads(RATEMATCH_JSON.read_text()).get("per_seed", {}) if RATEMATCH_JSON.exists() else {}
if not ratematch:
    print(f"!! {RATEMATCH_JSON} not found -> using MANUAL_SCALE={MANUAL_SCALE}")
def resolve_scale(seed):
    v = ratematch.get(str(seed))
    return float(v) if (v is not None and np.isfinite(v)) else float(MANUAL_SCALE)

for seed in SCALED_SEEDS:
    out_path = CKPT_OUT / OUT_SCALED.format(s=seed)
    if out_path.exists():
        ck = torch.load(out_path, map_location="cpu", weights_only=False)
        print(f"scaled seed {seed}: exists, skip (c={ck.get('init_weight_scale')}, acc={ck['history']['test_acc'][-1]:.3f})"); continue
    c = resolve_scale(seed)
    pair = src.build_seeded_pair(master_seed=seed, task_key=TASK_KEY, mem_distribution_family=MEM_FAMILY)
    assert pair["metadata"]["linear_sync_verified"], f"linear sync failed for seed {seed}"
    hom = pair["hom_model"]; hom_prms = dict(pair["hom_prms"]); hom_prms["weight_decay"] = 1e-4
    hom_prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
    assert hom_prms.get("train_hom_ab") == 1 and hom_prms.get("het_ab") == 0
    hom.load_state_dict(scaled_state(hom.state_dict(), c))
    print(f"\n=== scaled seed {seed}: c={c:.2f}  training (50 epochs) ===")
    t0 = time.time(); history = src.train_experiment(hom, hom_prms, train_cache, test_cache); el = time.time()-t0
    src.save_model_checkpoint(out_path, hom, history, el, hom_prms,
        extra={"label": out_path.stem, "task": TASK_KEY, "task_key": TASK_KEY, "pair_seed": seed,
               "pair_role": "homogeneous_scaled_init", "run_label": "small_test", "mem_distribution_family": MEM_FAMILY,
               "control_type": "hom_scaled_init", "init_weight_scale": float(c),
               "scale_which_weights": WHICH_WEIGHTS, "scale_target_metric": "het_trained_mean_rate"})
    print(f"   saved {out_path.name} | c={c:.2f} acc={history['test_acc'][-1]:.3f} ({el/60:.1f} min)")
print("\nPart 2 complete.")
""")

md(r"""## 8 - Part 3: verify (accuracy, geometry, gain) against het and learned hom""")
code(r"""
# ========================== 8. Verify ==========================
def load(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False); prms = dict(ck["prms"])
    prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
    m = src.RSNN(prms, rec=True).to(DEVICE); m.load_state_dict(ck["model_state_dict"]); m.eval()
    acc = ck.get("history", {}).get("test_acc", [float("nan")]); acc = acc[-1] if isinstance(acc, list) and acc else float("nan")
    return m, prms, ck, acc

seeds_involved = sorted({s for _, ss in FIXED_JOBS for s in ss} | set(SCALED_SEEDS))
rows = []
for seed in seeds_involved:
    cands = [("het", REFCK / HET_NAME.format(s=seed)), ("hom(learned)", REFCK / HOM_NAME.format(s=seed)),
             ("hom-scaled", CKPT_OUT / OUT_SCALED.format(s=seed))]
    for tau, ss in FIXED_JOBS:
        if seed in ss: cands.append((f"hom-fixed{tau:g}", CKPT_OUT / OUT_FIXED.format(s=seed, tag=tau_tag(tau))))
    for label, p in cands:
        if not p.exists(): continue
        m, prms, ck, acc = load(p); a, r, n90, pr = geom(fwd_spikes(m, prms, UN, TI, LA))
        sr = spectral_radius(ck["model_state_dict"])
        rows.append((seed, label, acc, a, r, n90, pr, sr))
        del m
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

print(f"\n{'seed':>5} {'net':>14} {'acc':>6} {'active':>7} {'rate':>7} {'n90':>5} {'PR':>6} {'gain(sr)':>9}")
for seed, label, acc, a, r, n90, pr, sr in rows:
    print(f"{seed:>5} {label:>14} {acc:>6.3f} {a:>7} {r:>7.4f} {n90:>5} {pr:>6.2f} {sr:>9.2f}")
print("\nREAD: does hom-fixed200 reach het-like gain (sr ~2-2.8) and accuracy, or stay low-gain like a collapsed")
print("      hom? Low gain + low accuracy => the co-adaptation of a LEARNED long tau is what mattered.")
""")

md(r"""## Done
Outputs in `outputs/checkpoints/`. The key comparison is **hom-fixed200 vs het vs hom-scaled** on accuracy and
the recurrent spectral radius (gain). Merge back into
`checkpoints/seeded_run_1_64n_controls/4class_lognormal/` for analysis with the other controls.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "Small Test - 200ms 2ms Scaled (Colab).ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, f"({len(cells)} cells)")

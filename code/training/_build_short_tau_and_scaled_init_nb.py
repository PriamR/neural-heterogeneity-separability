"""Builder: emits 'Short-Tau + Scaled-Init Controls (Colab).ipynb'.

ONE Colab notebook that trains the two remaining homogeneous control families around the *corrected* picture
(learned-hom collapses to ~5 ms, not ~25 ms):

  PART 1 (runs first)  Scaled-init hom control (6 nets, per-seed rate-match).  Rebuild each hom at init, scale
          its input+recurrent weights by the per-seed c that puts its init firing rate at het's TRAINED rate,
          retrain normally (single LEARNED tau, exactly the original hom recipe), and re-measure
          recruitment / n90 / accuracy. Validated locally on seed 202 (c=6.87), where it produced a
          DISSOCIATION (more activity, WORSE accuracy, unstable) -- this runs the 6 remaining seeds to test
          robustness. Consumes data_cache/init_scaling/ratematch_constants_inputrec.json (from
          code/expansions/init_scaling_controls/compute_ratematch_constants.py).

  PART 2 (runs second)  Short fixed-tau controls (2 tau values x 10 seeds = 20 nets).
          The original ladder (15 / 25 / 35 ms) was centred on the ~25 ms *init* geommean and sits entirely
          ABOVE the true ~5 ms learned optimum, so we only ever mapped the long-tau arm. These two SHORT
          fixed rungs (default 2.0 and 3.5 ms) fill in BELOW the optimum, so accuracy can be shown to fall off
          on *both* sides of ~5 ms (a proper dose-response bowl for the "why one tau is short" argument).

  PART 3  Cheap, sklearn-free verification: accuracy ladder + active-neuron count + numpy-PCA n90 for every
          condition present, so you can see immediately whether the dimensionality gap survives matched activity.

Both training parts are RESUMABLE (finished checkpoints are skipped) and reproduce the seeded_run_1_64n recipe
exactly, so the *only* differences from the original nets are (Part 1) the scaled init weights and (Part 2) the
frozen short tau.

Open choices are flagged [CONFIRM] in the config cell (short-tau values; which weights c touches) -- single-line
edits.

Regenerate:  python code/training/_build_short_tau_and_scaled_init_nb.py
"""
import json
from pathlib import Path

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

# ───────────────────────────────────────────────────────────────── intro
md(r"""# Short-tau + Scaled-init Homogeneous Controls (Colab)

Two remaining control families in one notebook, built around the **corrected** picture: the learned homogeneous
net collapses to **~5 ms** (mean 5.4 / median 4.7 across the 9 learned seeds), **not** the ~25 ms geommean
*init*. Requires a **GPU** runtime.

### Part 1 (runs first) — scaled-init hom control (6 nets, per-seed rate-match)
Rebuild each hom at init, scale its input+recurrent weights by the per-seed `c` that puts its init firing rate
at het's **trained** rate, then retrain normally (single *learned* tau, the exact original hom recipe).
Re-measure recruitment / n90 / accuracy. **Validated locally on seed 202 (c=6.87):** this did not recover het's
accuracy. More activity but lower accuracy and a redundant, hyperactive code, with an unstable training curve.
We run these 6 to confirm the dissociation is robust. Uses `ratematch_constants_inputrec.json`.

### Part 2 (runs second) — short fixed-tau controls (2 taus x 10 seeds)
The original fixed ladder 15 / 25 / 35 ms was centred on the ~25 ms init and sits entirely above the ~5 ms
learned optimum, so it only mapped the long arm (accuracy falls from 5 to 15 to 35). These two short rungs
(default 2.0 and 3.5 ms) fill in below the optimum so accuracy can be shown to fall off on both sides of ~5 ms,
the dose-response bowl the "a single tau is forced to be short" argument needs.

### Part 3 — verification (sklearn-free)
Accuracy ladder + active-neuron count + numpy-PCA n90 for every condition present.

---
**Before running:** (1) put this folder on Google Drive; (2) Runtime → Change runtime type → **GPU**;
(3) set `RESEARCH_ROOT` in the paths cell; (4) (Part 1) upload `ratematch_constants_inputrec.json` into this
folder's `init_scaling/`, or set a manual scale in the config cell; (5) Run all. Both training parts resume.
""")

# ───────────────────────────────────────────────────────────────── 1 colab
md(r"""## 1 — Colab setup & dependencies""")
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
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "tables"], check=False)  # PyTables for SHD .h5
print("Colab runtime:", IN_COLAB)
""")

# ───────────────────────────────────────────────────────────────── 2 paths
md(r"""## 2 — Paths & imports

`seeded_runs_common` bundles the paired-training machinery; the SNN model code is under `code/SuGD_code`. We add
both to `sys.path` **before** importing, then re-point the module's (Windows) paths at this Colab folder.""")
code(r"""
# ========================== 2. Paths + imports ==========================
import sys
from pathlib import Path
import numpy as np, torch, copy, json, time

if IN_COLAB:
    # ---- COLAB: everything under the uploaded folder on Drive ----
    # >>> EDIT RESEARCH_ROOT only if your Drive folder name differs <<<
    RESEARCH_ROOT = Path("/content/drive/MyDrive/research project (SNN Info Theory)")
    PROJECT_ROOT  = RESEARCH_ROOT / "Short-Tau + Scaled-Init Controls (Colab Upload)"
    SHD_DIR       = RESEARCH_ROOT / "data" / "shd"
    CODE, PAPER   = PROJECT_ROOT / "code", PROJECT_ROOT / "code" / "SuGD_code"
    REFCK         = PROJECT_ROOT / "reference_checkpoints"
    CKPT_OUT      = PROJECT_ROOT / "outputs" / "checkpoints"
    SCALING_DIR   = PROJECT_ROOT / "init_scaling"
    TAU_ARTIFACT  = PROJECT_ROOT / "Tau from 15 epoch run.json"
else:
    # ---- LOCAL (VS Code with a local kernel): run straight from the repo, no upload needed ----
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

# ───────────────────────────────────────────────────────────────── 3 config
md(r"""## 3 — Configuration

Reproduces the `seeded_run_1_64n` recipe exactly (64 neurons, 4-class, 50 epochs, Adam, lr 2e-3, WD 1e-4). The
only per-part differences are the scaled init weights (Part 1) and the frozen short tau (Part 2).

**[CONFIRM] with Marcus & Hardik** (all single-line edits):
- `SHORT_FIXED_TAUS_MS` — the two short rungs below ~5 ms. Default `[2.0, 3.5]`. If you want even *log*-spacing
  around 5 ms instead, use e.g. `[2.0, 3.5]` (short arm) and rely on the existing 15/35 for the long arm.
- `WHICH_WEIGHTS` — which weights the scale `c` multiplies (input i→h / recurrent h→h / readout h→out); the
  validated scheme is input+recurrent.
- `RATEMATCH_JSON` — per-seed `c` (het trained-rate match). Generated by `compute_ratematch_constants.py`.""")
code(r"""
# ========================== 3. Config ==========================
SEEDS       = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]   # the 10 paired seeds
TASK_KEY    = "4class"
MEM_FAMILY  = "lognormal"

# ---- Part 1: short fixed-tau rungs (ms) --------------------------------------------------  [CONFIRM]
SHORT_FIXED_TAUS_MS = [2.0, 3.5]        # both BELOW the ~5 ms learned optimum (fills the short arm)

# ---- Part 2: scaled-init control (6 remaining nets) — per-seed RATE-MATCH to het's TRAINED rate ----
# Validated locally on seed 202 (c=6.87): scale input+recurrent so hom STARTS at het's trained firing rate.
# NB the local 202 result was a DISSOCIATION (more active neurons + higher rate, but WORSE accuracy, and the
# hot init is unstable). This cell replicates it across seeds to test robustness, NOT expecting an improvement.
# 6 remaining scaled-init seeds to train HERE. Already trained elsewhere: seed 202 (local, c=6.87) and seeds
# 210, 340, 440 (Small Test notebook). Seed 101 is now learned and included. Full set = these 6 + those 4 = n=10.
SCALED_SEEDS  = [101, 550, 660, 710, 820, 930]
# each scaled hom is paired to its OWN seed's het via build_seeded_pair (identical init, only tau/scale differ)
WHICH_WEIGHTS = {"input_w": True, "recurrent_v": True, "output_w": False}   # input+recurrent (the validated scheme)
RATEMATCH_JSON = SCALING_DIR / "ratematch_constants_inputrec.json"          # per-seed c from the local sweep
MANUAL_SCALE   = 6.87                    # fallback if the json is absent (the seed-202 value)

# ---- reproduce the seeded_run_1_64n recipe exactly (patch BASE_PRMS before building any model) ----
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
src.BASE_PRMS["nb_recurrent"]        = 64
src.BASE_PRMS["batch_size"]          = 256
src.BASE_PRMS["nb_epochs"]           = 50
src.BASE_PRMS["early_stop_patience"] = 10
src.BASE_PRMS["optimizer"]           = "adam"
src.BASE_PRMS["weight_decay"]        = 1e-4     # homogeneous nets use WD=1e-4 (as in the original hom)
src.BASE_PRMS["lr"]                  = 2e-3
src.BASE_PRMS["lr_ab"]               = 2e-3
src.BASE_PRMS["lr_patience"]         = 5
src.BASE_PRMS["lr_factor"]           = 0.5
src.BASE_PRMS["lr_min"]              = 5e-5
src.BASE_PRMS["compile_model"]       = False

# hidden-layer weight keys in the state_dict
W_HID = "network.0.w"   # input -> hidden   (700 x 64)
V_HID = "network.0.v"   # hidden -> hidden  (64 x 64)
W_OUT = "network.1.w"   # hidden -> readout (64 x 4)

# checkpoint filename templates
def tau_tag(t):                                   # 2.0 -> "2", 3.5 -> "3p5"
    return f"{t:g}".replace(".", "p")
OUT_FIXED  = "4class_seed{s}_hom_fixed{tag}ms_lognormal.pt"   # Part 1 (this notebook)
OUT_SCALED = "4class_seed{s}_hom_scaledinit_inputrec_lognormal.pt"   # Part 2 (this notebook; c in metadata)
HET_NAME   = "4class_seed{s}_het_lognormal.pt"                # bundled reference (het)
HOM_NAME   = "4class_seed{s}_hom_geommean_lognormal.pt"       # bundled reference (hom, learned ~5 ms)

print(f"Part 1 (runs first): {len(SCALED_SEEDS)} scaled-init hom nets  |  per-seed rate-match  which={[k for k,v in WHICH_WEIGHTS.items() if v]}")
print(f"Part 2 (runs second): {len(SHORT_FIXED_TAUS_MS)} x {len(SEEDS)} = {len(SHORT_FIXED_TAUS_MS)*len(SEEDS)} short fixed-tau nets  taus={SHORT_FIXED_TAUS_MS} ms")
print(f"output -> {CKPT_OUT}")
""")

# ───────────────────────────────────────────────────────────────── 4 helpers
md(r"""## 4 — Helpers: build a frozen-tau hom, scale weights, forward → spikes, activity, numpy-PCA n90

`build_fixed_hom` is identical to the original Fixed-Tau Sweep notebook. The forward/activity/PCA helpers are the
sklearn-free ones from the local scaling diagnostic (numpy `eigh`, no OpenMP).""")
code(r'''
# ========================== 4. Helpers ==========================
def build_fixed_hom(seed, tau_mem_ms):
    """Homogeneous net with hidden tau_mem FROZEN at tau_mem_ms; init weights = the het seed's (paired)."""
    het_init, _ = src.build_heterogeneous_model(TASK_KEY, seed)     # only its NON-tau weights are used
    prms = src._base_prms_for_task(TASK_KEY, seed)
    prms.update({"het_ab": 0, "train_ab": 0, "train_hom_ab": 0})    # single, NON-trainable tau -> frozen
    prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
    src.set_seed(seed)
    model = src.RSNN(prms, rec=True).to(DEVICE)
    src.sync_non_tau_parameters(het_init, model)                   # identical init weights to the original hom
    src.set_layer_taus_ms(model.network[0], tau_syn_ms=src.DEFAULT_HIDDEN_TAU_SYN_MS,
                          tau_mem_ms=float(tau_mem_ms), time_step=prms["time_step"])
    src.set_layer_taus_ms(model.network[1], tau_syn_ms=src.DEFAULT_OUTPUT_TAU_SYN_MS,
                          tau_mem_ms=src.DEFAULT_OUTPUT_TAU_MEM_MS, time_step=prms["time_step"])
    n_trainable_tau = sum(p.requires_grad for n, p in model.named_parameters() if ("alpha" in n or "beta" in n))
    assert n_trainable_tau == 0, "tau params must be frozen for the fixed-tau control!"
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
def forward_hidden_spikes(model, prms, units, times, labels, n_batches=2):
    """Hidden binary spike tensor [n_hidden, n_samples] (index-2 = spikes), 1 ms resolution."""
    model.eval()
    nb_hidden = int(model.network[0].output_size)
    nb_steps  = int(prms["nb_steps"]); nb_inputs = int(prms["nb_inputs"])
    batch_size = int(prms["batch_size"]); inv_dt = 1.0 / float(prms["time_step"])
    sample_idx = np.where(np.isin(labels, prms["class_list"]))[0]
    n_batches = min(int(n_batches), -(-len(sample_idx) // batch_size))
    chunks = []
    for b in range(n_batches):
        bi = sample_idx[batch_size * b: min(len(sample_idx), batch_size * (b + 1))]
        abs_ = len(bi)
        t_arr = [np.round(times[i] * inv_dt).astype(np.int64) for i in bi]
        u_arr = [units[i] for i in bi]
        lens = np.array([len(a) for a in t_arr], dtype=np.int64)
        if lens.sum():
            ts = np.concatenate(t_arr); us = np.concatenate(u_arr)
            bc = np.repeat(np.arange(abs_, dtype=np.int64), lens)
            v = ts < nb_steps; ts, us, bc = ts[v], us[v], bc[v]
            idx = torch.from_numpy(np.stack([bc, ts, us]))
            x = torch.sparse_coo_tensor(idx, torch.ones(ts.size, dtype=torch.float32),
                                        torch.Size([abs_, nb_steps, nb_inputs])).to_dense()
        else:
            x = torch.zeros(abs_, nb_steps, nb_inputs)
        x = x.clamp(max=1.0).to(DEVICE)
        recs = model(0, 0, x)
        spk = recs[0][2].detach().cpu().numpy()                    # [trials, T, n_hidden], index 2 = spikes
        chunks.append(np.transpose(spk, (2, 0, 1)).reshape(nb_hidden, -1).astype(np.uint8))
    return np.concatenate(chunks, axis=1)

def activity_metrics(spk):
    per_neuron = spk.mean(axis=1)
    active = (spk.sum(axis=1) > 0)
    return {"pop_rate": float(per_neuron.mean()), "n_active": int(active.sum()),
            "active_frac": float(active.mean()), "n_hidden": int(spk.shape[0])}

def pca_n90(spk, stride=4):
    """numpy PCA n90 on [samples x neurons] (sklearn-free): #PCs for 90% variance."""
    X = spk[:, ::stride].T.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    if X.shape[0] < 2 or np.allclose(X, 0):
        return float("nan")
    ev = np.linalg.eigvalsh((X.T @ X) / (X.shape[0] - 1))[::-1].clip(min=0)
    tot = ev.sum()
    return int(np.searchsorted(np.cumsum(ev) / tot, 0.90) + 1) if tot > 0 else float("nan")

print("helpers ready.")
''')

# ───────────────────────────────────────────────────────────────── 5 data
md(r"""## 5 — Load SHD into RAM (once)""")
code(r"""
# ========================== 5. Data ==========================
train_cache, test_cache = src.load_default_caches()
UN, TI, LA = test_cache.units, test_cache.times, test_cache.labels
""")

# ───────────────────────────────────────────────────────────────── PART 1 (scaled-init, runs FIRST)
md(r"""# Part 1 — Scaled-init hom control (6 nets, per-seed rate-match)

Runs first. For each of the 6 remaining learned seeds (202 already trained locally; 210, 340, 440 from the Small
Test): rebuild the paired hom at init (correct geommean-tau init, non-tau weights synced from het), scale its
input+recurrent weights by the per-seed `c` that puts hom's init firing rate at het's *trained* rate, then
retrain with the exact original hom recipe (single *learned* tau, WD 1e-4, 50 epochs). `c` comes from
`init_scaling/ratematch_constants_inputrec.json`; if absent, falls back to `MANUAL_SCALE`.

> **Expectation, from the local seed-202 run (c=6.87):** this does not recover het's accuracy. It produced more
> active neurons and a higher rate but lower accuracy and the lowest participation ratio (a redundant,
> hyperactive code), with an unstable, wobbly training curve. We run these 6 to confirm the dissociation is
> robust, not a single-seed fluke. Watch each run for divergence.""")
code(r"""
# ========================== 6a. Resolve the per-seed rate-match scale c ==========================
ratematch = None
if RATEMATCH_JSON.exists():
    ratematch = json.loads(RATEMATCH_JSON.read_text()).get("per_seed", {})
    print("loaded rate-match constants:", RATEMATCH_JSON.name, "->", {k: round(v, 2) for k, v in ratematch.items()})
else:
    print(f"!! {RATEMATCH_JSON} NOT found -> using MANUAL_SCALE={MANUAL_SCALE} for every seed")
    print("   Generate locally: python code/expansions/init_scaling_controls/compute_ratematch_constants.py")

def resolve_scale(seed):
    if ratematch and str(seed) in ratematch and np.isfinite(ratematch[str(seed)]):
        return float(ratematch[str(seed)])
    return float(MANUAL_SCALE)

for s in SCALED_SEEDS:
    print(f"  seed {s}: c = {resolve_scale(s):.2f}")
""")
code(r"""
# ========================== 6b. Train scaled-init hom controls (6 seeds) ==========================
p2_results = []
for seed in SCALED_SEEDS:
    out_path = CKPT_OUT / OUT_SCALED.format(s=seed)
    if out_path.exists():
        ck = torch.load(out_path, map_location="cpu", weights_only=False)
        p2_results.append((seed, ck.get("init_weight_scale"), ck["history"]["test_acc"][-1]))
        print(f"scaled seed {seed}: exists, skip (c={ck.get('init_weight_scale')}, acc={ck['history']['test_acc'][-1]:.3f})")
        continue
    c = resolve_scale(seed)
    # paired init (correct geommean-tau init + het-synced non-tau weights), deterministic from the seed.
    # build_seeded_pair -> build_homogeneous_model gives a single LEARNED-tau recipe (train_hom_ab=1) for ALL
    # seeds, incl. 101 (whose bundled reference is the frozen anomaly -- we retrain it as a proper learned hom).
    pair = src.build_seeded_pair(master_seed=seed, task_key=TASK_KEY, mem_distribution_family=MEM_FAMILY)
    assert pair["metadata"]["linear_sync_verified"], f"linear sync failed for seed {seed}"
    hom = pair["hom_model"]
    hom_prms = dict(pair["hom_prms"]); hom_prms["weight_decay"] = 1e-4     # match the original hom recipe (WD 1e-4)
    hom_prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
    assert hom_prms.get("train_hom_ab") == 1 and hom_prms.get("het_ab") == 0, "recipe must be a single learned-tau hom!"
    # scale the hidden INIT weights, leave tau untouched
    hom.load_state_dict(scaled_state(hom.state_dict(), c))
    # sanity: tau still a single value (unchanged by the weight scaling)
    tau0 = src.decay_to_tau_ms(hom.network[0].beta.detach().cpu().numpy().ravel(), hom_prms["time_step"])
    assert np.unique(np.round(tau0, 6)).size == 1, "hom hidden tau should be a single shared value at init"
    print(f"\n=== scaled seed {seed}: c={c:.3f}  init tau={tau0[0]:.1f} ms  training (50 epochs) ===")
    t0 = time.time()
    history = src.train_experiment(hom, hom_prms, train_cache, test_cache)
    elapsed = time.time() - t0
    src.save_model_checkpoint(
        out_path, hom, history, elapsed, hom_prms,
        extra={"label": out_path.stem, "task": TASK_KEY, "task_key": TASK_KEY,
               "pair_seed": seed, "pair_role": "homogeneous_scaled_init",
               "run_label": "scaled_init", "mem_distribution_family": MEM_FAMILY,
               "control_type": "hom_scaled_init", "init_weight_scale": float(c),
               "scale_which_weights": WHICH_WEIGHTS, "scale_target_metric": "het_trained_mean_rate"})
    p2_results.append((seed, c, history["test_acc"][-1]))
    print(f"   saved -> {out_path.name} | final test_acc={history['test_acc'][-1]:.3f} ({elapsed/60:.1f} min)")
print("\nPart 1 (scaled-init) complete.")
""")

# ───────────────────────────────────────────────────────────────── PART 2 (short fixed-tau, runs SECOND)
md(r"""# Part 2 — Short fixed-tau controls

Runs second. For each seed, retrain a homogeneous net with hidden tau_mem frozen at each short value, paired to
its het seed (identical init weights, data, 50-epoch schedule; only the frozen tau differs). Resumable.""")
code(r"""
# ========================== 7. Train short fixed-tau controls ==========================
p1_results = []
for tau in SHORT_FIXED_TAUS_MS:
    tag = tau_tag(tau)
    for seed in SEEDS:
        out_path = CKPT_OUT / OUT_FIXED.format(s=seed, tag=tag)
        if out_path.exists():
            ck = torch.load(out_path, map_location="cpu", weights_only=False)
            p1_results.append((tau, seed, ck["history"]["test_acc"][-1]))
            print(f"tau={tau}ms seed {seed}: exists, skip (acc={ck['history']['test_acc'][-1]:.3f})")
            continue
        print(f"\n=== tau={tau}ms seed {seed}: training (50 epochs, frozen tau_mem={tau} ms) ===")
        model, prms = build_fixed_hom(seed, tau)
        t0 = time.time()
        history = src.train_experiment(model, prms, train_cache, test_cache)
        elapsed = time.time() - t0
        src.save_model_checkpoint(
            out_path, model, history, elapsed, prms,
            extra={"label": out_path.stem, "task": TASK_KEY, "task_key": TASK_KEY,
                   "pair_seed": seed, "pair_role": "homogeneous_fixed_tau",
                   "run_label": "short_fixed_tau", "mem_distribution_family": MEM_FAMILY,
                   "control_type": "hom_tau_frozen_fixed", "fixed_tau_mem_ms": float(tau)})
        p1_results.append((tau, seed, history["test_acc"][-1]))
        print(f"   saved -> {out_path.name} | final test_acc={history['test_acc'][-1]:.3f} ({elapsed/60:.1f} min)")
print("\nPart 2 (short fixed-tau) complete.")
""")

# ───────────────────────────────────────────────────────────────── PART 3
md(r"""# Part 3 — Verification (sklearn-free)

Accuracy + active-neuron count + numpy-PCA n90 for every condition present. The key read: does the **scaled-init**
hom (activity lifted toward het's) close the **n90** gap, or does het stay higher-dimensional at matched activity?
If n90 stays low despite matched activity, the dimensionality advantage is the tau **spread**, not recruitment.""")
code(r"""
# ========================== 8. Accuracy + activity + n90 across conditions ==========================
import matplotlib.pyplot as plt

def _short_paths():
    d = {}
    for tau in SHORT_FIXED_TAUS_MS:
        d[f"hom{tau:g}"] = (lambda s, tg=tau_tag(tau): CKPT_OUT / OUT_FIXED.format(s=s, tag=tg))
    return d

COND_PATH = {
    "het":       lambda s: REFCK / HET_NAME.format(s=s),
    "hom(~5)":   lambda s: REFCK / HOM_NAME.format(s=s),
    "hom_scaled": lambda s: CKPT_OUT / OUT_SCALED.format(s=s),
    **_short_paths(),
}
# order: het, learned hom, short fixed rungs (ascending), scaled-init
CONDS = ["het", "hom(~5)"] + [f"hom{t:g}" for t in sorted(SHORT_FIXED_TAUS_MS)] + ["hom_scaled"]
CONDS = [c for c in CONDS if any(COND_PATH[c](s).exists() for s in SEEDS)]

def load_net(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    prms = dict(ck["prms"]); prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
    model = src.RSNN(prms, rec=True).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    return model, prms, ck

rows = {c: {"acc": [], "n_active": [], "n90": []} for c in CONDS}
print(f"{'seed':>5} | " + " | ".join(f"{c:>10}" for c in CONDS))
for s in SEEDS:
    line = f"{s:>5} | "
    for c in CONDS:
        p = COND_PATH[c](s)
        if not p.exists():
            rows[c]["acc"].append(np.nan); rows[c]["n_active"].append(np.nan); rows[c]["n90"].append(np.nan)
            line += f"{'--':>10} | "; continue
        model, prms, ck = load_net(p)
        acc = float(ck["history"]["test_acc"][-1])
        spk = forward_hidden_spikes(model, prms, UN, TI, LA)
        m = activity_metrics(spk)
        rows[c]["acc"].append(acc); rows[c]["n_active"].append(m["n_active"]); rows[c]["n90"].append(pca_n90(spk))
        line += f"{acc:>10.3f} | "
        del model
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
    print(line + ("  <- seed101 hom is FROZEN ~25 ms (anomaly)" if s == 101 else ""))

print("\n   condition |   acc          | active        | n90")
for c in CONDS:
    a = np.array(rows[c]["acc"], float); na = np.array(rows[c]["n_active"], float); n9 = np.array(rows[c]["n90"], float)
    print(f"  {c:>10} | {np.nanmean(a):.3f}±{np.nanstd(a):.3f} | {np.nanmean(na):5.1f}±{np.nanstd(na):4.1f} | {np.nanmean(n9):5.1f}±{np.nanstd(n9):4.1f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, key, ylab in zip(axes, ["acc", "n_active", "n90"],
                         ["final test accuracy", "active neurons (of 64)", "PCA n90 (numpy)"]):
    means = [np.nanmean(rows[c][key]) for c in CONDS]; stds = [np.nanstd(rows[c][key]) for c in CONDS]
    ax.bar(range(len(CONDS)), means, yerr=stds, capsize=4, alpha=0.9)
    ax.set_xticks(range(len(CONDS))); ax.set_xticklabels(CONDS, rotation=30, ha="right")
    ax.set_ylabel(ylab); ax.set_title(ylab, fontweight="bold"); ax.grid(alpha=0.3, axis="y")
    if key == "acc": ax.axhline(0.25, color="gray", ls="--", lw=1)
plt.tight_layout(); plt.savefig(FIGDIR / "fig_short_tau_scaled_init_summary.png", dpi=150, bbox_inches="tight"); plt.show()
print("saved ->", FIGDIR / "fig_short_tau_scaled_init_summary.png")
""")

# ───────────────────────────────────────────────────────────────── done
md(r"""## Done — outputs & how to merge back

**Checkpoints (`outputs/checkpoints/`)**
- `4class_seed*_hom_scaledinit_inputrec_lognormal.pt` — the 6 scaled-init hom controls (Part 1; + seed 202 local
  and 210/340/440 from the Small Test = n=10), each storing
  `init_weight_scale` (the per-seed c), `scale_which_weights`, `scale_target_metric`.
- `4class_seed*_hom_fixed2ms_lognormal.pt`, `…_fixed3p5ms_…` — the 20 short fixed-tau controls (Part 2).
- `fig_short_tau_scaled_init_summary.png` — accuracy / active-neuron / n90 bars across conditions.

**Interpretation**
- **Part 1 (scaled init):** per the local seed-202 result, expect the scaled hom to show more active neurons and
  a higher rate but NOT higher accuracy (a redundant, hyperactive code), confirming activity is dissociated from
  performance. Check whether this holds across the 6 seeds and watch for unstable, wobbly training curves.
- **Part 2 (short rungs):** with the existing learned-hom (~5 ms) and long rungs (15/35 ms), you now have both
  arms of the single-tau accuracy curve. Expect accuracy to peak near ~5 ms and fall off toward 2 ms and toward
  35 ms. That is the direct "the best single tau is short, and it is a genuine interior optimum" figure.

**Merge back:** copy `outputs/checkpoints/*` into
`checkpoints/seeded_run_1_64n_controls/4class_lognormal/`, then re-run the corrected PCA / PC-reconstruction /
paired-stats passes (`code/representation/…`) over the extended condition set. Keep the paired design and use
`paired_stats.py`.

> **[CONFIRM] before the run:** short-tau values (`SHORT_FIXED_TAUS_MS`); which weights `c` touches
> (`WHICH_WEIGHTS`, validated = input+recurrent). Generate `ratematch_constants_inputrec.json` locally first
> (`python code/expansions/init_scaling_controls/compute_ratematch_constants.py`) and upload it, else Part 2
> uses `MANUAL_SCALE` (6.87) for every seed.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "Short-Tau + Scaled-Init Controls (Colab).ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, f"({len(cells)} cells)")

"""Builder: emits 'No-WD Hom Retrain (Colab).ipynb'.

Retrains the 9 homogeneous networks that were trained with L2 weight decay 1e-4 while their heterogeneous twins
were trained with none. Retraining them at weight_decay = 0.0 fixes two problems at once:

  1. THE PAIRED DESIGN. The design's claim is that within a seed the only free difference is the hidden membrane
     time constant. Read off the checkpoints, that is false for seeds 202..930: het has weight_decay 0.0, hom has
     1e-4. An L2 penalty on one arm of a paired comparison directly shrinks the recurrent weights, and the
     recurrent spectral radius is the metric on which the two networks differ most (2.46 vs 0.32). The confound
     points the same way as the effect we are trying to measure.

  2. FIDELITY TO THE REFERENCE. Perez-Nieves et al. did not regularise. In their code
     (references/neural_heterogeneity (Perez-Nieves 2021)/SuGD_code/): --weight_decay defaults to 0., their Adam
     is built without a weight_decay kwarg at all, and l2_weight_loss() in reg_loss.py is defined but never
     called. Our het arm (WD 0) matches them; the original hom arm (WD 1e-4) does not. Our own
     seeded_runs_common.build_homogeneous_model already sets weight_decay 0.0 and comments "no WD, matches
     paper", so the 1e-4 was introduced by the notebook that trained the original batch, against its own recipe.

Seeds 101 and 1001..1010 already have weight_decay 0.0 on both arms and are NOT retrained here.

The original WD=1e-4 homs are NOT overwritten. They stay as the matched baseline for the fixed-tau ladder and
the scaled-init controls, which all carry WD=1e-4 and whose primary comparisons are against the learned hom.
Keeping both gives two internally consistent families instead of one contaminated one.

Everything else reproduces the original recipe exactly: 64 neurons, 4-class, Adam, lr 2e-3, batch 256, 50 epochs,
early stop patience 10, single LEARNED tau (train_hom_ab=1) initialised at the geometric mean, and the non-tau
init weights synced from the seed's het twin so the pair still shares an initialisation.

Resumable: finished checkpoints are skipped.

Regenerate:  python code/training/_build_nowd_hom_retrain_nb.py
"""
import json
from pathlib import Path

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

# ───────────────────────────────────────────────────────────────── intro
md(r"""# Retrain the 9 homogeneous nets without weight decay (Colab)

Fixes a confound in the paired design. Requires a **GPU** runtime.

### What is wrong

Read straight off the checkpoint parameters:

| batch | seeds | het weight decay | hom weight decay | matched? |
|---|---|---|---|---|
| original | 202, 210, 340, 440, 550, 660, 710, 820, 930 | 0.0 | **1e-4** | **no** |
| retrained 101 + new | 101, 1001..1010 | 0.0 | 0.0 | yes |

The paired design exists to make the hidden membrane time constant the only free difference within a seed. For
nine of the twenty pairs it is not: the homogeneous net was penalised with L2 and its twin was not. That matters
because L2 shrinks the recurrent weights, and the recurrent spectral radius is exactly where the two networks
differ most (2.46 vs 0.32 in this batch). The confound pushes in the same direction as the effect being measured,
and it is a sufficient explanation for the homogeneous "collapse" to ~13 active units at gain ~0.3.

### Why weight decay 0 is the correct target, not 1e-4

**Perez-Nieves et al. did not regularise.** In their code: `--weight_decay` defaults to `0.`, their Adam is
constructed without a `weight_decay` kwarg at all, and `l2_weight_loss()` in `reg_loss.py` is defined but never
called from the loss. Our heterogeneous arm (WD 0) matches the reference; the original homogeneous arm (WD 1e-4)
departs from it. `seeded_runs_common.build_homogeneous_model` already sets `weight_decay = 0.0` with the comment
"no WD, matches paper", so the 1e-4 came from the notebook that trained the original batch, against its own
module's recipe. Retraining at 0 restores the paired design and fidelity to the reference at the same time.

### What this notebook does

Retrains **9 homogeneous nets** (seeds 202..930) at `weight_decay = 0.0`. Everything else is the original recipe:
64 neurons, 4-class, Adam, lr 2e-3, batch 256, 50 epochs, early stop patience 10, a single **learned** tau
initialised at the geometric mean, and non-tau init weights synced from the seed's het twin.

**The old WD=1e-4 homs are not overwritten.** They stay as the matched baseline for the fixed-tau ladder and the
scaled-init controls, which all carry WD=1e-4 and compare against the learned hom. Part 2 reads both.

### Cost

About 30 to 60 minutes per net on a T4, so roughly 5 to 9 hours for all nine. **The notebook is resumable**:
finished checkpoints are skipped, so if the session drops, just run it again. You can also cut `SEEDS` down and
do them in batches.

---
**Before running:** (1) put this folder on Google Drive; (2) Runtime → Change runtime type → **GPU**;
(3) check `RESEARCH_ROOT` in the paths cell; (4) Run all.
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
import numpy as np, torch, json, time

if IN_COLAB:
    # ---- COLAB: everything under the uploaded folder on Drive ----
    # >>> EDIT RESEARCH_ROOT only if your Drive folder name differs <<<
    RESEARCH_ROOT = Path("/content/drive/MyDrive/research project (SNN Info Theory)")
    PROJECT_ROOT  = RESEARCH_ROOT / "No-WD Hom Retrain (Colab Upload)"
    SHD_DIR       = RESEARCH_ROOT / "data" / "shd"
    CODE, PAPER   = PROJECT_ROOT / "code", PROJECT_ROOT / "code" / "SuGD_code"
    REFCK         = PROJECT_ROOT / "reference_checkpoints"
    CKPT_OUT      = PROJECT_ROOT / "outputs" / "checkpoints"
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
    REFCK        = REPO / "checkpoints" / "all_networks_4class_64n"      # het + original WD homs live here
    CKPT_OUT     = REPO / "checkpoints" / "seeded_run_1_64n_nowd" / "4class_lognormal"
    TAU_ARTIFACT = REPO / "code" / "training" / "Tau from 15 epoch run.json"

CKPT_OUT.mkdir(parents=True, exist_ok=True)

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

The original `seeded_run_1_64n` recipe with **one deliberate change**: `weight_decay = 0.0` instead of `1e-4`,
which is what the het twin used and what Perez-Nieves used.""")
code(r"""
# ========================== 3. Config ==========================
# The 9 pairs whose hom carried WD=1e-4 while het carried 0.0. Seeds 101 and 1001..1010 are already matched
# at WD=0 on both arms and are deliberately NOT retrained.
SEEDS      = [202, 210, 340, 440, 550, 660, 710, 820, 930]
TASK_KEY   = "4class"
MEM_FAMILY = "lognormal"

# ---- the original recipe, with weight_decay corrected to 0.0 ----
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
src.BASE_PRMS["nb_recurrent"]        = 64
src.BASE_PRMS["batch_size"]          = 256
src.BASE_PRMS["nb_epochs"]           = 50
src.BASE_PRMS["early_stop_patience"] = 10
src.BASE_PRMS["optimizer"]           = "adam"
src.BASE_PRMS["weight_decay"]        = 0.0      # <<< THE FIX. was 1e-4 for these homs; het used 0.0
src.BASE_PRMS["lr"]                  = 2e-3
src.BASE_PRMS["lr_ab"]               = 2e-3
src.BASE_PRMS["lr_patience"]         = 5
src.BASE_PRMS["lr_factor"]           = 0.5
src.BASE_PRMS["lr_min"]              = 5e-5
src.BASE_PRMS["compile_model"]       = False

OUT_NOWD = "4class_seed{s}_hom_geommean_nowd_lognormal.pt"    # new nets (WD 0)
HET_NAME = "4class_seed{s}_het_lognormal.pt"                  # reference het (WD 0)
HOM_NAME = "4class_seed{s}_hom_geommean_lognormal.pt"         # reference hom (WD 1e-4), kept, not overwritten

print(f"retraining {len(SEEDS)} homogeneous nets at weight_decay=0.0: {SEEDS}")
print(f"output -> {CKPT_OUT}")
""")

# ───────────────────────────────────────────────────────────────── 4 audit
md(r"""## 4 — Confirm the confound before fixing it

Reads the bundled reference checkpoints and prints the weight decay actually stored in each. Do not take the fix
on trust: this cell is what establishes that there is something to fix.""")
code(r'''
# ========================== 4. Audit the reference checkpoints ==========================
def wd_of(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return float(dict(ck["prms"])["weight_decay"]), len(ck.get("history", {}).get("test_acc", []))

print(f"{'seed':>6} | {'het WD':>8} {'hom WD':>8} | {'het ep':>6} {'hom ep':>6} | matched?")
n_bad = 0
for s in SEEDS:
    hw, he = wd_of(REFCK / HET_NAME.format(s=s))
    ow, oe = wd_of(REFCK / HOM_NAME.format(s=s))
    ok = (hw == ow)
    n_bad += (not ok)
    print(f"{s:>6} | {hw:>8} {ow:>8} | {he:>6} {oe:>6} | {'yes' if ok else 'NO  <<<'}")
assert n_bad == len(SEEDS), (
    f"expected all {len(SEEDS)} seeds to be confounded, found {n_bad}. Check you are pointing at the right "
    f"reference checkpoints before retraining.")
print(f"\n{n_bad}/{len(SEEDS)} pairs confounded, as expected. Retraining the hom arm at WD=0 to match het.")
''')

# ───────────────────────────────────────────────────────────────── 5 data
md(r"""## 5 — Load SHD into RAM (once)""")
code(r"""
# ========================== 5. Data ==========================
train_cache, test_cache = src.load_default_caches()
print("SHD loaded.")
""")

# ───────────────────────────────────────────────────────────────── 6 train
md(r"""## 6 — Retrain the 9 homogeneous nets at weight decay 0

For each seed, `build_seeded_pair` rebuilds the pair deterministically: het and hom are constructed from the same
seed and hom's non-tau weights are copied from het, so the pair shares an initialisation exactly as the original
did. We then take the hom and train it with the untouched recipe, whose `weight_decay` is already 0.0 from
`build_homogeneous_model`. Nothing is overridden, which is the point: the original notebook's override is what
caused this.

The asserts are the contract. If any fires, stop rather than train a net you cannot use.""")
code(r'''
# ========================== 6. Train ==========================
results = []
for seed in SEEDS:
    out_path = CKPT_OUT / OUT_NOWD.format(s=seed)
    if out_path.exists():
        ck = torch.load(out_path, map_location="cpu", weights_only=False)
        results.append((seed, ck["history"]["test_acc"][-1]))
        print(f"seed {seed}: exists, skip (acc={ck['history']['test_acc'][-1]:.3f})")
        continue

    pair = src.build_seeded_pair(master_seed=seed, task_key=TASK_KEY, mem_distribution_family=MEM_FAMILY)
    assert pair["metadata"]["linear_sync_verified"], f"linear sync failed for seed {seed}"
    hom = pair["hom_model"]
    hom_prms = dict(pair["hom_prms"])
    hom_prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})

    # ---- the contract ----
    # (a) the single manipulated variable really is tau: one shared, LEARNED constant
    assert hom_prms.get("train_hom_ab") == 1 and hom_prms.get("het_ab") == 0 and hom_prms.get("train_ab") == 0, \
        f"recipe must be a single learned-tau hom, got {hom_prms.get('het_ab')=} {hom_prms.get('train_hom_ab')=}"
    # (b) THE FIX: no weight decay, and it matches the het twin's stored value
    het_wd = float(dict(torch.load(REFCK / HET_NAME.format(s=seed), map_location="cpu",
                                   weights_only=False)["prms"])["weight_decay"])
    assert hom_prms["weight_decay"] == 0.0, f"weight_decay must be 0.0, got {hom_prms['weight_decay']}"
    assert hom_prms["weight_decay"] == het_wd, f"hom WD {hom_prms['weight_decay']} != het WD {het_wd}"
    # (c) tau starts as a single value at the geometric mean of the seed's lognormal sample
    tau0 = src.decay_to_tau_ms(hom.network[0].beta.detach().cpu().numpy().ravel(), hom_prms["time_step"])
    assert np.unique(np.round(tau0, 6)).size == 1, "hom hidden tau must be a single shared value at init"
    assert np.isclose(tau0[0], pair["metadata"]["hidden_tau_mem_geom_mean_ms"], atol=1e-4)

    print(f"\n=== seed {seed}: WD={hom_prms['weight_decay']} (het twin: {het_wd})  "
          f"init tau={tau0[0]:.1f} ms  training (up to 50 epochs) ===", flush=True)
    t0 = time.time()
    history = src.train_experiment(hom, hom_prms, train_cache, test_cache)
    elapsed = time.time() - t0
    src.save_model_checkpoint(
        out_path, hom, history, elapsed, hom_prms,
        extra={"label": out_path.stem, "task": TASK_KEY, "task_key": TASK_KEY,
               "pair_seed": seed, "pair_role": "homogeneous_nowd",
               "run_label": "nowd_retrain", "mem_distribution_family": MEM_FAMILY,
               "control_type": "hom_no_weight_decay",
               "supersedes": HOM_NAME.format(s=seed),
               "reason": "original hom trained with weight_decay=1e-4 while its het twin used 0.0; "
                         "Perez-Nieves used no weight decay. Retrained at 0.0 to restore the paired design.",
               "hidden_tau_mem_geom_mean_ms": pair["metadata"]["hidden_tau_mem_geom_mean_ms"]})
    results.append((seed, history["test_acc"][-1]))
    print(f"   saved -> {out_path.name} | final test_acc={history['test_acc'][-1]:.3f} "
          f"({elapsed/60:.1f} min, {len(history['test_acc'])} epochs)", flush=True)

print("\nRetraining complete.")
for s, a in results:
    print(f"  seed {s}: {a:.3f}")
''')

# ───────────────────────────────────────────────────────────────── 7 verify
md(r"""## 7 — Verification: did the weight decay cause the collapse?

The question this notebook exists to answer. For each seed, compare three networks on accuracy, recruitment,
firing rate, recurrent gain and numpy-PCA n90:

- **het** (WD 0), the reference
- **hom WD** (WD 1e-4), the original, confounded
- **hom noWD** (WD 0), the retrain

Two hypotheses, and the table decides between them:

- If the collapse was the weight decay, **hom noWD should recruit far more than hom WD and reach a much higher
  gain**, moving toward het. Whether it also closes the *accuracy* gap is the real result: if it recruits like
  het and still loses, that is a clean, powerful demonstration that recruitment is not what buys the advantage,
  and it does not depend on the scaled-init control at all.
- If hom noWD still collapses, the weight decay was incidental and the collapse is a property of a single
  learned tau. That would be worth knowing too.

sklearn-free (numpy `eigh`, no OpenMP clash with torch).""")
code(r'''
# ========================== 7. Verify ==========================
@torch.no_grad()
def forward_spikes(model, prms, cache, n_batches=6):
    """[n_hidden, samples] hidden spikes, plus per-trial accuracy inputs."""
    model.eval()
    U, T, L = cache.units, cache.times, cache.labels
    nh = int(model.network[0].output_size)
    ns, ni = int(prms["nb_steps"]), int(prms["nb_inputs"])
    bs = int(prms["batch_size"]); inv = 1.0 / float(prms["time_step"])
    tlm = prms.get("task_label_map") or {i: i for i in range(20)}
    idx = np.where(np.isin(L, prms["class_list"]))[0]
    nb = min(n_batches, -(-len(idx) // bs))
    chunks, correct, total = [], 0, 0
    for b in range(nb):
        bi = idx[bs * b: min(len(idx), bs * (b + 1))]; ab = len(bi)
        ta = [np.round(T[i] * inv).astype(np.int64) for i in bi]; ua = [U[i] for i in bi]
        ln = np.array([len(a) for a in ta])
        ts = np.concatenate(ta); us = np.concatenate(ua); bc = np.repeat(np.arange(ab), ln); m = ts < ns
        x = torch.sparse_coo_tensor(torch.from_numpy(np.stack([bc[m], ts[m], us[m]])),
                                    torch.ones(int(m.sum()), dtype=torch.float32),
                                    torch.Size([ab, ns, ni])).to_dense().clamp(max=1.0).to(DEVICE)
        recs = model(0, 0, x)
        spk = recs[0][2].detach().cpu().numpy()
        pred = recs[-1][1].max(1).values.argmax(1).cpu().numpy()
        y = np.array([tlm[int(L[i])] for i in bi])
        correct += int((pred == y).sum()); total += ab
        chunks.append(np.transpose(spk, (2, 0, 1)).reshape(nh, -1).astype(np.uint8))
    return np.concatenate(chunks, axis=1), correct / total

def pca_n90(spk, stride=4):
    X = spk[:, ::stride].T.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    if X.shape[0] < 2 or np.allclose(X, 0):
        return float("nan")
    ev = np.linalg.eigvalsh((X.T @ X) / (X.shape[0] - 1))[::-1].clip(min=0)
    tot = ev.sum()
    return int(np.searchsorted(np.cumsum(ev) / tot, 0.90) + 1) if tot > 0 else float("nan")

def load_net(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    prms = dict(ck["prms"]); prms.update({"dtype": torch.float, "device": DEVICE, "cuda": DEVICE.type == "cuda"})
    m = src.RSNN(prms, rec=True).to(DEVICE); m.load_state_dict(ck["model_state_dict"]); m.eval()
    gain = float(np.max(np.abs(np.linalg.eigvals(ck["model_state_dict"]["network.0.v"].cpu().numpy()))))
    tau = src.decay_to_tau_ms(ck["model_state_dict"]["network.0.beta"].cpu().numpy().ravel(), prms["time_step"])
    return m, prms, gain, float(np.median(tau))

rows = []
print(f"{'seed':>5} {'net':>10} | {'acc':>6} {'active':>7} {'rate':>7} {'gain':>6} {'n90':>4} {'tau':>7}")
for seed in SEEDS:
    trio = [("het", REFCK / HET_NAME.format(s=seed)),
            ("hom WD", REFCK / HOM_NAME.format(s=seed)),
            ("hom noWD", CKPT_OUT / OUT_NOWD.format(s=seed))]
    for lbl, path in trio:
        if not path.exists():
            print(f"{seed:>5} {lbl:>10} | missing"); continue
        m, prms, gain, tau = load_net(path)
        spk, acc = forward_spikes(m, prms, test_cache)
        active = int((spk.sum(1) > 0).sum()); rate = float(spk.mean())
        n90 = pca_n90(spk)
        rows.append(dict(seed=seed, net=lbl, acc=acc, active=active, rate=rate, gain=gain, n90=n90, tau=tau))
        print(f"{seed:>5} {lbl:>10} | {acc:>6.3f} {active:>7} {rate:>7.4f} {gain:>6.2f} {n90:>4} {tau:>7.1f}")
        del m
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

# ---- summary: the comparison the notebook exists for ----
import statistics as st
def agg(net, k):
    v = [r[k] for r in rows if r["net"] == net]
    return st.mean(v) if v else float("nan")
print("\n" + "=" * 74)
print(f"{'':>10} {'acc':>8} {'active':>8} {'rate':>8} {'gain':>8} {'n90':>6} {'tau(ms)':>8}")
for net in ["het", "hom WD", "hom noWD"]:
    print(f"{net:>10} {agg(net,'acc'):>8.3f} {agg(net,'active'):>8.1f} {agg(net,'rate'):>8.4f} "
          f"{agg(net,'gain'):>8.2f} {agg(net,'n90'):>6.1f} {agg(net,'tau'):>8.1f}")
print("=" * 74)
print(f"\nRemoving weight decay moved the hom: active {agg('hom WD','active'):.1f} -> {agg('hom noWD','active'):.1f}, "
      f"gain {agg('hom WD','gain'):.2f} -> {agg('hom noWD','gain'):.2f}, "
      f"acc {agg('hom WD','acc'):.3f} -> {agg('hom noWD','acc'):.3f}   (het: active {agg('het','active'):.1f}, "
      f"gain {agg('het','gain'):.2f}, acc {agg('het','acc'):.3f})")
print("\nThe accuracy gap that remains against het, on a properly matched pair, is the real result.")
print("Copy the new checkpoints back into the repo and rerun code/analysis_suite/extract_all.py.")
json.dump(rows, open(CKPT_OUT.parent / "nowd_verification.json", "w"), indent=1)
print(f"\nwrote {CKPT_OUT.parent / 'nowd_verification.json'}")
''')

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = Path(__file__).resolve().parent / "No-WD Hom Retrain (Colab).ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}  ({len(cells)} cells)")

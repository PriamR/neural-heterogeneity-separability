"""Builder: emits 'notebooks/Initialization Weight-Scaling Diagnostic (Local).ipynb'.

The LOCAL, no-retraining half of the weight/initialisation control (supervisor worklist A1 + A3-local).
Everything here runs on the existing checkpoints + a fresh init on the user's GPU; the actual 10-hom
retraining-from-scaled-init is a SEPARATE Colab notebook that consumes the constants this notebook picks.

Three parts:
  Part A  Trained-net context — load the paired het/hom checkpoints, forward the SHD test set, report the
          recruitment gap (~56 vs ~19 active) and the dimensionality gap (n90 ~26 vs ~11); plus the trained
          weight-distribution comparison (A1: per-neuron in/out weight norms, recurrent spectral radius).
          This is the target the retraining has to beat.
  Part B  Init-time weight-scaling sweep (A3-local) — build each net at INITIALISATION, forward the test set,
          and sweep a scalar `c` on the hidden weights; map c -> mean firing rate / active fraction /
          init dimensionality for hom, with het as the reference. No training.
  Part C  Pick the constant(s) — interpolate the c that lifts hom's init activity to het's, plus a couple of
          absolute targets; save to data_cache/init_scaling/ for the Colab retraining notebook.

sklearn-free on purpose (Windows OpenMP segfault trap): all PCA is numpy SVD/eigh.

Regenerate:  python code/expansions/_build_init_scaling_diagnostic.py
"""
import json
from pathlib import Path

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

# ───────────────────────────────────────────────────────────────── intro
md(r"""# Initialisation Weight-Scaling Diagnostic (Local)

The **local, no-retraining** half of the weight/initialisation control (supervisor notes A1 + A3). It runs on
the existing checkpoints and a fresh init on your local GPU. The actual **10-hom retraining from scaled init**
is a *separate Colab notebook* that reads the constant(s) this notebook picks.

**The control it sets up.** A skeptic says het only looks higher-dimensional because more of its neurons are
active (~56 vs ~19). To refute that we will retrain hom from an init whose weights are scaled up so its
activity matches het's, and check the dimensionality gap survives. This notebook does everything *except* that
retraining:

- **Part A — trained-net context.** Confirm the recruitment gap (~56 vs ~19 active) and dimensionality gap
  (n90 ~26 vs ~11) on this machine, and compare the trained weight distributions (A1). This is the target.
- **Part B — init-time scaling sweep.** Build each net at initialisation, forward the test set, and sweep a
  scalar `c` on the hidden weights; map `c → mean firing rate / active fraction / init dimensionality` for
  hom, with het as reference.
- **Part C — pick the constant.** Interpolate the `c` that lifts hom's init activity to het's (plus a couple
  of absolute targets); save to `data_cache/init_scaling/` for the retraining notebook.

> **sklearn-free** by design (Windows `matplotlib+torch+sklearn` OpenMP segfault). All PCA here is numpy.
""")

# ───────────────────────────────────────────────────────────────── 1 setup
md(r"""## 1 — Paths & imports (local)

Auto-finds the repo root (looks for `code/training/seeded_runs_common.py`), points the training module at the
in-repo model code, and points `SHD_DIR` at your local SHD copy (not bundled in the repo).""")
code(r'''
# ========================== 1. Paths + imports ==========================
import sys, json, time, copy
from pathlib import Path
import numpy as np, torch
import matplotlib.pyplot as plt

# --- find the repo root (dir containing code/training/seeded_runs_common.py) ---
_here = Path.cwd()
REPO_ROOT = next((c for c in [_here, *_here.parents]
                  if (c / "code" / "training" / "seeded_runs_common.py").exists()), None)
if REPO_ROOT is None:  # fallback: edit if you launched jupyter elsewhere
    REPO_ROOT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\Paper Repo (Clean)")
assert (REPO_ROOT / "code").exists(), f"repo root not found (got {REPO_ROOT})"

# >>> EDIT if your SHD .h5 live elsewhere (not bundled in the repo) <<<
SHD_DIR = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\data\shd")

CODE   = REPO_ROOT / "code"
TRAIN  = CODE / "training"
PAPER  = CODE / "model_perez_nieves"
CKPTS  = REPO_ROOT / "checkpoints" / "seeded_run_1_64n" / "4class_lognormal"
FIGDIR = REPO_ROOT / "figures"; FIGDIR.mkdir(exist_ok=True)
OUTDIR = REPO_ROOT / "data_cache" / "init_scaling"; OUTDIR.mkdir(parents=True, exist_ok=True)

for p in (str(TRAIN), str(PAPER)):
    if p not in sys.path:
        sys.path.insert(0, p)
import seeded_runs_common as src

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src.DEVICE          = DEVICE
src.PROJECT_ROOT    = REPO_ROOT
src.PAPER_ROOT      = PAPER
src.CHECKPOINT_ROOT = CKPTS
# tau-fit artifact (bundled next to the training module); fall back to the main project copy
_tau = TRAIN / "Tau from 15 epoch run.json"
src.TAU_ARTIFACT_PATH = _tau if _tau.exists() else Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)\Tau from 15 epoch run.json")
src.SHD_TRAIN = SHD_DIR / "shd_train.h5"
src.SHD_TEST  = SHD_DIR / "shd_test.h5"
assert src.SHD_TEST.exists(), f"SHD test set not found at {src.SHD_TEST} — set SHD_DIR"
assert src.TAU_ARTIFACT_PATH.exists(), f"tau artifact not found at {src.TAU_ARTIFACT_PATH}"
print("repo :", REPO_ROOT)
print("device:", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")
''')

md(r"""## 2 — Configuration""")
code(r'''
# ========================== 2. Config ==========================
SEEDS      = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]   # the 10 paired seeds (retrain set = the homs)
TASK_KEY   = "4class"
MEM_FAMILY = "lognormal"

src.BASE_PRMS["nb_recurrent"] = 64      # match the trained nets when building fresh inits
src.BASE_PRMS["batch_size"]   = 256

# which weights the scalar c multiplies. Input (i->h) and recurrent (h->h) drive hidden activity; the readout
# (h->out) does not, so it is off by default. [confirm with supervisors which set they meant.]
WHICH_WEIGHTS = {"input_w": True, "recurrent_v": True, "output_w": False}
SCALE_GRID    = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]   # multipliers applied to hom's init weights
TARGET        = "match_het"    # pick c so hom's init pop. mean rate == het's; also records active-fraction match

N_BATCHES = 2      # SHD-test batches (x256 trials) per forward — 512 trials is plenty for rate/active/n90
STRIDE_DIM = 4     # 4 ms subsample for the init-dimensionality proxy (rate uses full 1 ms)
ACTIVE_EPS = 0.0   # a neuron is "active" if it emits > ACTIVE_EPS total spikes over the sample

W_HID = "network.0.w"   # input -> hidden   (700 x 64)
V_HID = "network.0.v"   # hidden -> hidden  (64 x 64)
W_OUT = "network.1.w"   # hidden -> readout (64 x 4)
print("scaling", [k for k,v in WHICH_WEIGHTS.items() if v], "| grid", SCALE_GRID, "| target", TARGET)
''')

md(r"""## 3 — Helpers: forward → spikes, activity metrics, numpy PCA, weight scaling""")
code(r'''
# ========================== 3. Helpers ==========================
@torch.no_grad()
def forward_hidden_spikes(model, prms, units, times, labels, n_batches=N_BATCHES):
    """Return the hidden binary spike tensor [n_hidden, n_samples] (index-2 = spikes), 1 ms resolution."""
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
        spk = recs[0][2].detach().cpu().numpy()               # [trials, T, n_hidden], index 2 = spikes
        chunks.append(np.transpose(spk, (2, 0, 1)).reshape(nb_hidden, -1).astype(np.uint8))
    return np.concatenate(chunks, axis=1)                     # [n_hidden, trials*T]

def activity_metrics(spk):
    """mean spikes/step per neuron, population mean rate, active-neuron count/fraction."""
    per_neuron = spk.mean(axis=1)                             # spikes per 1 ms step, per neuron
    active = (spk.sum(axis=1) > ACTIVE_EPS)
    return {"rate_per_neuron": per_neuron, "pop_rate": float(per_neuron.mean()),
            "n_active": int(active.sum()), "active_frac": float(active.mean()),
            "n_hidden": int(spk.shape[0])}

def pca_n90(spk, stride=STRIDE_DIM):
    """numpy PCA n90 on [samples x neurons] activity (sklearn-free). Init-time proxy for dimensionality."""
    X = spk[:, ::stride].T.astype(np.float64)                 # [samples, neurons]
    X = X - X.mean(axis=0, keepdims=True)
    if X.shape[0] < 2 or np.allclose(X, 0):
        return float("nan")
    cov = (X.T @ X) / (X.shape[0] - 1)
    ev = np.linalg.eigvalsh(cov)[::-1].clip(min=0)
    tot = ev.sum()
    if tot <= 0:
        return float("nan")
    return int(np.searchsorted(np.cumsum(ev) / tot, 0.90) + 1)

def scaled_state(base_state, c, which=WHICH_WEIGHTS):
    s = {k: v.clone() for k, v in base_state.items()}
    if which.get("input_w"):     s[W_HID] = s[W_HID] * c
    if which.get("recurrent_v"): s[V_HID] = s[V_HID] * c
    if which.get("output_w"):    s[W_OUT] = s[W_OUT] * c
    return s

print("helpers ready.")
''')

md(r"""## 4 — Load the SHD test set once""")
code(r'''
# ========================== 4. Data ==========================
test_cache = src.SHDCache(src.SHD_TEST)
UN, TI, LA = test_cache.units, test_cache.times, test_cache.labels
print("test samples:", len(LA))
''')

# ───────────────────────────────────────────────────────────────── PART A
md(r"""# Part A — Trained-net context (the target to beat)

Forward the **trained** het/hom checkpoints and confirm, on this machine, the recruitment gap (~56 vs ~19
active) and dimensionality gap (n90 ~26 vs ~11). Also compare the trained weight distributions (A1). Seed 101
hom is the frozen anomaly — flagged, kept for context.""")
code(r'''
# ========================== 5. Trained-net activity + dimensionality ==========================
def load_trained(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    prms = dict(ck["prms"]); prms["dtype"] = torch.float
    prms["device"] = DEVICE; prms["cuda"] = DEVICE.type == "cuda"
    model = src.RSNN(prms, rec=True).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    return model, prms, ck

HET_F = "4class_seed{s}_het_lognormal.pt"
HOM_F = "4class_seed{s}_hom_geommean_lognormal.pt"

trained = {"het": {}, "hom": {}}
print(f"{'seed':>5} | {'het active':>10} {'het n90':>8} | {'hom active':>10} {'hom n90':>8}")
for s in SEEDS:
    for role, fn in (("het", HET_F), ("hom", HOM_F)):
        p = CKPTS / fn.format(s=s)
        if not p.exists():
            print(f"  seed {s} {role}: MISSING {p.name}"); continue
        model, prms, ck = load_trained(p)
        spk = forward_hidden_spikes(model, prms, UN, TI, LA)
        m = activity_metrics(spk); m["n90"] = pca_n90(spk)
        m["state"] = {k: v.detach().cpu().numpy() for k, v in ck["model_state_dict"].items()
                      if k in (W_HID, V_HID, W_OUT)}
        trained[role][s] = m
        del model
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
    he, ho = trained["het"].get(s), trained["hom"].get(s)
    if he and ho:
        print(f"{s:>5} | {he['n_active']:>10} {he['n90']:>8} | {ho['n_active']:>10} {ho['n90']:>8}"
              + ("   <- seed101 hom frozen" if s == 101 else ""))

def _mean(role, key):
    vals = [m[key] for m in trained[role].values() if np.isfinite(m[key])]
    return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))
for key, lab in (("n_active", "active neurons"), ("pop_rate", "pop mean rate"), ("n90", "PCA n90")):
    hm, hs = _mean("het", key); om, os_ = _mean("hom", key)
    print(f"  {lab:>14}:  het {hm:.3f}±{hs:.3f}   hom {om:.3f}±{os_:.3f}")
''')

md(r"""## 6 — Trained weight distributions (A1)

Per-neuron **input** norm ‖W_in[:,j]‖, **recurrent out/in** norms from V, **readout** norm, and the recurrent
**spectral radius** max|eig(V)| — het vs hom. Tests whether het's recruitment edge is legible in the weights
(e.g. hom leaving many neurons weakly wired / a smaller effective recurrent gain).""")
code(r'''
# ========================== 6. Weight-distribution comparison ==========================
def per_neuron_norms(state):
    Win = state[W_HID]; V = state[V_HID]; Wout = state[W_OUT]
    return {"in_norm": np.linalg.norm(Win, axis=0),          # per hidden neuron (columns)
            "rec_out": np.linalg.norm(V, axis=1),            # neuron j -> others
            "rec_in":  np.linalg.norm(V, axis=0),            # others -> neuron j
            "out_norm": np.linalg.norm(Wout, axis=1),        # neuron j -> readout
            "spec_radius": float(np.max(np.abs(np.linalg.eigvals(V))))}

agg = {"het": {"in_norm": [], "rec_out": [], "out_norm": [], "spec": []},
       "hom": {"in_norm": [], "rec_out": [], "out_norm": [], "spec": []}}
for role in ("het", "hom"):
    for s, m in trained[role].items():
        if "state" not in m: continue
        pn = per_neuron_norms(m["state"])
        agg[role]["in_norm"].append(pn["in_norm"]); agg[role]["rec_out"].append(pn["rec_out"])
        agg[role]["out_norm"].append(pn["out_norm"]); agg[role]["spec"].append(pn["spec_radius"])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, key, title in zip(axes, ["in_norm", "rec_out", "out_norm"],
                          ["input weight norm / neuron", "recurrent out-norm / neuron", "readout norm / neuron"]):
    for role, col in (("het", "#d1495b"), ("hom", "#30638e")):
        if agg[role][key]:
            vals = np.concatenate(agg[role][key])
            ax.hist(vals, bins=40, alpha=0.5, color=col, density=True, label=role)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("‖·‖₂"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIGDIR / "fig_init_weight_distributions.png", dpi=150, bbox_inches="tight"); plt.show()
for role in ("het", "hom"):
    sp = agg[role]["spec"]
    if sp: print(f"{role}: recurrent spectral radius {np.mean(sp):.3f} ± {np.std(sp):.3f}")
print("saved ->", FIGDIR / "fig_init_weight_distributions.png")
''')

# ───────────────────────────────────────────────────────────────── PART B
md(r"""# Part B — Init-time weight-scaling sweep (A3, local)

Build each net **at initialisation** (untrained, deterministic from the seed; het and hom share non-τ weights),
forward the test set, and sweep the scalar `c` on hom's hidden weights. Record, per `c`: population mean rate,
active fraction, and the init-dimensionality proxy. het at `c = 1` is the reference (its activity is the
target).""")
code(r'''
# ========================== 7. Init sweep ==========================
def build_init_pair(seed):
    """Untrained het/hom at init (paired). Returns cpu init state_dicts + prms."""
    pair = src.build_seeded_pair(master_seed=seed, task_key=TASK_KEY, mem_distribution_family=MEM_FAMILY)
    het, hom = pair["hetero_model"], pair["hom_model"]
    het_state = copy.deepcopy(het.state_dict()); hom_state = copy.deepcopy(hom.state_dict())
    return het, pair["hetero_prms"], het_state, hom, pair["hom_prms"], hom_state

sweep = {s: {} for s in SEEDS}       # sweep[seed] = {"het": metrics@1, "hom": {c: metrics}}
het_ref = {}
for s in SEEDS:
    het, hp, het_state, hom, op, hom_state = build_init_pair(s)
    # het reference at c=1 (its init activity is the target)
    het.load_state_dict(het_state)
    spk = forward_hidden_spikes(het, hp, UN, TI, LA)
    hm = activity_metrics(spk); hm["n90"] = pca_n90(spk); het_ref[s] = hm
    # hom across the scale grid
    hom_curve = {}
    for c in SCALE_GRID:
        hom.load_state_dict(scaled_state(hom_state, c))
        spk = forward_hidden_spikes(hom, op, UN, TI, LA)
        mm = activity_metrics(spk); mm["n90"] = pca_n90(spk)
        hom_curve[c] = mm
    sweep[s] = {"het": hm, "hom": hom_curve}
    del het, hom
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    print(f"seed {s}: het init pop_rate={hm['pop_rate']:.4f} active={hm['n_active']} | "
          f"hom c=1 pop_rate={hom_curve[1.0]['pop_rate']:.4f} active={hom_curve[1.0]['n_active']}")
print("sweep done.")
''')

md(r"""## 8 — Sweep figures (het reference + hom vs scale, ±1 std across seeds)""")
code(r'''
# ========================== 8. Sweep figures ==========================
def curve(metric):
    M = np.array([[sweep[s]["hom"][c][metric] for c in SCALE_GRID] for s in SEEDS], dtype=float)
    return np.nanmean(M, axis=0), np.nanstd(M, axis=0)
def het_line(metric):
    v = np.array([het_ref[s][metric] for s in SEEDS], dtype=float); return np.nanmean(v), np.nanstd(v)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, metric, ylab in zip(axes, ["pop_rate", "active_frac", "n90"],
                            ["population mean rate", "active fraction", "init PCA n90 (proxy)"]):
    mean, std = curve(metric); g = np.array(SCALE_GRID)
    ax.plot(g, mean, color="#30638e", lw=2, marker="o", label="hom (scaled)")
    ax.fill_between(g, mean - std, mean + std, color="#30638e", alpha=0.15)
    hm, hs = het_line(metric)
    ax.axhline(hm, color="#d1495b", lw=2, ls="--", label="het (init, c=1)")
    ax.axhspan(hm - hs, hm + hs, color="#d1495b", alpha=0.10)
    ax.axvline(1.0, color="gray", lw=1, ls=":")
    ax.set_xlabel("weight scale c"); ax.set_ylabel(ylab); ax.set_title(ylab, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig(FIGDIR / "fig_init_scaling_sweep.png", dpi=150, bbox_inches="tight"); plt.show()
print("saved ->", FIGDIR / "fig_init_scaling_sweep.png")
''')

# ───────────────────────────────────────────────────────────────── PART C
md(r"""# Part C — Pick the constant(s) and save

For each seed, interpolate the `c` that lifts hom's init **population rate** (and, separately, **active
fraction**) to het's init value. Report the median across seeds and a recommended scale set for the retraining
notebook: `{1.0 (control), c_match, a higher stress value}`.""")
code(r'''
# ========================== 9. Solve for c* and save ==========================
def interp_c(seed, metric):
    """smallest-grid interpolation of c where hom(metric) crosses het's value for this seed."""
    g = np.array(SCALE_GRID, dtype=float)
    y = np.array([sweep[seed]["hom"][c][metric] for c in SCALE_GRID], dtype=float)
    target = het_ref[seed][metric]
    if np.all(y < target): return float("inf")     # never reaches het even at max c
    if y[0] >= target:     return float(g[0])       # already at/above target at smallest c
    k = int(np.argmax(y >= target))                 # first index at/above target
    x0, x1, y0, y1 = g[k-1], g[k], y[k-1], y[k]
    return float(x0 + (target - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(g[k])

rows = []
for s in SEEDS:
    rows.append({"seed": s,
                 "c_match_rate":   interp_c(s, "pop_rate"),
                 "c_match_active": interp_c(s, "active_frac"),
                 "het_init_rate":  het_ref[s]["pop_rate"],
                 "het_init_active": het_ref[s]["active_frac"],
                 "hom_init_rate":  sweep[s]["hom"][1.0]["pop_rate"],
                 "hom_init_active": sweep[s]["hom"][1.0]["active_frac"]})
def _med(key):
    v = np.array([r[key] for r in rows], dtype=float); v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")
c_rate, c_active = _med("c_match_rate"), _med("c_match_active")
print(f"median c (match het rate)   = {c_rate:.3f}")
print(f"median c (match het active) = {c_active:.3f}")

c_hi = round(max(c_rate, c_active) * 1.5, 2)
recommended = sorted({1.0, round(c_rate, 2), c_hi})
payload = {
    "config": {"seeds": SEEDS, "task_key": TASK_KEY, "which_weights": WHICH_WEIGHTS,
               "scale_grid": SCALE_GRID, "target": TARGET, "active_eps": ACTIVE_EPS,
               "n_batches": N_BATCHES, "note": "constants apply to hom hidden weights at init, then retrain"},
    "per_seed": rows,
    "median_c_match_rate": c_rate, "median_c_match_active": c_active,
    "recommended_retrain_scales": recommended,
}
out_json = OUTDIR / "init_scaling_constants.json"
with open(out_json, "w") as fh: json.dump(payload, fh, indent=2)
print("\nrecommended retrain scales for the Colab notebook:", recommended)
print("saved ->", out_json)
''')

md(r"""## Done — what this produced & the handoff to Colab

**Outputs**
- `data_cache/init_scaling/init_scaling_constants.json` — the chosen `c` (per seed + median) and a
  `recommended_retrain_scales` list.
- `figures/fig_init_weight_distributions.png` — trained het vs hom weight norms + spectral radius (A1).
- `figures/fig_init_scaling_sweep.png` — hom activity/dimensionality vs scale, with het's init reference.

**Read the result like this.** If even at the top of the grid hom's init activity can't reach het's
(`c_match = inf`), that itself is informative. Otherwise `c_match` is the multiplier that equalises init
activity — the retraining notebook starts hom from `scaled_state(hom_init, c_match)` and trains normally.

**The separate Colab notebook** (the 10-hom retraining) should: read `init_scaling_constants.json`, rebuild
each hom at init, multiply the hidden weights by the recommended `c`, retrain 50 epochs (same recipe), then
re-measure active-neuron count / n90 / accuracy — and compare to the *unchanged* het reference. If the gap
survives at matched activity, the advantage is the τ spread, not the activity level.

> **[confirm with supervisors]** which weights `c` should touch (input / recurrent / readout), and whether to
> match het's mean **rate** or **active fraction**. Both constants are saved so either can be used.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO_ROOT = Path(__file__).resolve().parents[2] / "notebooks" / "Initialization Weight-Scaling Diagnostic (Local).ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, f"({len(cells)} cells)")

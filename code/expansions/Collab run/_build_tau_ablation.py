"""Builder: emits 'Tau Ablation (het).ipynb' (WS4 + expansion A2).

Ablate het hidden neurons ordered by membrane time constant tau_mem -- largest-first vs smallest-first vs
random -- and watch task accuracy drop. Tests Hardik's question: are the LONG time constants responsible for
the higher accuracy? Uses the cached 4 ms spike tensors + the network's own output membrane (the same faithful
readout replay validated in run_pc_reconstruction, m=64 -> ~0.66). No retraining.

Run once:  python _build_tau_ablation.py
"""
import json
from pathlib import Path

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""# τ-ordered ablation (heterogeneous nets) — WS4 + A2

**Hardik's ask:** on the het networks, ablate neurons **ordered by τ_mem — largest-first vs smallest-first**
— and watch accuracy drop. If largest-first collapses accuracy faster, the **long-τ neurons carry the task**.

**How we do it (no retraining).** For each het net we load the cached **4 ms binary spike tensor**
`[64 × 512,000] = 2,048 trials × 250 steps`, read the per-neuron τ_mem from the checkpoint, and replay the
network's **real output membrane** (hidden→output weights + leaky integration at the 4 ms step, decays
`α⁴, β⁴`, max-over-time → argmax — the network's own decision rule, validated in `run_pc_reconstruction`).
"Ablating" a neuron = zeroing its spike row before the readout. We sweep the number ablated for three
orderings — **largest-τ-first, smallest-τ-first, random** — and also record the **per-class** accuracy drop.

**Read the result as:** the *gap* between the largest-first and smallest-first curves. Steeper drop for
largest-first ⇒ long-τ neurons are disproportionately responsible for accuracy (and, per the per-class
panel, we expect the long-integration hard class *Ger-odd* to fall first).
""")

code(r'''
# ============================ CONFIG ============================
from pathlib import Path
import numpy as np, torch

DEFAULT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
ROOT = DEFAULT if DEFAULT.exists() else Path.cwd()
# Spike cache (4 ms) and checkpoints. If running from the Colab folder, these also resolve under it.
SPK = ROOT / "Project Files" / "Discrete Zscore Pipeline" / "outputs" / "spk_cache"
CK  = ROOT / "Project Files" / "Checkpoints" / "SeededRuns" / "seeded_run_1_64n" / "4class_lognormal"
if not CK.exists():                      # fallback: the reference checkpoints bundled for Colab
    alt = ROOT / "reference_checkpoints"
    if alt.exists(): CK = alt
FIGS = ROOT / "Project Files" / "Explanation files" / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
HERE = Path.cwd()

SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
NTR, TSTEP = 2048, 250                    # cache = 2048 trials x 250 (4 ms) steps
COL = {"largest": "#d1495b", "smallest": "#30638e", "random": "#7a7a7a"}
CLASS_NAMES = ["Eng-even", "Eng-odd", "Ger-even", "Ger-odd"]
tau_ms = lambda b: -1.0 / np.log(np.clip(b, 1e-9, 1 - 1e-9))
print("ROOT:", ROOT, "\nCK:", CK, "\nSPK:", SPK)
''')

code(r'''
# 4-class labels for the first 2,048 SHD-test trials (cache order = test order, no shuffle)
import sys
sys.path.insert(0, str(ROOT / "Project Files"))
sys.path.insert(0, str(ROOT / "neural_heterogeneity" / "SuGD_code"))
try:
    from data_gen import open_file
    _u, _t, _l = open_file(str(ROOT / "data" / "shd" / "shd_test.h5"))
    labels = np.array(_l[:]); _u._v_file.close()
except Exception as e:
    # fallback: SHD test labels must be provided; edit this path if needed
    raise RuntimeError(f"Could not open SHD test labels ({e}). Point at shd_test.h5.")
tlm = {i: (int(i >= 10) * 2 + int(i % 2)) for i in range(20)}   # 20 SHD digits -> language x parity
Y = np.array([tlm[int(v)] for v in labels[:NTR]])
print("labels:", np.bincount(Y))
''')

code(r'''
# ---- faithful readout replay (vectorised over trials) ----
def replay_logits(spk3, w, alpha, beta):
    """spk3: [NTR, TSTEP, 64] (0/1). w: [64,4]. Leaky output membrane at the 4 ms step -> max-over-time logits."""
    drive = spk3 @ w                                   # [NTR, TSTEP, 4]  hidden->output drive
    B, T, C = drive.shape
    syn = np.zeros((B, C), np.float32); mem = np.zeros((B, C), np.float32); best = np.full((B, C), -1e9, np.float32)
    for ti in range(T):
        syn = alpha * syn + drive[:, ti]
        mem = beta * mem + (1.0 - beta) * syn
        best = np.maximum(best, mem)
    return best                                        # [NTR, 4]

def load_net(seed):
    ck = torch.load(CK / f"4class_seed{seed}_het_lognormal.pt", map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    w  = sd["network.1.w"].numpy().astype(np.float32)                         # [64,4]
    a1 = float(sd["network.1.alpha"].numpy().ravel()[0]) ** 4                 # 1 ms -> 4 ms
    b1 = float(sd["network.1.beta"].numpy().ravel()[0]) ** 4
    beta_mem = sd["network.0.beta"].numpy().ravel()                            # [64]
    tau = tau_ms(beta_mem)                                                     # per-neuron tau (ms)
    spk = np.load(SPK / f"4class_seed{seed}_het_spk.npy").astype(np.float32)   # [64, 512000]
    spk3 = spk.T.reshape(NTR, TSTEP, 64)                                       # [2048,250,64] trial-major
    active = np.where(spk.sum(1) > 0)[0]                                       # ablate among active only
    return spk3, w, a1, b1, tau, active
''')

code(r'''
# ---- ablation sweep: largest-first / smallest-first / random, per seed ----
import json
def accuracy(spk3, w, a1, b1, mask):
    lg = replay_logits(spk3 * mask[None, None, :], w, a1, b1)
    return lg.argmax(1)

records = {}
GRID_FRAC = np.linspace(0, 1, 21)          # fraction of active neurons ablated
for seed in SEEDS:
    spk3, w, a1, b1, tau, active = load_net(seed)
    na = active.size
    counts = np.unique((GRID_FRAC * na).astype(int))
    order = {
        "largest":  active[np.argsort(-tau[active])],   # descending tau
        "smallest": active[np.argsort( tau[active])],   # ascending tau
    }
    seed_rec = {"n_active": int(na), "counts": counts.tolist(), "curves": {}, "perclass": {}}
    for name, idx in order.items():
        accs, pc = [], []
        for c in counts:
            mask = np.ones(64, np.float32); mask[idx[:c]] = 0.0
            pred = accuracy(spk3, w, a1, b1, mask)
            accs.append(float((pred == Y).mean()))
            pc.append([float((pred[Y == k] == k).mean()) for k in range(4)])
        seed_rec["curves"][name] = accs; seed_rec["perclass"][name] = pc
    # random ordering: average over 5 shuffles
    rng = np.random.default_rng(seed); rand_accs = []
    for _ in range(5):
        idx = rng.permutation(active); a = []
        for c in counts:
            mask = np.ones(64, np.float32); mask[idx[:c]] = 0.0
            a.append(float((accuracy(spk3, w, a1, b1, mask) == Y).mean()))
        rand_accs.append(a)
    seed_rec["curves"]["random"] = np.mean(rand_accs, 0).tolist()
    records[seed] = seed_rec
    print(f"seed {seed}: active={na}  acc@0={seed_rec['curves']['largest'][0]:.3f}  "
          f"largest@50%={seed_rec['curves']['largest'][len(counts)//2]:.3f}  "
          f"smallest@50%={seed_rec['curves']['smallest'][len(counts)//2]:.3f}", flush=True)
json.dump(records, open(HERE / "tau_ablation_records.json", "w"), indent=1)
print("SAVED tau_ablation_records.json")
''')

code(r'''
# ---- figure: accuracy vs fraction ablated (mean over seeds, aligned on fraction grid) ----
import matplotlib.pyplot as plt
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
xf = GRID_FRAC
for name in ["largest", "smallest", "random"]:
    curves = []
    for seed in SEEDS:
        r = records[seed]; fr = np.array(r["counts"]) / r["n_active"]
        curves.append(np.interp(xf, fr, r["curves"][name]))
    M = np.array(curves); m = M.mean(0); sd = M.std(0)
    ax.plot(xf, m, color=COL[name], lw=2.3, marker="o", ms=4, label=f"{name}-τ-first")
    ax.fill_between(xf, m - sd, m + sd, color=COL[name], alpha=0.15)
ax.axhline(0.25, color="k", ls=":", lw=1.1, label="chance")
ax.set_xlabel("fraction of active neurons ablated"); ax.set_ylabel("task accuracy")
ax.set_title("τ-ordered ablation (het): do the long-τ neurons carry accuracy?", fontweight="bold")
ax.grid(alpha=0.3); ax.legend()

# per-class drop under largest-first (mean over seeds)
for k in range(4):
    curves = []
    for seed in SEEDS:
        r = records[seed]; fr = np.array(r["counts"]) / r["n_active"]
        curves.append(np.interp(xf, fr, [row[k] for row in r["perclass"]["largest"]]))
    ax2.plot(xf, np.mean(curves, 0), lw=2, marker="o", ms=3, label=CLASS_NAMES[k])
ax2.axhline(0.25, color="k", ls=":", lw=1.1)
ax2.set_xlabel("fraction ablated (largest-τ-first)"); ax2.set_ylabel("per-class accuracy")
ax2.set_title("Which class falls first when the slow neurons go?", fontweight="bold")
ax2.grid(alpha=0.3); ax2.legend()
plt.tight_layout(); plt.savefig(FIGS / "fig_tau_ablation.png", dpi=150, bbox_inches="tight")
print("SAVED", FIGS / "fig_tau_ablation.png"); plt.show()
''')

md(r"""## Reading the output

- **Largest-first steeper than smallest-first** → the long-τ neurons are disproportionately responsible for
  accuracy (supports "long τ matters"). **Curves similar** → accuracy is spread across the τ range
  (supports "diversity, not just long τ"). The **random** curve is the neutral reference.
- **Per-class panel:** if *Ger-odd* (the hard, long-integration class) collapses first under largest-first,
  that localises the long-τ contribution to the classes that need sustained memory.
- Pair this with the fixed-τ ladder (WS1/WS5): ablation says *which neurons matter within het*; the ladder
  says *whether a single long τ in hom reproduces the effect*. Together they separate diversity from long-τ.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "Tau Ablation (het).ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("WROTE", out)

"""Builder: emits 'A3 — Causal Tau Shuffle, Transplant, Dose-Response.ipynb'.

Avenue A3 (the cleanest causal separation of DIVERSITY vs LONG-τ, no retraining):
  1. tau-shuffle (het): permute tau_mem across neurons, keeping the marginal distribution -> does the learned
     weight<->tau pairing matter?
  2. tau-transplant (hom <- het): give a trained hom net het's tau distribution at eval.
  3. dose-response: shrink het's tau spread toward its geometric mean by a factor gamma in [0,1] and plot
     accuracy vs spread -> the marginal value of diversity.
All are eval-only forward passes through the trained checkpoints (CPU-friendly).

Run once:  python _build_A3_tau_causal.py
"""
import json
from pathlib import Path

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""# A3 — Causal τ interventions: shuffle · transplant · dose–response

The fixed-τ ladder (WS1) needs retraining to separate *diversity* from *having long τ*. This notebook does
it **causally, eval-only, on the trained nets** — three interventions on τ_mem with the weights untouched:

1. **τ-shuffle (het):** permute τ_mem across neurons — the **marginal τ distribution is identical**, only
   the *assignment* of each τ to its trained weights is broken. Accuracy drop ⇒ the network **co-adapted**
   weights to specific timescales (diversity is *used*, not just present).
2. **τ-transplant (hom ← het):** give a trained hom net het's τ distribution at eval. How much of the gap is
   the **timescales alone** vs the weights learned around them?
3. **Dose–response:** continuously shrink het's τ spread toward its geometric mean
   (`log τ → mean_logτ + γ·(log τ − mean_logτ)`, γ: 1 = full het, 0 = collapsed to one τ) and plot accuracy
   vs the surviving spread — a smooth **marginal-value-of-diversity** curve, richer than the 3-point ladder.

Uses the repo forward pass (`seeded_runs_common` + `model.RSNN`) over the SHD test set. CPU is fine.
""")

code(r'''
import sys, importlib
from pathlib import Path
import numpy as np, torch

DEFAULT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
ROOT = DEFAULT if DEFAULT.exists() else Path.cwd()
PF = ROOT / "Project Files"; PAPER = ROOT / "neural_heterogeneity" / "SuGD_code"
CK = PF / "Checkpoints" / "SeededRuns" / "seeded_run_1_64n" / "4class_lognormal"
SHD = ROOT / "data" / "shd" / "shd_test.h5"
FIGS = ROOT / "Project Files" / "Explanation files" / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
HERE = Path.cwd()
for p in (str(PF), str(PAPER)):
    if p not in sys.path: sys.path.insert(0, p)
import seeded_runs_common as src
RSNN = importlib.import_module("model").RSNN
from data_gen import open_file
DEV = torch.device("cpu"); src.DEVICE = DEV
try: torch.set_num_threads(max(1, __import__("os").cpu_count() or 4))
except Exception: pass
SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
# CPU forward passes over the 1000-step recurrent net are slow; cap the number of test trials evaluated so
# the causal-DIRECTION read finishes quickly. Set MAX_TRIALS = None for the full test set (final paired stats).
MAX_TRIALS = 768
tau_ms  = lambda b: -1.0 / np.log(np.clip(b, 1e-9, 1 - 1e-9))
beta_of = lambda tau: np.exp(-1.0 / np.clip(tau, 1e-6, None)).astype(np.float32)

_u, _t, _l = open_file(str(SHD)); units = list(_u[:]); times = list(_t[:]); labels = np.array(_l[:]); _u._v_file.close()
''')

code(r'''
@torch.no_grad()
def eval_acc(seed, arch, beta_override=None):
    """Forward pass over the SHD test set; optionally overwrite network.0.beta (the membrane τ). Returns acc."""
    fn = f"4class_seed{seed}_{'het_lognormal' if arch=='het' else 'hom_geommean_lognormal'}.pt"
    ck = torch.load(CK / fn, map_location=DEV, weights_only=False)
    prms = dict(ck["prms"]); prms["dtype"] = torch.float; prms["device"] = DEV; prms["cuda"] = False
    model = RSNN(prms, rec=True).to(DEV); model.load_state_dict(ck["model_state_dict"]); model.eval()
    if beta_override is not None:
        # replace the membrane-β parameter (hom's is scalar (1,), het's is (1,64)); a (1,64) override
        # broadcasts correctly in the forward pass regardless of the original shape.
        newb = torch.tensor(np.asarray(beta_override, dtype=np.float32)).reshape(1, -1)
        mod = model
        for part in "network.0".split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        mod.beta = torch.nn.Parameter(newb, requires_grad=False)
    correct = tot = 0
    for x, y in src.shd_generator(units, times, labels, prms, shuffle=False, epoch=0, drop_last=False):
        rec = model(0, 0, x.to(DEV)); logits = rec[-1][1].max(1).values
        correct += int((logits.argmax(1).cpu() == y).sum()); tot += y.numel()
        if MAX_TRIALS is not None and tot >= MAX_TRIALS: break
    del model
    return correct / tot

def het_beta(seed):
    ck = torch.load(CK / f"4class_seed{seed}_het_lognormal.pt", map_location=DEV, weights_only=False)
    return ck["model_state_dict"]["network.0.beta"].numpy().ravel().copy()
''')

code(r'''
# ---- Interventions 1 & 2: shuffle (het) and transplant (hom<-het) ----
import json
rng = np.random.default_rng(0)
rec = {"base_het": [], "shuffle_het": [], "base_hom": [], "transplant_hom": []}
for seed in SEEDS:
    b_het = het_beta(seed)
    rec["base_het"].append(eval_acc(seed, "het"))
    rec["shuffle_het"].append(np.mean([eval_acc(seed, "het", b_het[rng.permutation(b_het.size)]) for _ in range(2)]))
    rec["base_hom"].append(eval_acc(seed, "hom"))
    rec["transplant_hom"].append(eval_acc(seed, "hom", b_het.copy()))   # hom weights, het timescales
    print(f"seed {seed}: het {rec['base_het'][-1]:.3f} -> shuffle {rec['shuffle_het'][-1]:.3f} | "
          f"hom {rec['base_hom'][-1]:.3f} -> +het-τ {rec['transplant_hom'][-1]:.3f}", flush=True)
from scipy import stats
for a, b, name in [("base_het","shuffle_het","het: shuffle drop"), ("base_hom","transplant_hom","hom: +het-τ gain")]:
    x, y = np.array(rec[a]), np.array(rec[b]); t = stats.ttest_rel(y, x)
    print(f"{name}: Δ={np.mean(y-x):+.3f}  paired-t p={t.pvalue:.3f}")
json.dump(rec, open(HERE / "A3_shuffle_transplant.json", "w"), indent=1)
''')

code(r'''
# ---- Intervention 3: dose-response (shrink het τ spread toward its geomean) ----
GAMMAS = np.linspace(0.0, 1.0, 5)     # 0 = single τ (geomean), 1 = full het spread
dose = {float(g): [] for g in GAMMAS}
for seed in SEEDS:
    b_het = het_beta(seed); logtau = np.log(tau_ms(b_het)); mlt = logtau.mean()
    for g in GAMMAS:
        tau_g = np.exp(mlt + g * (logtau - mlt))
        dose[float(g)].append(eval_acc(seed, "het", beta_of(tau_g)))
    print(f"seed {seed}: γ0={dose[0.0][-1]:.3f} ... γ1={dose[1.0][-1]:.3f}", flush=True)
json.dump(dose, open(HERE / "A3_dose_response.json", "w"), indent=1)
''')

code(r'''
import matplotlib.pyplot as plt
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
# left: shuffle & transplant
labels_ = ["het\n(base)", "het\nτ-shuffled", "hom\n(base)", "hom\n+het-τ"]
means = [np.mean(rec[k]) for k in ["base_het","shuffle_het","base_hom","transplant_hom"]]
sds   = [np.std(rec[k])  for k in ["base_het","shuffle_het","base_hom","transplant_hom"]]
ax.bar(range(4), means, yerr=sds, color=["#d1495b","#e79aa6","#30638e","#8fb3cf"], capsize=4)
ax.set_xticks(range(4)); ax.set_xticklabels(labels_); ax.set_ylabel("test accuracy")
ax.axhline(0.25, color="k", ls=":", lw=1); ax.set_title("Causal τ interventions (eval-only)", fontweight="bold")
# right: dose-response
m = [np.mean(dose[float(g)]) for g in GAMMAS]; s = [np.std(dose[float(g)]) for g in GAMMAS]
ax2.errorbar(GAMMAS, m, yerr=s, color="#d1495b", lw=2.2, marker="o", capsize=3)
ax2.set_xlabel("τ-spread factor γ   (0 = single τ, 1 = full het)"); ax2.set_ylabel("test accuracy")
ax2.axhline(0.25, color="k", ls=":", lw=1); ax2.set_title("Dose–response: marginal value of diversity", fontweight="bold")
ax2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(FIGS / "fig_A3_tau_causal.png", dpi=150, bbox_inches="tight")
print("SAVED", FIGS / "fig_A3_tau_causal.png"); plt.show()
''')

md(r"""## Reading the output

- **het τ-shuffle drops accuracy** ⇒ the *assignment* of timescales to trained weights matters (diversity is
  co-adapted, not incidental). **No drop** ⇒ only the marginal τ distribution matters.
- **hom + het-τ recovers part of the gap** ⇒ some of het's advantage is the *timescales alone*; the residual
  is the weights learned around them. This bounds "long-τ vs diversity" without retraining.
- **Dose–response:** a monotone rise of accuracy with spread γ is direct evidence that *more diversity → more
  accuracy*; a plateau would say a little spread suffices. Complements Marcus's discrete 15/25/35 ms ladder
  with a continuous curve.
- **Caveat:** these interventions change τ without re-optimising weights, so they *lower-bound* what a
  retrained net achieves — read them as causal-direction tests, and confirm magnitudes against the ladder.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "A3 — Causal Tau Shuffle, Transplant, Dose-Response.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("WROTE", out)

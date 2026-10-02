"""Builder: emits 'A1 — Tau-Resolved Single-Neuron Function.ipynb'.

Avenue A1: localise the heterogeneity advantage to the unit. For every het neuron, relate its membrane
time constant tau_mem to (i) its mutual information with the class, (ii) its firing rate, and (iii) the task
axis it codes (language vs parity vs conjunction). Tests whether tau diversity produces a DIVISION OF
TEMPORAL LABOUR (slow neurons -> sustained/context features, fast -> onset/language). Existing data only.

Run once:  python _build_A1_tau_resolved.py
"""
import json
from pathlib import Path

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""# A1 — τ-resolved single-neuron function (het)

*Where does the advantage live?* This notebook relates each het neuron's **membrane time constant τ_mem**
to what it does for the task:
- **task MI** `I(spikeᵢ ; class)` — how much the neuron tells you about language×parity;
- **firing rate**;
- **coded axis** — is it more informative about **language** (Eng/Ger) or **parity** (even/odd)?

**Hypothesis (division of temporal labour):** slow neurons (large τ) carry sustained/context information —
especially **parity** and the hard *Ger-odd* class that needs long integration — while fast neurons (small
τ) carry onset/**language** detail. If so, the imposed τ diversity is *functionally* organised, which is the
mechanism behind het's memory + separability edge.

Inputs: `Representation Analysis/outputs/repr_seed*_het.npz` (rate code + labels) and each checkpoint's
`network.0.beta` (→ τ_mem). No forward pass, no retraining.
""")

code(r'''
from pathlib import Path
import numpy as np, torch
# NOTE: sklearn is deliberately NOT imported -- matplotlib + torch + sklearn together trip an OpenMP
# segfault on this Windows setup, so we use a self-contained numpy mutual-information estimator.

DEFAULT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
ROOT = DEFAULT if DEFAULT.exists() else Path.cwd()
REPR = ROOT / "Project Files" / "Representation Analysis" / "outputs"
CK   = ROOT / "Project Files" / "Checkpoints" / "SeededRuns" / "seeded_run_1_64n" / "4class_lognormal"
FIGS = ROOT / "Project Files" / "Explanation files" / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
HERE = Path.cwd()
SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
tau_ms = lambda b: -1.0 / np.log(np.clip(b, 1e-9, 1 - 1e-9))

def mi_bits(a, b):  # both integer-labelled; plug-in MI in bits
    a = np.unique(a, return_inverse=True)[1]; b = np.unique(b, return_inverse=True)[1]
    n = a.size; P = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(P, (a, b), 1.0); P /= n
    Pa = P.sum(1, keepdims=True); Pb = P.sum(0, keepdims=True); den = Pa * Pb
    m = P > 0
    return float((P[m] * np.log2(P[m] / den[m])).sum())
''')

code(r'''
# ---- per-neuron features for every het net ----
rows = []   # (seed, neuron, tau, rate, mi_class, mi_lang, mi_parity, axis)
for seed in SEEDS:
    f = REPR / f"repr_seed{seed}_het.npz"
    if not f.exists():
        print("MISSING", f, "-> run run_repr_extract.py first"); continue
    d = np.load(f); rate = d["rate"]; y = d["y"].astype(int)          # rate [n,64], y [n] in 0..3
    lang = (y >= 2).astype(int)                                        # 0 Eng, 1 Ger
    parity = (y % 2).astype(int)                                       # 0 even, 1 odd
    beta = torch.load(CK / f"4class_seed{seed}_het_lognormal.pt", map_location="cpu",
                      weights_only=False)["model_state_dict"]["network.0.beta"].numpy().ravel()
    tau = tau_ms(beta)
    active = np.where(rate.mean(0) > 0)[0]
    for i in active:
        xi = (rate[:, i] > np.median(rate[:, i])).astype(int)         # median-binarise this neuron
        if xi.min() == xi.max():                                      # degenerate (all same) -> skip
            continue
        mic = mi_bits(xi, y); mil = mi_bits(xi, lang); mip = mi_bits(xi, parity)
        axis = "language" if mil > mip else "parity"
        rows.append((seed, int(i), float(tau[i]), float(rate[:, i].mean()), mic, mil, mip, axis))
import numpy as np
tau  = np.array([r[2] for r in rows]); rate = np.array([r[3] for r in rows])
mic  = np.array([r[4] for r in rows]); mil = np.array([r[5] for r in rows]); mip = np.array([r[6] for r in rows])
axis = np.array([r[7] for r in rows])
print(f"{len(rows)} active het neurons across {len(SEEDS)} seeds")
''')

code(r'''
from scipy import stats
print("=== correlations (log τ vs feature), pooled over neurons ===")
lt = np.log(tau)
for nm, v in [("task MI", mic), ("firing rate", rate), ("language MI", mil), ("parity MI", mip)]:
    pr = stats.pearsonr(lt, v); sp = stats.spearmanr(lt, v)
    print(f"log τ vs {nm:>12}:  Pearson r={pr.statistic:+.3f} (p={pr.pvalue:.1e})   Spearman ρ={sp.statistic:+.3f}")
# does the coded axis differ by τ?  (slow -> parity?)
t_lang = tau[axis == "language"]; t_par = tau[axis == "parity"]
u = stats.mannwhitneyu(t_lang, t_par)
print(f"\nmedian τ  language-coding={np.median(t_lang):.1f} ms   parity-coding={np.median(t_par):.1f} ms   "
      f"(Mann-Whitney p={u.pvalue:.1e})")
''')

code(r'''
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
CMAP = {"language": "#e08a1e", "parity": "#3a7d44"}
c = [CMAP[a] for a in axis]
ax[0].scatter(tau, mic, s=14, c=c, alpha=0.5)
b, a0 = np.polyfit(np.log(tau), mic, 1); xx = np.linspace(tau.min(), tau.max(), 100)
ax[0].plot(xx, a0 + b*np.log(xx), "k--", lw=1.5)
ax[0].set_xscale("log"); ax[0].set_xlabel("τ_mem (ms)"); ax[0].set_ylabel("task MI (bits)")
ax[0].set_title("τ vs task information", fontweight="bold")
ax[1].scatter(tau, rate, s=14, c=c, alpha=0.5); ax[1].set_xscale("log")
ax[1].set_xlabel("τ_mem (ms)"); ax[1].set_ylabel("mean firing rate"); ax[1].set_title("τ vs rate", fontweight="bold")
# τ distribution split by coded axis
ax[2].hist([np.log10(t_lang), np.log10(t_par)], bins=18, color=[CMAP["language"], CMAP["parity"]],
           label=["language-coding", "parity-coding"])
ax[2].set_xlabel("log₁₀ τ_mem (ms)"); ax[2].set_ylabel("neurons")
ax[2].set_title("do slow neurons code parity?", fontweight="bold"); ax[2].legend()
from matplotlib.lines import Line2D
ax[0].legend(handles=[Line2D([0],[0], marker="o", ls="", color=CMAP[k], label=k) for k in CMAP])
plt.tight_layout(); plt.savefig(FIGS / "fig_A1_tau_resolved.png", dpi=150, bbox_inches="tight")
print("SAVED", FIGS / "fig_A1_tau_resolved.png"); plt.show()

import json
json.dump({"tau": tau.tolist(), "task_mi": mic.tolist(), "rate": rate.tolist(),
           "lang_mi": mil.tolist(), "parity_mi": mip.tolist(), "axis": axis.tolist()},
          open(HERE / "A1_tau_resolved_records.json", "w"))
print("SAVED A1_tau_resolved_records.json")
''')

md(r"""## Reading the output

- A **positive log τ ↔ task-MI** correlation would say slower neurons carry more task information; a **shift
  of parity-coding neurons toward larger τ** (Mann-Whitney) is the direct signature of the *division of
  temporal labour*.
- Cross-reference with the ablation notebook: the neurons this analysis flags as high-τ / high-MI /
  parity-coding should be the ones whose removal (largest-first) collapses accuracy — two independent views
  of the same mechanism.
- Re-run this on the **fixed-τ 15/25/35 ms hom controls** once trained: a single-τ population cannot show a
  τ↔function gradient, which is exactly the diversity signal.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "A1 — Tau-Resolved Single-Neuron Function.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("WROTE", out)

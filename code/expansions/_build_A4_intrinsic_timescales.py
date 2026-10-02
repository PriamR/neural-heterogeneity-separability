"""Builder: emits 'A4 — Intrinsic (Functional) Timescales.ipynb'.

Avenue A4: does the IMPOSED tau diversity survive recurrence + training? Measure each neuron's spike-train
autocorrelation decay (tau_intrinsic, Murray et al. 2014 method) from the cached 4 ms spikes and compare it
to its imposed tau_mem. Also compare the spread of tau_intrinsic in het vs hom -- does hom develop any
functional spread despite a single membrane tau? Existing data only.

Run once:  python _build_A4_intrinsic_timescales.py
"""
import json
from pathlib import Path

cells = []
def md(s):   cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""# A4 — Intrinsic (functional) timescales vs imposed τ_mem

*Impose ≠ express.* We set a spread of **membrane** time constants, but recurrence and training could
homogenise (or amplify) the **functional** timescales the neurons actually express. This notebook measures
each neuron's **intrinsic timescale** τ_intrinsic from its spike-train autocorrelation (Murray et al. 2014:
fit `AC(lag) ∝ exp(−lag/τ)`) using the cached 4 ms spikes, and asks:

1. Does τ_intrinsic **track** the imposed τ_mem in het (does the diversity survive)?
2. Is the **spread** of τ_intrinsic larger in het than hom (does hom stay single-timescale, or does the
   network manufacture some diversity anyway)?

Inputs: `Discrete Zscore Pipeline/outputs/spk_cache/4class_seed*_{het,hom}_spk.npy` `[64 × 512,000]`
and each checkpoint's `network.0.beta`. No forward pass.
""")

code(r'''
from pathlib import Path
import numpy as np, torch

DEFAULT = Path(r"C:\Users\Priya\Desktop\research project (SNN Info Theory)")
ROOT = DEFAULT if DEFAULT.exists() else Path.cwd()
SPK = ROOT / "Project Files" / "Discrete Zscore Pipeline" / "outputs" / "spk_cache"
CK  = ROOT / "Project Files" / "Checkpoints" / "SeededRuns" / "seeded_run_1_64n" / "4class_lognormal"
FIGS = ROOT / "Project Files" / "Explanation files" / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
HERE = Path.cwd()
SEEDS = [101, 202, 210, 340, 440, 550, 660, 710, 820, 930]
NTR, TSTEP, DT = 2048, 250, 4.0          # 250 steps of 4 ms
tau_ms = lambda b: -1.0 / np.log(np.clip(b, 1e-9, 1 - 1e-9))
CKN = {"het": "4class_seed{s}_het_lognormal.pt", "hom": "4class_seed{s}_hom_geommean_lognormal.pt"}
''')

code(r'''
def intrinsic_tau(spk3, max_lag=25):
    """spk3: [NTR, TSTEP, 64] binary. Per neuron: trial-averaged autocorrelation over lags 1..max_lag,
    fit log AC = -lag/tau (ms). Returns tau_intrinsic [64] (nan where undefined)."""
    NTR, T, N = spk3.shape
    x = spk3 - spk3.mean(1, keepdims=True)                 # remove per-trial mean
    taus = np.full(N, np.nan)
    for i in range(N):
        xi = x[:, :, i]                                    # [NTR, T]
        var = (xi * xi).mean()
        if var <= 1e-9: continue
        ac = np.array([ (xi[:, :T-L] * xi[:, L:]).mean() / var for L in range(1, max_lag + 1) ])
        m = ac > 1e-3                                       # positive, above floor
        if m.sum() < 3: continue
        lags = np.arange(1, max_lag + 1)[m] * DT           # ms
        b = np.polyfit(lags, np.log(ac[m]), 1)[0]          # slope = -1/tau
        if b < 0: taus[i] = -1.0 / b
    return taus

rows = {"het": [], "hom": []}   # (tau_mem, tau_intrinsic)
for seed in SEEDS:
    for arch in ("het", "hom"):
        f = SPK / f"4class_seed{seed}_{arch}_spk.npy"
        if not f.exists(): print("MISSING", f); continue
        spk3 = np.load(f).astype(np.float32).T.reshape(NTR, TSTEP, 64)
        beta = torch.load(CK / CKN[arch].format(s=seed), map_location="cpu",
                          weights_only=False)["model_state_dict"]["network.0.beta"].numpy().ravel()
        if beta.size == 1: beta = np.repeat(beta, 64)      # hom stores a single shared β -> broadcast
        tmem = tau_ms(beta); tint = intrinsic_tau(spk3)
        active = np.where(spk3.reshape(-1, 64).sum(0) > 0)[0]
        for i in active:
            if np.isfinite(tint[i]): rows[arch].append((float(tmem[i]), float(tint[i])))
    print(f"seed {seed} done", flush=True)
for a in rows: rows[a] = np.array(rows[a])
print("het pairs:", len(rows["het"]), " hom pairs:", len(rows["hom"]))
''')

code(r'''
from scipy import stats
het = rows["het"]; hom = rows["hom"]
pr = stats.pearsonr(np.log(het[:,0]), np.log(het[:,1]))
print(f"het: log τ_mem vs log τ_intrinsic  Pearson r={pr.statistic:+.3f} (p={pr.pvalue:.1e})")
print(f"τ_intrinsic spread (std of log10):  het={np.std(np.log10(het[:,1])):.3f}   hom={np.std(np.log10(hom[:,1])):.3f}")
print(f"τ_intrinsic median (ms):            het={np.median(het[:,1]):.1f}   hom={np.median(hom[:,1]):.1f}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 5.3))
ax[0].scatter(het[:,0], het[:,1], s=14, alpha=0.4, color="#d1495b")
lim = [min(het[:,0].min(), het[:,1].min()), max(het[:,0].max(), het[:,1].max())]
ax[0].plot(lim, lim, "k--", lw=1, label="y = x")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("imposed τ_mem (ms)"); ax[0].set_ylabel("intrinsic τ (ms)")
ax[0].set_title("het: does imposed diversity survive?", fontweight="bold"); ax[0].legend()
ax[1].hist([np.log10(het[:,1]), np.log10(hom[:,1])], bins=20, color=["#d1495b","#30638e"],
           label=["het", "hom"]); ax[1].set_xlabel("log₁₀ intrinsic τ (ms)"); ax[1].set_ylabel("neurons")
ax[1].set_title("functional-timescale spread: het vs hom", fontweight="bold"); ax[1].legend()
plt.tight_layout(); plt.savefig(FIGS / "fig_A4_intrinsic_timescales.png", dpi=150, bbox_inches="tight")
print("SAVED", FIGS / "fig_A4_intrinsic_timescales.png"); plt.show()

import json
json.dump({a: rows[a].tolist() for a in rows}, open(HERE / "A4_intrinsic_timescales.json", "w"))
print("SAVED A4_intrinsic_timescales.json")
''')

md(r"""## Reading the output

- A **positive log-log correlation** (τ_mem ↔ τ_intrinsic) in het confirms the imposed diversity is
  *expressed* in the dynamics — the manipulation does what we claim. A near-flat relationship would mean
  recurrence collapses the timescales, which would reframe the whole story.
- A **wider τ_intrinsic histogram for het** is the functional-diversity signal; if **hom also spreads**,
  the network manufactures some timescale diversity from recurrence alone despite one membrane τ — itself an
  interesting result to report.
- τ_intrinsic gives an *emergent* axis to correlate against function (feed it into A1 in place of τ_mem).
- **Caveat:** at 4 ms bins over 250 steps the fittable range is ~4–100 ms; very long τ saturate. Treat
  τ_intrinsic as a comparative index, not an absolute constant.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).parent / "A4 — Intrinsic (Functional) Timescales.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("WROTE", out)

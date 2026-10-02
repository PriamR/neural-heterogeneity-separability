# Expansion Analyses — First-Run Results (2026-07-03)

First execution of the mechanism-expansion notebooks in this folder, on the **existing** 10-seed 64-neuron
4-class caches/checkpoints (no new training). Purpose: sharpen *where and how* the heterogeneity advantage
arises, ahead of the fixed-τ ladder (WS1) training.

> **Status: provisional.** These run on the current het/hom pairs only. **We will re-run all of this once the
> 15 / 25 / 35 ms fixed-τ controls (and any additional seeds) are trained**, so the ladder can be read across
> the same measures and the diversity-vs-long-τ question answered cleanly. Numbers below are the current
> read, not final. A3 was run with `MAX_TRIALS=768` (capped eval for speed) — re-run with the full test set
> before quoting its accuracies as final.

Figures: `Project Files/Explanation files/figures/fig_{A1,A3,A4,B1,B3}_*.png`, `fig_tau_ablation.png`.
Records: `*.json` next to each notebook.

---

## Headline

The single strongest new result is **A3**: heterogeneity's advantage is not just *owning a good set of
timescales* — each neuron's **weights are co-adapted to its specific τ**. Break the pairing (shuffle) or
hand the timescales to another network (transplant) and accuracy collapses to near chance; dial the spread
up continuously and accuracy rises **monotonically**. The premise is sound (**A4**: imposed diversity is
genuinely expressed in the dynamics), the extra dimensions are **task-aligned** (**B3**), and the advantage
is **breadth, not synergy** (**B1**) — while a naïve one-neuron-one-role story does **not** hold (**A1**),
and the long-τ tail alone does **not** carry accuracy (**ablation**).

---

## A3 — Causal τ interventions (shuffle · transplant · dose–response)  ★ strongest result

**What it did.** Three edits to the trained nets, no retraining: (1) **shuffle** τ across neurons (same set
of τ's, broken assignment); (2) **transplant** het's τ distribution onto hom's weights; (3) **dose–response**
— shrink het's τ spread toward its geometric mean by a factor γ (1 = full het, 0 = single τ).

**Results (10 seeds; capped eval, `MAX_TRIALS=768`).**

| condition | accuracy | vs its baseline |
|---|---|---|
| het (base) | **0.685 ± 0.045** | — |
| het, τ-shuffled | **0.304 ± 0.032** | **Δ = −0.382, paired-t p < 0.001** |
| hom (base) | 0.563 ± 0.031 | — |
| hom + het-τ (transplant) | **0.322 ± 0.048** | **Δ = −0.241, paired-t p < 0.001** |

Dose–response (mean accuracy by surviving τ-spread γ):

| γ (spread) | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| accuracy | 0.322 | 0.364 | 0.455 | 0.591 | **0.685** |

**What it means.**
- **Shuffle collapses het to ~chance.** The *marginal* set of timescales is identical — only which neuron
  holds which τ changed — yet accuracy falls 0.685 → 0.304. So the network **co-adapts each neuron's
  weights to its own timescale**; diversity is *used*, not merely present.
- **Transplant doesn't help hom — it hurts.** Injecting het's timescales into hom's weights gives 0.322, not
  an improvement. You cannot get het's advantage by handing over the timescales without the matching weights.
- **Dose–response is monotone.** Accuracy climbs smoothly with the τ spread — direct evidence that *more
  diversity → more accuracy* (a continuous version of Marcus's 15/25/35 ms ladder). Collapsing a het net to a
  single τ (γ=0) lands at 0.322, far below het's 0.685 and even below hom's own 0.563 (because het's weights
  need the diversity they were trained with).

**Caveat.** These are eval-only interventions (weights not re-optimised), so they **lower-bound** what a
retrained network would do — read them as causal-*direction* tests, and confirm magnitudes against the
trained fixed-τ ladder.

---

## A4 — Intrinsic (functional) timescales — premise validated

**What it did.** Measured each neuron's *actual* timescale from its spike-train autocorrelation decay and
compared it to the *imposed* τ_mem.

**Results.** het log τ_mem ↔ log τ_intrinsic **Pearson r = +0.25, p = 3.4 × 10⁻⁹**; het functional-timescale
spread wider than hom (std log₁₀ **0.39 vs 0.27**), median **95 vs 67 ms**.

**What it means.** The imposed diversity **survives recurrence and training** and is expressed in the
dynamics — the manipulation does what we claim. This underwrites every other analysis.
*Caveat:* at 4 ms bins over 250 steps the fittable range is ~4–100 ms; treat τ_intrinsic as comparative.

---

## B3 — Task-subspace alignment — het's extra dimensions are task-aligned

**What it did.** Measured how many principal components each task axis (language, parity, XOR) needs, het vs
hom.

**Results.** Participation ratio **6.53 (het) vs 5.31 (hom)**. PCs to reach 80 % of the coding vector:

| axis | het | hom |
|---|---|---|
| language | 7 | 4 |
| parity | 3 | 3 |
| XOR | 5 | 3 |

**What it means.** het spreads the task — especially **language and the XOR conjunction** — across **more
dimensions**, i.e. its higher functional dimensionality is spent on **task-relevant** directions, not noise.
Consistent with the PC-reconstruction crossover and Wakhloo et al. (task-latent PCs ↔ neural PCs).

---

## B1 — Target-aware PID (redundant / unique / synergistic about the class)

**What it did.** Split the information neuron **pairs** carry about the class into redundant / unique /
synergistic atoms (Williams–Beer `I_min`, plug-in).

**Results (bits per pair, 10 seeds).**

| atom | het | hom | Δ (het−hom) | p |
|---|---|---|---|---|
| redundant | 0.0228 | 0.0271 | −0.0043 | 0.23 |
| unique (U₁+U₂) | 0.0650 | 0.0733 | −0.0083 | 0.32 |
| synergistic | 0.0319 | 0.0366 | −0.0048 | 0.08 |
| total I(pair;Y) | 0.1197 | 0.1371 | −0.0174 | 0.17 |

**What it means.** Per pair the atoms are **comparable (slightly lower for het)** — het is **not** winning
through synergy, consistent with the whole project. **Important caveat:** this is *per pair*, and het has
~2.8× more active neurons, so its **population-total** unique information is still larger. The breadth lives
in the neuron **count**, not per-pair richness — **add a population/summed PID read** to test the breadth
hypothesis directly (flagged for the re-run).

---

## A1 — τ-resolved single-neuron function — near-null (honest)

**What it did.** For every het neuron, related its τ to task-MI, firing rate, and coded axis
(language vs parity).

**Results (563 active het neurons).** log τ vs task-MI **r = −0.03 (n.s.)**; parity-coding neurons only
marginally slower than language-coding (median **27.6 vs 20.2 ms, p = 0.39**).

**What it means.** A tidy "slow = context/parity, fast = onset/language" split is **not** visible in the
per-trial mean-rate code. If a division of labour exists it is subtler than one-neuron-one-role.
**For the re-run:** repeat on the **τ_intrinsic** axis (A4) and on **per-time-bin** activity rather than the
trial-averaged rate — the trial mean may wash out the temporal signal this analysis is looking for.

---

## Ablation — τ-ordered neuron removal (het) — no clean long-τ effect

**What it did.** Removed het neurons largest-τ-first vs smallest-τ-first vs random, tracking accuracy (and
per-class) drop.

**Results.** **Seed-dependent** — largest-first drops faster in some seeds, smallest-first in others; on
average removing **small-τ** neurons hurts slightly more.

**What it means.** Accuracy is **spread across the whole τ range**, not concentrated in the long tail — a
nudge toward "diversity matters" over "just have long τ," but this is the noisiest measure and should be read
**alongside the trained fixed-τ ladder**, not on its own.

---

## What to redo once more networks are trained

1. **Run every measure across the 15 / 25 / 35 ms fixed-τ ladder + het** (not just het vs the ~25 ms hom) —
   this is what actually separates *diversity* from *long τ*. A3's dose–response and the ablation especially.
2. **A3 with the full test set** (`MAX_TRIALS = None`) for final paired accuracies; consider a **brief
   fine-tune** after each intervention to turn the eval-only lower bounds into trained estimates.
3. **B1 population-level PID** (sum unique info over all active neurons, or a k>2 read) to test the *breadth*
   hypothesis the per-pair version cannot see.
4. **A1 on τ_intrinsic and per-time-bin activity** rather than trial-mean rate.
5. **More seeds** where any effect is borderline (A1, B1, ablation) to firm up significance.
6. Re-generate all figures/records after re-runs; this doc is the provisional baseline to diff against.

*Notebooks + builders: this folder. Method rationale + paper links: `Mechanism Avenues.md`. Overall project
state: `Project Files/Explanation files/Project Context & Status — Updated (2026-07-03).md`.*

# New Quantification Avenues — *Where and How* the Heterogeneity Advantage Arises

**Purpose.** We have established *that* het wins (+13 pp, 10/10) and broadly *what* the advantage is
(geometric separability + long memory + functional dimensionality, **not** synergy). The next question the
supervisors want answered is **mechanistic**: *where in the network, and how, does the improvement come
from — and is it diversity or just longer τ?* The already-planned work (fixed-τ ladder WS1/WS5, min/max
O-Info WS2/WS3, τ-ordered ablation WS4) attacks the diversity-vs-long-τ confound. This doc proposes
**additional avenues of quantification** that localise and causally test the mechanism, each tagged with
**feasibility** and the **paper** it connects to.

Feasibility legend: **[E]** existing data/checkpoints only, no training · **[R⁺]** cheap re-eval or
readout-only re-fit (no full retrain) · **[T]** needs (re)training.

---

## Tier 1 — highest value, do first

### A1. τ-resolved single-neuron function — *localise the mechanism to the unit* **[E]**
For every het neuron, from the checkpoint τ_mem and the cached rate/spike code compute:
- per-neuron **task MI** `I(spike_i ; class)` and the **axis it codes** (language vs parity vs conjunction,
  from single-neuron logistic weights / selectivity index);
- **firing rate**, **decoder-weight magnitude**, and **per-neuron ablation importance** (§A2).

Then correlate each against τ_mem. **Hypothesis:** slow neurons carry sustained/context features (parity,
the hard Ger-odd rescue) and fast neurons carry onset/language — i.e. τ diversity produces a *division of
temporal labour*. This is the most direct "where" and connects the imposed τ spread to functional roles.
*Papers:* Murray 2014 (timescale hierarchy → function); Perez-Nieves 2021; Rigotti 2013 (mixed selectivity).

### A2. Ablation, sharpened — beyond largest/smallest-first (WS4) **[E/R⁺]**
Extend the planned τ-ordered ablation with three refinements that turn it from a single curve into a
mechanism map:
- **τ-band knockouts:** remove only the slow tail / only the fast head / only the mid-band, and read the
  **per-class** accuracy drop. Prediction: dropping slow neurons specifically collapses the hard,
  long-integration classes (Ger-odd) — pinning the memory advantage to the long-τ subpopulation.
- **Ordering comparison:** τ-order vs **greedy information order** (ablate by per-neuron task-MI) vs random.
  If τ-order ≈ info-order, τ *is* the functional axis.
- **Importance distribution (robustness signature):** is het's single-knockout importance **spread across
  many neurons** (graceful degradation) while hom's is **concentrated** in its ~20 active units? This makes
  Perez-Nieves' "robust" claim quantitative and ties it to the recruitment finding.
*Papers:* Perez-Nieves 2021 (robustness); the project's active-neuron result.

### A3. Causal τ-transplant / τ-shuffle — *diversity as a set vs diversity as learned assignment* **[R⁺]**
Three eval-only (or short-fine-tune) interventions on trained nets, no full retrain:
1. **τ-shuffle (het):** permute τ_mem across neurons, breaking the learned weight↔τ pairing while keeping
   the *marginal* τ distribution identical. Accuracy drop ⇒ the **assignment** of timescales to specific
   weighted neurons matters (co-adaptation), not just the τ histogram.
2. **τ-transplant (hom←het):** give a trained hom net het's τ distribution at eval. How much of the gap is
   the timescales *alone* vs the weights learned around them?
3. **Progressive homogenisation (dose–response):** shrink the het τ spread toward the geomean by a factor
   γ ∈ [0,1] and plot accuracy vs spread — a continuous curve is far more informative than the 3-point
   15/25/35 ms ladder and directly measures the *marginal value of diversity*.

This is the cleanest causal separation of "diversity" from "long τ" and complements WS1 without new training.
*Papers:* Perez-Nieves 2021; motivates/extends Marcus's fixed-τ logic.

### A4. Intrinsic (functional) timescales — *does imposed diversity survive recurrence + training?* **[E]**
Impose ≠ express. Measure each neuron's **spike-train autocorrelation decay** τ_intrinsic (Murray 2014
method) and compare to its imposed τ_mem. Two questions: (i) does the het population actually express a
**broad band of functional timescales**, or does recurrence homogenise them? (ii) does hom nonetheless
develop *some* timescale spread via network dynamics? This validates the entire premise of the manipulation
and gives an emergent-timescale axis to correlate with A1.
*Papers:* Murray 2014; [H] Sussillo–Barak (dynamics).

---

## Tier 2 — deepen the information-theoretic and geometric story

### B1. Target-aware PID (unique / redundant / synergistic about the class) **[E]**
Co-information gives one number; upgrade to a full 2-source **PID** (Williams–Beer, or the vendored
`gpid`/`dit`) on pairs/triplets → **unique vs redundant vs synergistic** information about the class. The
prediction from our story: het's edge lives in **unique** information carried by **more** neurons (breadth /
distributed coding), with synergy shared — the target-aware confirmation of the pairwise/dimensional story.
*Papers:* [S], [N]; Williams–Beer 2010; Schneidman 2003.

### B2. ΦID / temporal information decomposition — *formalise the memory advantage* **[E]**
The NuMIT MATLAB toolbox already vendored here includes **`NuMIT_PhiID.m`** (integrated information
decomposition). Apply ΦID to the het/hom time-lagged activity to split predictive information into
**storage / transfer / synergy** atoms. If het increases the **information-storage** atom, that gives a
formal IIT-style counterpart to the reservoir memory-capacity result (MC 14 vs 8.6) — connecting the
temporal fingerprint to a principled decomposition rather than a regression R².
*Papers:* [N] NuMIT/ΦID; Dambre 2012; Jaeger 2001.

### B3. Task-subspace alignment — *which dimensions carry which task variable* **[E]**
We know het has more *functional* PCs (PC-reconstruction crossover). Decompose the readout-relevant subspace
and ask **how many dimensions each task axis (language, parity, XOR) needs** in het vs hom, and whether het's
extra PCs are the ones that separate the hard axis (parity/Ger-odd). Use the CCGP coding vectors + Wakhloo's
prediction that **task-latent PCs map onto neural-activity PCs**. Turns "higher-dimensional" into "higher-
dimensional *in the task-relevant directions*."
*Papers:* [G] Wakhloo (PC↔latent map, PR); Bernardi 2020; [F] Driscoll (cross-condition subspaces).

### B4. Dynamical-mode spectrum (DMD / linearised timescales) **[E/R⁺]**
Fit a data-driven linear model (DMD/Koopman, or jPCA) to the hidden trajectories and compare the **spectrum
of decay timescales the dynamics actually express** in het vs hom. Does imposed τ diversity produce a broader
band of *emergent dynamical* timescales, or does the network compress to a few modes (consistent with the
low ~3-D trajectory PR)? Bridges the imposed-τ manipulation to observable dynamics.
*Papers:* [H] Sussillo–Barak; [F] Driscoll (dynamical motifs).

---

## Tier 3 — learning dynamics, robustness, generalisation (the "how it arises")

### C1. Training trajectory — *optimisation effect vs capacity effect* **[T, cheap]**
Retrain one het/hom pair with per-epoch checkpointing and track accuracy, **active-neuron count**, W,
separability across epochs. Does het's advantage appear **early** (an optimisation/gradient-flow effect that
prevents the dead-neuron collapse hom suffers — recall hom uses only ~20/64) or **accumulate late** (a
representational-capacity effect)? The active-neuron collapse in hom strongly suggests an optimisation story
worth pinning down. *Papers:* Perez-Nieves 2021 (robust learning); [G] (dim↔learning regime).

### C2. Temporal credit assignment — *why diverse τ trains better* **[R⁺]**
Diverse membrane timescales may improve gradient propagation over the 1 s window. Measure surrogate-gradient
norms / effective time-horizon of credit assignment across the trial for het vs hom. If het maintains
gradient signal further back in time, that mechanistically explains both the longer memory and the easier
optimisation. *Papers:* Perez-Nieves 2021; Jaeger 2001 / Dambre 2012 (memory horizon).

### C3. Robustness battery — *make Perez-Nieves' headline quantitative here* **[R⁺]**
Eval-only stress tests: input spike jitter, spike deletion (`p_del`), τ perturbation, weight quantisation,
and readout under fewer neurons. Prediction from the geometry (bigger margins, higher α_c capacity): het
**degrades more gracefully**. Ties separability → functional robustness. *Papers:* Perez-Nieves 2021;
Chung–Lee–Sompolinsky 2018 (capacity ↔ margin).

### C4. Sample-efficiency / low-data generalisation **[T]**
Wakhloo predicts low-dimensional codes generalise better in the low-data regime, higher-dim in the many-
sample regime. Retrain on fractions of the training set and see whether het's advantage **grows or shrinks**
with less data — a direct test of the dimension↔generalisation trade-off on our own networks.
*Papers:* [G] Wakhloo (explicit low-data prediction).

---

## Suggested sequencing

1. **Immediately, on existing data (no training):** A1 (τ-resolved function), A2 (sharpened ablation, folds
   into WS4), A4 (intrinsic timescales), B1 (target PID), B2 (ΦID — the toolbox is already here), B3
   (task-subspace alignment). These are the highest mechanistic yield per unit effort.
2. **Cheap re-eval (no full retrain):** A3 (τ-shuffle/transplant/dose–response — arguably the single most
   decisive causal test of diversity-vs-long-τ), B4 (dynamical modes), C2/C3 (credit assignment, robustness).
3. **Alongside the fixed-τ ladder (WS1):** run A1/A4/B3 on the 15/25/35 ms controls too, so every mechanistic
   measure is read across the τ ladder — that is what ultimately separates *diversity* from *long τ*.
4. **If time allows (training):** C1 (training trajectory), C4 (sample efficiency).

**The two I would prioritise:** **A3 (causal τ-shuffle / dose–response)** because it cleanly isolates
diversity from long-τ *without* retraining and complements WS1; and **A1 (τ-resolved single-neuron
function)** because it is the most direct answer to *where* the advantage lives and reframes every other
result around a single mechanistic axis (τ → role). B2 (ΦID) is a low-cost high-payoff bonus given the
toolbox is already vendored.

# Discrete W/M Z-scoring Pipeline (4-class, 64-neuron)

Discrete-estimator pipeline for computing **M-information z-scores** on the spiking activity of the
10-seed × {heterogeneous, homogeneous} 64-neuron 4-class SNNs. It replicates and scales up
Hardik's `wimfo_test.ipynb`, fixing the issues in the old Gaussian z-scoring pipeline.

## What it fixes vs. the old pipeline

1. **Spike extraction bug.** Spikes are taken from **index 2** (`layer_recs[0][2]`) of the recurrent
   layer's `(syn, mem, spk)` record — the binary spike output — not index 0 (synaptic current),
   which the earlier seeded notebooks were using as "spk".
2. **Wrong null family.** Uses the **discrete** `W_M_calculator(type="discrete")` (validated on
   XOR→pure-M / COPY→pure-W) with a **discrete TDMI-matched null** (`numit_discrete_optimized`),
   instead of a Gaussian-copula estimator + Gaussian VAR(1) null, which is not appropriate for
   binary spike data.
3. **Silent neurons.** Neurons that never spike are **excluded** before estimation (≈12/64 here),
   rather than being patched over with injected jitter.

## Contents

| File | Role |
|---|---|
| `Seeded run 1 4class 64neurons Discrete Zscore.ipynb` | the pipeline (run this) |
| `_build_discrete_zscore_nb.py` | builder that regenerates the `.ipynb` from plain-Python cell sources (edit here, re-run to rebuild) |
| `numit_discrete_optimized.py` | Hardik's discrete NuMIT null generator (bundled dependency) |
| `wimfo/` | the `wimfo` W/M-information package (bundled dependency) |
| `outputs/` | generated results — **gitignored** (spike cache, records, figures) |
| `Simplified Shuffle Zscore/` | sibling pipeline using a **shuffle null** instead of NuMIT (simpler, faster); shares this folder's `wimfo` and spike cache. See its own README. |

**Bundled** (committed with this folder): `numit_discrete_optimized.py`, `wimfo/`.
**Referenced from the repo** (not bundled — large/shared assets): the trained checkpoints under
`Project Files/Checkpoints/SeededRuns/seeded_run_1_64n/`, the SHD test set under `data/shd/`, and the
SNN model code under `neural_heterogeneity/SuGD_code/` (`model.py`, `data_gen.py`).

## Requirements

Everything except **numba** is already in the project env. Install it once:

```
pip install numba
```

Also needs: `torch` (CUDA used if available), `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`,
`h5py`, `tqdm`. See `requirements.txt`.

## How to run

Open the notebook and run top to bottom. It has three stages, **all resumable**:

- **Stage 0** — sanity-check the discrete estimator (XOR→W≈0,M≈1; COPY→W≈1,M≈0).
- **Stage 1** — load each checkpoint, extract index-2 binary spikes, cache one `.npy` per network in
  `outputs/spk_cache/`. Skips any network already cached. (~40 s/net, ~13 min total.)
- **Stage 2** — for each network and each subset size k, sample random k-subsets, compute observed
  W/M (group encoding, alphabet `2^(k/2)`), draw TDMI-matched discrete nulls, and z-score observed M
  (and W). Appends one JSON line per scored subset to `outputs/discrete_zscore_records.jsonl`; an
  interrupted run resumes by skipping already-recorded `(seed, arch, k, subset)` keys.
- **Stage 3** — summary table + het-vs-hom boxplots of M-z and W-z by k → `outputs/`.

## Method (faithful to Hardik)

Per k-subset: split the k neurons into two groups of k/2, encode each group to one symbol of
alphabet `A = 2^(k/2)`, and compute W/M on the 4-row system `[S1, S2, T1, T2]` at lag τ. The null
draws random logic-gate transitions with flip-noise η calibrated to the observed TDMI = W+M
(bivariate gates for k=2, multivariate for k≥4), and the observed M is scored as
`z = (M_obs − mean(M_null)) / std(M_null)` (population std). Default τ=1 step (≈4 ms at stride 4).

## Subset sizes and runtime

Capped at **k ∈ {2, 4, 6, 8}**. Measured per-null cost on this machine (the null is dominated by the
convex solve + gate search, so it is *not* monotonic in k):

| k | alphabet | ms/null | 100×100 per net | ×20 nets |
|---|---|---|---|---|
| 2 | 2 | 505 | 84 min | 28.0 h |
| 4 | 4 | 633 | 105 min | 35.1 h |
| 6 | 8 | 985 | 164 min | 54.7 h |
| 8 | 16 | 191 | 32 min | 10.6 h |
| 10 | 32 | 1,840 | 307 min | 102 h |
| 12 | 64 | 20,244 | 56 h | 1,125 h |

k=10/12 are excluded by default (infeasible); they can be re-added with a small budget in the config
cell, but note the observed PMF is **undersampled** there (cells `2^(2k)` ≫ samples), so their
z-scores are biased — the notebook prints a `samples/cells` ratio and an UNDERSAMPLED flag per (net,k).

> **Caveat:** the table above uses an easy fixed target (TDMI=0.3). On real spikes the TDMI targets
> are higher, so the NuMIT gate search runs longer — measured ~2.3 s/null at k=8 (vs 0.19 s above).
> Realistic cost for {2,4,6,8} @ 100×100 is therefore **~13 h/net** (~78 h for 3 seeds / 6 nets;
> ~260 h for the full 10 seeds). This is the main reason the shuffle-null variant in
> `Simplified Shuffle Zscore/` (~2.7× faster, ~5 h/net) is worth trying first. Currently set to
> **3 seeds**; the config cell lists the full 10. Drop the per-k budget to 50×50 to quarter these.

## Expected result

Validation on seed 101 reproduces Hardik's representative finding: observed M is near-zero and
dominated by W, and M z-scores are **negative** — i.e. the collective code is essentially pairwise,
with no genuine higher-order synergy above the TDMI-matched baseline at small k.

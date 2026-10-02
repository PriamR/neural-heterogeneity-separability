# Short-Tau + Scaled-Init Controls — Colab notebook

`Short-Tau + Scaled-Init Controls (Colab).ipynb` trains the two remaining homogeneous control families in one
resumable GPU notebook. Regenerate with `python code/training/_build_short_tau_and_scaled_init_nb.py`.

- **Part 1 — short fixed-τ (2 × 10 = 20 nets).** Frozen hidden τ_mem at **2.0** and **3.5 ms** — *below* the
  ~5 ms learned-hom optimum, to fill the short arm the old 15/25/35 ladder never covered.
- **Part 2 — scaled-init hom (10 nets).** Rebuild each hom at init, scale hidden weights by `c` to match het's
  activity, retrain the original learned-τ recipe, re-measure recruitment / n90 / accuracy.
- **Part 3 — sklearn-free verification** (accuracy / active-count / numpy-PCA n90 across all conditions).

## Assemble the Colab-upload folder
Mirror the existing `Fixed-Tau Sweep (Colab Upload)/` layout. Create
`Short-Tau + Scaled-Init Controls (Colab Upload)/` on Drive containing:

```
Short-Tau + Scaled-Init Controls (Colab Upload)/
  Short-Tau + Scaled-Init Controls (Colab).ipynb   # this notebook
  code/
    seeded_runs_common.py                          # from code/training/
    SuGD_code/                                      # copy of code/model_perez_nieves/ (renamed SuGD_code)
  reference_checkpoints/                            # het + hom(learned ~5 ms) .pt for all 10 seeds
  Tau from 15 epoch run.json                        # from code/training/
  init_scaling/
    init_scaling_constants.json                     # from the LOCAL diagnostic (Part 2 scale source)
```
SHD `.h5` live under `<RESEARCH_ROOT>/data/shd/` on Drive (not bundled). Set `RESEARCH_ROOT` in the paths cell.

## [CONFIRM] before running (single-line edits in the config cell)
- `SHORT_FIXED_TAUS_MS = [2.0, 3.5]` — the short rungs.
- `WHICH_WEIGHTS` — which weights `c` scales (default input + recurrent, readout off).
- `TARGET_METRIC` — match het mean-**rate** vs **active-fraction**.
- Run `notebooks/Initialization Weight-Scaling Diagnostic (Local).ipynb` first so `init_scaling_constants.json`
  supplies real matched `c`; otherwise Part 2 falls back to `MANUAL_SCALE` with a warning.

## Merge back
Copy `outputs/checkpoints/*` into `checkpoints/seeded_run_1_64n_controls/4class_lognormal/`, then re-run the
corrected PCA / PC-reconstruction / `paired_stats.py` passes over the extended condition set.

# Colab Run — Session Handoff

This folder is the launch point for **running the supervisor-feedback experiments** from a VS Code session
wired to the **Colab extension** (Colab GPU kernel). It contains everything a fresh assistant/session needs:
full context, the exact experiments to run, where the notebooks are, and a copy-paste prompt to start.

---

## 0. Copy-paste prompt to start the wired session

> **You are picking up the SNN temporal-heterogeneity project to run the supervisor-feedback experiments on
> a Colab GPU kernel (VS Code Colab extension).** Read, in order:
> `Project Files/Explanation files/Project Context & Status — Updated (2026-07-03).md`,
> `Project Files/Explanation files/Immediate Next Steps (Supervisor Feedback).md`, and this file
> (`Collab run/HANDOFF.md`). Then:
> 1. Confirm the Colab kernel is connected and GPU is on (`Runtime → Change runtime type → GPU`).
> 2. Confirm `PROJECT_ROOT` (this folder on Drive) and `SHD_DIR` (the folder with `shd_train.h5` /
>    `shd_test.h5`) are set in the training notebook's config cell.
> 3. **Run** `Fixed-Tau Sweep (Colab Upload)/Fixed-Tau Sweep + O-Info (Colab).ipynb` **top to bottom** —
>    Part 1 trains the 15 ms & 35 ms fixed-τ homogeneous controls (10 seeds each = 20 checkpoints), Part 2
>    computes the min/max O-Info + the Δτ-vs-info scatter across the τ ladder. It is **resumable**: finished
>    checkpoints / spike caches / O-records are skipped, so reconnect-and-rerun is safe.
> 4. Then run `Collab run/Tau Ablation (het).ipynb` (needs only existing het checkpoints, no training).
> 5. Report: per-seed accuracies for the 15/35 ms controls, the min/max-O table across 15/25/35/het, and the
>    ablation curves (largest- vs smallest-first). Flag any seed that fails to train or any missing cache.
> **Do not retrain the het or ~25 ms hom baselines** — they already exist in `reference_checkpoints/`.

---

## 1. What we are testing (why these runs)

Both het and hom currently learn solutions around ~25 ms. The open question: **is het's +13 pp advantage due
to the *heterogeneity* of time constants, or just to *having some longer* ones?** Plus a corrected O-Info
read (min/max, not average). See the two context docs above for the full background and results so far.

## 2. The experiments and where they live

| Experiment | Notebook | Needs GPU? | Output |
|---|---|---|---|
| **Train 15 ms & 35 ms fixed-τ hom controls** (10 seeds each, paired) | `Fixed-Tau Sweep (Colab Upload)/Fixed-Tau Sweep + O-Info (Colab).ipynb` — **Part 1** | **yes** | 20 checkpoints `…_hom_fixed15ms_…`, `…_hom_fixed35ms_…` |
| **Min/Max O-Info + Δτ scatter** across 15/25/35/het | same notebook — **Part 2** | no (CPU ok) | `fig_minmax_oinfo.png`, `fig_delay_vs_info.png`, `os_info_records.jsonl` |
| **τ-ordered ablation (het)** largest/smallest/random-first + per-class | `Collab run/Tau Ablation (het).ipynb` | no | `fig_tau_ablation.png`, `tau_ablation_records.json` |

The training notebook is **self-contained** (bundles `seeded_runs_common.py` + minimal `SuGD_code/`, and the
existing het + 25 ms hom reference checkpoints for comparison). Its own `README.md` in that folder documents
paths, the resumability contract, and the `syntropy` dependency for Part 2.

## 3. Run order & timing

1. **Part 1 training** — ~30–60 min/seed on a T4 → a few hours for the 20 nets. Resumable; disconnect-safe.
2. **Part 2 O-Info** — CPU, minutes, once caches exist.
3. **Ablation notebook** — CPU, minutes; runs on the existing het checkpoints in `reference_checkpoints/`
   (or the main-project `…/seeded_run_1_64n/4class_lognormal/` if running locally).

## 4. Merging results back

Copy `outputs/checkpoints/*` (the 20 controls) into
`Project Files/Checkpoints/SeededRuns/seeded_run_1_64n_homfix/4class_lognormal/`, then run **action 7**
(dimensionality vs τ) from `Immediate Next Steps (Supervisor Feedback).md` by re-pointing
`Representation Analysis/make_pca_corrected.py` and `run_pc_reconstruction_1ms.py` at those checkpoints.

## 5. Side-by-side

While training runs, the mechanism-expansion notebooks in `proposed expansions/` can run **in parallel** —
they need only existing checkpoints/caches (no GPU). See `proposed expansions/README` for the menu.

## 6. Guardrails

- **Don't** retrain het or the ~25 ms hom baseline (they exist; retraining would break the paired design).
- **Don't** unfreeze τ in the controls — the whole point is a *fixed* τ per control (`train_hom_ab=0`).
- Keep the paired-seed correspondence intact (each control seed pairs to the same het seed).
- If `syntropy` won't install on Colab, copy the local working package to Drive and add its parent to
  `sys.path` (one commented line in the notebook's setup cell). Part 1 (training) has no such dependency.

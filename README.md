# Neural heterogeneity promotes robust learning by increasing representational separability

Code, trained networks and analyses for the MSc thesis of the same title
(Priyam Rai, Imperial College London, 2026; supervised by Marcus Ghosh and
Hardik Rajpal), which builds on Perez-Nieves et al., "Neural heterogeneity
promotes robust learning", Nat Commun 12, 5791 (2021).

Twenty matched pairs of recurrent LIF networks are trained on the Spiking
Heidelberg Digits collapsed to four classes; the heterogeneous network of a
pair learns sixty-four per-neuron membrane time constants, its homogeneous
twin learns one shared constant, and everything else is held identical. The
heterogeneous networks reach 70.3% against 60.2% for the homogeneous ones,
and the thesis traces that gap to representational separability, which
predicts accuracy across all forty networks at r = 0.87.

## Layout

    model/          LIF network, surrogate gradient, data loading (Perez-Nieves et al. 2021, GPL-3.0)
    code/           the full research pipeline, plus the notebook helpers at its top level
      analysis_suite/        main analyses and thesis figures (accuracy, PCA, shattering,
                             capacity, O-information, memory, input spectrum, LaTeX tables)
      supervisor_extension/  cross-seed structure analyses (modularity, division of labour,
                             block concentration, temporal lock-in)
      training/              Colab training notebooks for the pairs and controls, and
                             seeded_runs_common.py
      information_theory/    NuMIT and O-information estimators
      expansions/            exploratory follow-ups (tau-resolved function, causal tau,
                             intrinsic timescales, target PID, task subspace)
      model_perez_nieves/    the model code again, vendored for the script pipeline
    notebooks/      curated end-to-end pipeline, numbered in run order
    checkpoints/    the 40 trained networks plus fixed-tau and scaled-init controls
    cache/          per-network summaries included; spike and representation caches from the link below
    figures/        notebook output
    tables/         notebook output
    replication/    the cross-seed replication package as reviewed by the supervisors,
                    kept verbatim (its own README documents it)

The script paths under `code/` match the ones cited in Appendix C of the
thesis.

## Setup

Python 3.10+, then:

    pip install torch numpy scipy scikit-learn matplotlib h5py tables joblib syntropyx pymanopt cvxopt

Data: download `shd_train.h5` and `shd_test.h5` from
https://zenkelab.org/datasets/ into `data/shd/` (create the folder if needed).

Caches: the binarised spike trains and forward-pass representations (~1.5 GB)
are here:
https://drive.google.com/drive/folders/1dLovdaxYBwl7-9LFib68Z23cY5EhENvK
The download unpacks as `spk_cache_4ms/` with `representation_clean/` inside
it; place `spk_cache_4ms/` in `cache/` and move `representation_clean/` up so
both sit directly under `cache/`. Alternatively regenerate both with
`notebooks/01_extract.ipynb` (needs a GPU, ~1 h). The per-network
summaries (`cache/network_json/`) and precomputed analysis results
(`cache/results/`) ship with the repo, so notebooks 02 and 04-06 run without
the large caches. 03's shattering and temporal figures also run from the
shipped summaries; its capacity figure and any recompute from scratch need
the caches.

## Run order

Training is optional, all checkpoints are included.

    notebooks/train_pairs.ipynb       the 20 het/hom pairs (Colab GPU)
    notebooks/train_controls.ipynb    fixed-tau ladder and scaled-init controls
    notebooks/01_extract.ipynb        checkpoints -> caches
    notebooks/02_main_results.ipynb   accuracy, tau distributions, recruitment, separability
    notebooks/03_geometry.ipynb       shattering, temporal generalisation, capacity
    notebooks/04_fixed_tau_controls.ipynb
    notebooks/05_oinfo.ipynb          O-information
    notebooks/06_theory.ipynb         filter-bank model and input spectrum

When regenerating everything from scratch, rerun 02 after 03: one figure in 02
reads a capacity summary that 03 writes (the shipped copy in cache/results
covers a fresh clone).

The scripts under `code/analysis_suite/` and `code/supervisor_extension/` are
the versions behind the thesis figures and tables; they expect the full cache
layout above and name their inputs at the top of each file.

## License

`model/` and `code/model_perez_nieves/` are the surrogate-gradient code
released with Perez-Nieves, Leung, Dragotti and Goodman, "Neural heterogeneity
promotes robust learning", Nat Commun 12, 5791 (2021),
https://doi.org/10.1038/s41467-021-26022-3, GPL-3.0. The rest of the
repository is released under GPL-3.0 as well; see LICENSE.

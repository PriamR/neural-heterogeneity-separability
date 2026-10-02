# Cross-seed replication of the information analyses

Twenty paired seeds, forty networks. Reply to the O-Info Review Package, which ran its analyses on seed 1001
alone and named cross-seed replication as its first outstanding item.

## Where to start

Open **`Cross-Seed Replication.ipynb`**. Each method is demonstrated live on one pair of networks, where it
runs in seconds, and the twenty-seed result is loaded and its statistics recomputed in the notebook rather
than read from a stored summary. The full sweep takes about four hours on sixteen cores, so it is not run
inside the notebook; every script that produced it is in `code/`.

Then read **`REPORT - Structure and Replication.md`**, which is the write-up with every figure and table.
Part I is our own new representational work (shattering, manifold capacity, dimensionality, timing); Part II
is the replication.

## Contents

| | |
|---|---|
| `Cross-Seed Replication.ipynb` | runnable review notebook |
| `REPORT - Structure and Replication.md` | the full write-up, 28 figures |
| `significance_tables.md` | every comparison with paired t, Wilcoxon and exact sign-flip |
| `replication_evidence.md` | claim-by-claim verdicts against the original |
| `METHOD_NOTES.md` | where this departs from the original package, and why |
| `code/` | every script, including the three audits |
| `results/` | result JSON with per-network values, so any statistic can be re-derived |
| `figures/` | every figure |

## Reading the statistics

Every comparison is paired within seed: the two networks of a seed share initial weights, data order and
training schedule, so the only free difference is the hidden membrane time constant. Three tests are always
reported together, never one: paired t, Wilcoxon, and an exact sign-flip permutation over all 2^20 sign
assignments. Both non-parametric tests floor at 2/2^20 = 1.9e-6, which is what a unanimous 20/20 reads as; a
p at that floor is a maximally significant result, not a weak one.

## Reading the figures

Chart types follow the original package: grouped bars with error bars for group comparisons, `-o` lines for
swept parameters, matrices and spring layouts for structure. Bars are means over the twenty networks with a
**95% CI on the mean** as the whisker, faint dots are individual networks, and each bar group is annotated
with its own permutation p and the number of seeds that moved in the direction of the mean. Sweeps carry a
shaded 95% CI band. `S1_modularity_n20.png` is the panel-for-panel analogue of `network_analysis.ipynb`
cell 19, with each line now a twenty-seed mean instead of one network's value.

One change of substance from the original package. It tests pair-level results with a Mann-Whitney over unit
pairs. Pairs inside a network are not independent draws, so the effective sample size is far below the
nominal one and the resulting p describes variation among pairs rather than between architectures. Here each
network contributes one number and the twenty within-seed differences are the unit of analysis.

## Requirements

`numpy scipy scikit-learn networkx matplotlib`. The notebook needs nothing else; the full pipeline
additionally uses `joblib`, `h5py`, `syntropy` and `mftma`.

## Reproducing

```
python code/supervisor_extension/run_modularity_n20.py
python code/supervisor_extension/run_structure_matrices_n20.py
python code/supervisor_extension/run_block_concentration_n20.py
python code/supervisor_extension/run_block_k_sweep.py
python code/supervisor_extension/run_rsi_n20.py
python code/supervisor_extension/run_temporal_lockin_n20.py
python code/supervisor_extension/run_division_of_labour_n20.py
```

Paths in these scripts point at the original repository layout; in this package the caches are under
`spike_caches/` and bit-packed, so `sup_common.load_spikes` needs its cache path and unpacking adjusted. The
notebook shows the loader for the packaged form.

## The spike caches are supplied separately

This copy carries no `spike_caches/` directory: the forty binarised spike matrices are about 46 MB packed
and travel by shared link rather than by email. Nothing in the write-up depends on having them.

The notebook ships **already executed**, so every number and figure in it is readable as it stands. Of its
cells, only the three that demonstrate a method live on one pair of networks read the caches; the cross-seed
statistics are recomputed from `results/*.json`, which is included here. Drop the `spike_caches/` folder from
the link beside this README and the whole notebook re-runs end to end.

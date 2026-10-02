# Method notes: where this extension departs from the seed-1001 package

Five things changed in going from n = 1 to n = 20. Four are corrections, one is an optimisation. All of them
affect how the numbers should be read, so they are recorded here rather than buried in the scripts.

## 1. The unit of analysis is the network, not the pair

The package tests its RSI and functional-connectivity results with a Mann-Whitney U over unit pairs, and
reports p < 1e-4. That test is not valid for the claim it is attached to. The pairs inside a network are not
independent draws: they are built from the same 47 units, the same training run and the same random seed, so
the effective sample size is far below the nominal one and the p-value describes variation among pairs rather
than between architectures. With one network per arm there is no way to separate the two.

Here every network contributes a single number, the mean over its sampled pairs, and the twenty within-seed
differences are the unit of analysis. This is a much smaller nominal n and a much larger real one. Pair-level
distributions are still plotted, as a description of spread, and no claim rests on them.

## 2. The cache is sub-sampled, not binned

`ANALYSIS_SUMMARY.md` describes the cache as "downsampled to 4 ms/bin". The code that built it takes
`spk[:, ::4, :]` (`code/analysis_suite/run_oinfo.py:91`), which keeps every fourth 1 ms step and discards the
other three rather than combining them. A cached "count" is therefore about a quarter of the unit's true
spike count, and the 250 samples are a subsample of the trial rather than a coarsening of it.

Nothing in these analyses is invalidated: both arms go through the identical transform, and every measure is
either null-corrected or paired within seed, so the transform cancels. But absolute firing rates quoted from
this pipeline are not the networks' firing rates, and the "active-bin fraction" numbers in the package
(12.7% against 4.6%) inherit the same caveat.

## 3. Seed 101's caches elsewhere in the repository are a stale network

Twenty-nine of the forty caches existed. Nine of the missing het caches were recoverable from
`Project Files/Discrete Zscore Pipeline/outputs/spk_cache`, and were verified against
`data_cache/representation_clean` by two independent checks: correlation of per-trial counts (0.995 to 0.999,
matching the value a known-good cache gives against itself under the stride-4 subsample) and the exact
inequality that a subsampled count can never exceed the full count, which held for every one of the 2048
trials times 47 units.

Seed 101 failed both checks in every location it appeared, at correlation 0.19 and with the count inequality
violated on 22% of entries. It is an older network under the same filename. Rather than mix provenance, all
eleven missing caches were regenerated from the checkpoints through the same `build_cache` that produced the
original twenty-nine, so every cache in the directory now comes from one code path. All forty then pass a
label-alignment check by linear decode (0.636 to 0.841 against a 0.25 chance line).

## 4. On dense graphs the degree-preserving null is mostly a weight permutation

`randomize` calls `nx.double_edge_swap(H, nswap=10*m, max_tries=100*m)` inside a bare `except Exception:
pass`. On the full functional graph, whose density is about 0.89, and on the cumulative-count graph, whose
density is close to 1.0, almost every proposed swap would create an edge that already exists. The call
exhausts `max_tries` and raises, which the handler swallows; the graph is left partially swapped, and the
weight reshuffle that follows does most of the randomising.

This does not invalidate the normalisation. For a near-complete weighted graph, modularity is carried almost
entirely by the pattern of weights rather than by which edges exist, so permuting weights across a fixed
topology is the right null and arguably a cleaner one. It does mean `Q/Q_rand` on the full graph should be
described as normalised against a weight-permutation null, not a degree-preserving rewiring null, and the
matched-density sweep (where the graphs are sparse and swaps succeed) is the version that tests topology
proper. Both are reported.

## 5. The accumulation estimator was recast as a matrix product

`disc_jmi_smi` contracts the class-conditional indicator stack with `np.einsum('ain,bjn->abij', Ac, Ac)`.
Flattening the level and unit axes into one row index makes that the Gram matrix `Ac @ Ac.T`, which is the
same arithmetic routed through BLAS. It is exact, verified to zero absolute difference against the einsum on
both a four-class and a two-class target, and about an order of magnitude faster once the indicator stack is
also hoisted out of the permutation loop. At roughly 1300 evaluations per network times forty networks the
original form was not affordable.

Two smaller changes in the same analysis. Lock-in times are linearly interpolated at the ninety-per-cent
crossing rather than snapped to the nearest grid point: the grid spacing is 25 ms and the seed-1001 gap under
test was 176 ms, so quantisation could visibly distort it. And the same permutations are reused at every time
point, which pairs the null across time and removes a source of jitter from the crossing estimate.

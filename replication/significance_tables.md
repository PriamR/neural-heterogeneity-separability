# Cross-seed significance tables

Twenty within-seed pairs. Each pair shares initial weights, data order and training schedule; only
the hidden membrane time constant differs. `Δ` is the mean paired difference het minus hom, `d_z`
is Cohen's d for paired differences, and `wins` counts the seeds on which the heterogeneous network
is higher. The sign-flip permutation is exact over all 2^20 sign assignments and floors at 1.9e-6,
which is what a unanimous 20/20 reads as. Holm adjusts across the rows of each table.

### Functional-network modularity

| Metric | het | hom | Δ | d_z | paired t | Wilcoxon | sign-flip | Holm | wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-graph $Q/Q_{rand}$ (binned) | 0.573 | 0.674 | -0.1012 | -0.25 | 0.281 | 0.231 | 0.280 | 0.559 | 7/20 |
| Full-graph modularity $Q$ | 0.157 | 0.193 | -0.03553 | -0.35 | 0.131 | 0.143 | 0.131 | 0.522 | 8/20 |
| Small-worldness $\sigma$ | 0.507 | 0.395 | +0.1111 | +0.45 | 0.060 | 0.058 | 0.060 | 0.421 | 14/20 |
| Communities (full graph) | 3.5 | 3.1 | +0.4 | +0.40 | 0.088 | 0.083 | 0.136 | 0.522 | 9/20 |
| Cumulative-count $Q/Q_{rand}$ | 1.251 | 1.018 | +0.2325 | +0.42 | 0.073 | 0.097 | 0.074 | 0.442 | 14/20 |
| Cumulative-count $\sigma$ | 0.726 | 0.561 | +0.1656 | +0.97 | 3.5e-04 | 0.001 | 6.2e-04 | 0.007 | 16/20 |
| Community $\eta^2$ language | 0.0288 | 0.0101 | +0.01871 | +0.73 | 0.004 | 0.011 | 0.002 | 0.021 | 14/20 |
| Community $\eta^2$ parity | 0.0752 | 0.0668 | +0.008433 | +0.12 | 0.590 | 0.498 | 0.589 | 0.589 | 10/20 |
| Cumulative community $\eta^2$ 4-class | 0.1133 | 0.0823 | +0.03107 | +0.51 | 0.034 | 0.036 | 0.034 | 0.273 | 12/20 |
| Active units | 47.6 | 42.5 | +5.15 | +0.94 | 4.7e-04 | 0.003 | 6.3e-04 | 0.007 | 17/20 |
| Mean edge MI (bits) | 0.0227 | 0.0334 | -0.01072 | -0.39 | 0.100 | 0.048 | 0.089 | 0.444 | 5/20 |

A $Q/Q_{rand}$ of one is the random null: values at or below it mean the community structure is no stronger than chance. On the full and cumulative graphs, whose densities are about 0.89 and 1.0, the degree-preserving swap cannot find legal moves and the null reduces to a permutation of weights over a near-fixed topology; the matched-density sweep, where the graphs are sparse and swaps succeed, is the test of topology proper. See METHOD_NOTES.md.

#### Modularity at matched edge density

The full-graph rows above cannot separate topology from weights. These can: at a fixed edge density both graphs have the same number of edges by construction, they are sparse enough for the degree-preserving swap to succeed, and the binary variant discards weights entirely so only topology remains.

| Weighting | Density | het | hom | Δ | d_z | sign-flip p | het higher |
|---|---:|---:|---:|---:|---:|---:|---:|
| weighted | 0.05 | 0.808 | 0.773 | +0.035 | +0.07 | 0.759 | 11/20 |
| weighted | 0.10 | 0.796 | 0.804 | -0.009 | -0.02 | 0.938 | 9/20 |
| weighted | 0.15 | 0.771 | 0.865 | -0.094 | -0.18 | 0.430 | 7/20 |
| weighted | 0.20 | 0.739 | 0.852 | -0.113 | -0.22 | 0.336 | 8/20 |
| weighted | 0.25 | 0.712 | 0.839 | -0.127 | -0.25 | 0.267 | 8/20 |
| weighted | 0.30 | 0.688 | 0.808 | -0.120 | -0.24 | 0.285 | 8/20 |
| binary | 0.05 | 1.292 | 1.279 | +0.013 | +0.04 | 0.866 | 11/20 |
| binary | 0.10 | 1.317 | 1.429 | -0.113 | -0.25 | 0.276 | 10/20 |
| binary | 0.15 | 1.359 | 1.587 | -0.228 | -0.68 | 0.007 | 5/20 |
| binary | 0.20 | 1.385 | 1.631 | -0.246 | -0.55 | 0.024 | 6/20 |
| binary | 0.25 | 1.427 | 1.706 | -0.279 | -0.63 | 0.013 | 6/20 |
| binary | 0.30 | 1.513 | 1.767 | -0.254 | -0.65 | 0.010 | 7/20 |

### Modularity without a randomisation null

| Metric | het | hom | Δ | d_z | paired t | Wilcoxon | sign-flip | Holm | wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Within/between weight ratio (Louvain) | 2.98 | 4.59 | -1.605 | -0.99 | 3.0e-04 | 1.3e-04 | 1.2e-04 | 3.5e-04 | 3/20 |
| Within/between weight ratio (fixed K = 3) | 4.08 | 7.14 | -3.054 | -0.97 | 3.7e-04 | 0.001 | 7.4e-04 | 0.001 | 5/20 |
| Within/between weight ratio (cumulative graph) | 1.93 | 2.54 | -0.6087 | -0.79 | 0.002 | 0.001 | 0.002 | 0.002 | 4/20 |

The ratio of mean within-community edge weight to mean between-community edge weight: what a block matrix shows, with no randomisation anywhere in it. Every row favours the homogeneous network, including the fixed-K row where the community count cannot contribute and the cumulative row where $Q/Q_{rand}$ said the opposite. Across the forty networks this ratio correlates with full-graph $Q/Q_{rand}$ at only r = +0.21, which is why the two measures reach different conclusions.

### Pairwise redundancy and synergy of the class information

| Metric | het | hom | Δ | d_z | paired t | Wilcoxon | sign-flip | Holm | wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single-unit MI $I(X_i;Y)$ (bits) | 0.1924 | 0.1504 | +0.04206 | +0.91 | 6.2e-04 | 3.9e-04 | 4.7e-04 | 0.003 | 16/20 |
| Pair joint MI $I(X_i,X_j;Y)$ (bits) | 0.4272 | 0.3482 | +0.07894 | +0.98 | 3.2e-04 | 1.3e-04 | 1.9e-04 | 0.001 | 18/20 |
| Redundancy $\min(I_i,I_j)$ (bits) | 0.1352 | 0.0995 | +0.03572 | +1.10 | 9.6e-05 | 4.8e-05 | 6.5e-05 | 5.8e-04 | 17/20 |
| Synergy $I_{joint}-\max(I_i,I_j)$ (bits) | 0.1776 | 0.1470 | +0.03055 | +0.89 | 7.9e-04 | 2.6e-04 | 3.3e-04 | 0.002 | 16/20 |
| RSI $=$ synergy $-$ redundancy (bits) | 0.0424 | 0.0475 | -0.005172 | -0.15 | 0.508 | 0.546 | 0.504 | 1.000 | 10/20 |
| Redundancy $/\,I_{joint}$ | 0.3190 | 0.2878 | +0.03116 | +0.77 | 0.003 | 0.002 | 0.004 | 0.019 | 15/20 |
| Synergy $/\,I_{joint}$ | 0.4110 | 0.4192 | -0.00813 | -0.14 | 0.535 | 0.409 | 0.533 | 1.000 | 7/20 |
| RSI $/\,I_{joint}$ | 0.0921 | 0.1314 | -0.03929 | -0.45 | 0.058 | 0.070 | 0.059 | 0.237 | 7/20 |
| Fraction of pairs with RSI $>0$ | 0.6988 | 0.7528 | -0.054 | -0.40 | 0.090 | 0.167 | 0.091 | 0.274 | 9/20 |

The upper block is in bits and the lower block divides each quantity by the pair's joint MI. The division is the test that matters: raw bits rise with how much class information a pair carries, so a difference there can be scale alone, while the normalised rows ask whether the redundancy-synergy balance itself has changed.

### Timing of the class code and the order in which the factors resolve

| Metric | het | hom | Δ | d_z | paired t | Wilcoxon | sign-flip | Holm | wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Instantaneous 4-class COM (ms) | 466.5 | 432.3 | +34.22 | +0.33 | 0.154 | 0.294 | 0.152 | 0.527 | 11/20 |
| Instantaneous language COM (ms) | 426.9 | 320.7 | +106.2 | +1.09 | 1.1e-04 | 2.1e-04 | 1.7e-04 | 0.001 | 16/20 |
| Instantaneous parity COM (ms) | 490.5 | 462.5 | +28.07 | +0.24 | 0.292 | 0.349 | 0.289 | 0.577 | 10/20 |
| Instantaneous 4-class spread (ms) | 229.3 | 226.3 | +3.011 | +0.07 | 0.758 | 0.869 | 0.756 | 0.756 | 9/20 |
| Instantaneous COM gap (parity $-$ language, ms) | 63.6 | 141.7 | -78.17 | -0.56 | 0.021 | 0.024 | 0.022 | 0.109 | 7/20 |
| 4-class lock-in (ms) | 550.9 | 430.7 | +120.2 | +0.76 | 0.003 | 0.003 | 0.004 | 0.027 | 17/20 |
| Language lock-in (ms) | 551.6 | 353.1 | +198.5 | +0.99 | 2.9e-04 | 7.1e-04 | 5.3e-04 | 0.004 | 17/20 |
| Parity lock-in (ms) | 578.8 | 468.8 | +110 | +0.75 | 0.003 | 0.003 | 0.004 | 0.027 | 17/20 |
| Lock-in gap (parity $-$ language, ms) | 27.3 | 115.7 | -88.49 | -0.35 | 0.132 | 0.165 | 0.132 | 0.527 | 8/20 |

A positive gap means language resolves before parity. The paired column asks whether the two arms differ in their gap; whether either arm has a non-zero gap at all is a separate question, tabulated below.

### Is either arm actually ordered?

The table above compares the two arms' gaps to each other. This one asks the prior question, separately for each arm: is the gap different from zero at all? A one-sample sign-flip permutation on the twenty per-seed gaps, plus the count of seeds in which language resolves first.

| Arm | Clock | Mean gap (ms) | Median (ms) | SD (ms) | Language-first | Sign-flip p |
|---|---|---:|---:|---:|---:|---:|
| het | accumulated lock-in | +27.3 | -13.8 | 149.7 | 8/20 | 0.425 |
| het | instantaneous COM | +63.6 | +55.3 | 76.9 | 15/20 | 0.001 |
| hom | accumulated lock-in | +115.7 | +156.2 | 191.4 | 13/20 | 0.015 |
| hom | instantaneous COM | +141.7 | +117.2 | 121.9 | 18/20 | 2.7e-05 |

Ordering tally with a 20 ms tolerance band: `{"het": {"parity-first": 9, "language-first": 8, "concurrent": 3}, "hom": {"parity-first": 7, "language-first": 13}}`

### Division of labour across modules

| Metric | het | hom | Δ | d_z | paired t | Wilcoxon | sign-flip | Holm | wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| $\eta^2$(selectivity $\sim$ module) | 0.113 | 0.211 | -0.09822 | -0.58 | 0.018 | 0.021 | 0.019 | 0.076 | 6/20 |
| $\eta^2$ excess over its permutation null | 0.059 | 0.160 | -0.1009 | -0.61 | 0.013 | 0.015 | 0.015 | 0.073 | 7/20 |
| Number of modules | 3.5 | 3.1 | +0.4 | +0.40 | 0.088 | 0.083 | 0.136 | 0.408 | 9/20 |
| corr(language MI, parity MI) over units | 0.116 | 0.108 | +0.008394 | +0.03 | 0.888 | 1.000 | 0.887 | 0.887 | 9/20 |
| Jaccard overlap of top language/parity units | 0.187 | 0.161 | +0.02661 | +0.18 | 0.422 | 0.341 | 0.421 | 0.842 | 11/20 |
| Mean per-unit language MI (bits) | 0.0277 | 0.0157 | +0.01202 | +0.98 | 3.1e-04 | 2.1e-04 | 1.9e-04 | 0.001 | 17/20 |
| Mean per-unit parity MI (bits) | 0.0646 | 0.0491 | +0.0155 | +0.86 | 0.001 | 7.1e-04 | 9.7e-04 | 0.006 | 18/20 |

The excess row subtracts each network's own permutation null, because $\eta^2$ rises with the number of groups even under the null and the two arms do not find the same number of modules. Within-network permutation tests reached p < 0.05 in 5/20 heterogeneous and 13/20 homogeneous networks.

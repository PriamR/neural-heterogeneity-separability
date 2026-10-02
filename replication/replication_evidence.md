# Cross-seed replication of the information analyses

Every result below is a within-seed paired comparison over 20 seeds, against the supervisor's
seed-1001 values from `ANALYSIS_SUMMARY.md`. The seed-1001 column is what was claimed; the het and
hom columns are the twenty-seed means; the verdict applies the Holm-adjusted sign-flip p within
each section.

## Modularity of the functional network, by the package's statistic

The topological claim: the homogeneous graph is modular and small-world, the heterogeneous one is not. Read this table together with the next one: on these dense graphs Q/Q_rand is unreliable in both arms, so the nulls below are evidence about the statistic rather than about the networks.

| Quantity | seed 1001 | het (n=20) | hom (n=20) | Δ | d_z | sign-flip p | Holm | wins | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Full-graph Q/Q_rand | het 0.62, hom 1.39 | 0.5731 | 0.6743 | -0.1012 | -0.25 | 0.280 | 0.559 | 7/20 | holds in direction, not significant |
| Cumulative-count Q/Q_rand | het 1.10, hom 1.87 | 1.251 | 1.018 | +0.2325 | +0.42 | 0.074 | 0.240 | 14/20 | claimed effect absent |
| Small-worldness sigma | hom up to 3.3, het about 1 | 0.5066 | 0.3955 | +0.1111 | +0.45 | 0.060 | 0.240 | 14/20 | claimed effect absent |
| Community eta2 language | het 0.13 higher | 0.02877 | 0.01006 | +0.01871 | +0.73 | 0.002 | 0.014 | 14/20 | replicates |
| Community eta2 parity | hom 0.23 higher | 0.07519 | 0.06676 | +0.008433 | +0.12 | 0.589 | 0.589 | 10/20 | claimed effect absent |
| Cumulative community eta2 4-class | het 0.11, hom 0.03 | 0.1133 | 0.08225 | +0.03107 | +0.51 | 0.034 | 0.171 | 12/20 | holds in direction, not significant |

## Modularity of the functional network, without a null

The same question asked by comparing within-community edge weight against between-community edge weight directly. This is what the block matrices display, and it is the version that should be believed.

| Quantity | seed 1001 | het (n=20) | hom (n=20) | Δ | d_z | sign-flip p | Holm | wins | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Within/between weight ratio (Louvain) | the block matrices | 2.985 | 4.59 | -1.605 | -0.99 | 1.2e-04 | 3.5e-04 | 3/20 | replicates |
| Within/between weight ratio (fixed K) | the block matrices | 4.082 | 7.136 | -3.054 | -0.97 | 7.4e-04 | 0.001 | 5/20 | replicates |
| Within/between weight ratio (cumulative) | the block matrices | 1.928 | 2.537 | -0.6087 | -0.79 | 0.002 | 0.002 | 4/20 | replicates |

## Redundancy and synergy of the class information

The information claim: heterogeneous pairs carry more of both, but the balance between them is unchanged.

| Quantity | seed 1001 | het (n=20) | hom (n=20) | Δ | d_z | sign-flip p | Holm | wins | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Redundancy (bits) | het 0.104, hom 0.051 | 0.1352 | 0.09953 | +0.03572 | +1.10 | 6.5e-05 | 3.9e-04 | 17/20 | replicates |
| Synergy (bits) | het 0.187, hom 0.097 | 0.1776 | 0.147 | +0.03055 | +0.89 | 3.3e-04 | 0.002 | 16/20 | replicates |
| RSI (bits) | het 0.082, hom 0.046 | 0.04235 | 0.04752 | -0.005172 | -0.15 | 0.504 | 1.000 | 10/20 | claimed effect absent |
| Redundancy / I_joint | fraction survives | 0.319 | 0.2878 | +0.03116 | +0.77 | 0.004 | 0.015 | 15/20 | replicates |
| Synergy / I_joint | fraction survives | 0.411 | 0.4192 | -0.00813 | -0.14 | 0.533 | 1.000 | 7/20 | claimed effect absent |
| RSI / I_joint | null at seed 1001 | 0.09209 | 0.1314 | -0.03929 | -0.45 | 0.059 | 0.178 | 7/20 | null confirmed |

## Timing and the order in which the factors resolve

The processing claim: the homogeneous network resolves language first and parity second, the heterogeneous one integrates both concurrently and late.

| Quantity | seed 1001 | het (n=20) | hom (n=20) | Δ | d_z | sign-flip p | Holm | wins | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Instantaneous 4-class COM (ms) | het 479, hom 413 | 466.5 | 432.3 | +34.22 | +0.33 | 0.152 | 0.264 | 11/20 | holds in direction, not significant |
| 4-class lock-in (ms) | het 536, hom 360 | 550.9 | 430.7 | +120.2 | +0.76 | 0.004 | 0.016 | 17/20 | replicates |
| Language lock-in (ms) | het 612, hom 232 | 551.6 | 353.1 | +198.5 | +0.99 | 5.3e-04 | 0.003 | 17/20 | replicates |
| Parity lock-in (ms) | het 512, hom 408 | 578.8 | 468.8 | +110 | +0.75 | 0.004 | 0.016 | 17/20 | replicates |
| Lock-in gap, parity minus language (ms) | het -100, hom +176 | 27.26 | 115.7 | -88.49 | -0.35 | 0.132 | 0.264 | 8/20 | holds in direction, not significant |

The paired row above asks whether the two arms differ in their gap. Whether either arm has
a gap at all is the prior question, and it is the one the sequential reading depends on:

| Arm | Mean gap (ms) | Median | Language-first | Sign-flip p |
|---|---:|---:|---:|---:|
| het | +27.3 | -13.8 | 8/20 | 0.425 |
| hom | +115.7 | +156.2 | 13/20 | 0.015 |

Ordering tally with a 20 ms tolerance: `{"het": {"parity-first": 9, "language-first": 8, "concurrent": 3}, "hom": {"parity-first": 7, "language-first": 13}}`

## Division of labour across modules

The keystone: module identity predicts what a unit codes for in the homogeneous network and not in the heterogeneous one.

| Quantity | seed 1001 | het (n=20) | hom (n=20) | Δ | d_z | sign-flip p | Holm | wins | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| eta2(selectivity ~ module) | het 0.09, hom 0.27 | 0.1131 | 0.2113 | -0.09822 | -0.58 | 0.019 | 0.059 | 6/20 | holds in direction, not significant |
| eta2 above its own permutation null | not reported | 0.05924 | 0.1601 | -0.1009 | -0.61 | 0.015 | 0.059 | 7/20 | holds in direction, not significant |
| corr(language MI, parity MI) | het overlapping, hom disjoint | 0.1159 | 0.1075 | +0.008394 | +0.03 | 0.887 | 0.887 | 9/20 | holds in direction, not significant |
| Jaccard of top language/parity units | het 0.16, hom 0.05 | 0.1874 | 0.1607 | +0.02661 | +0.18 | 0.421 | 0.842 | 11/20 | holds in direction, not significant |

Within-network permutation tests of module alignment reached p < 0.05 in 5/20 heterogeneous and 13/20 homogeneous networks.

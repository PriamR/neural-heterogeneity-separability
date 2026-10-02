"""The structural figures of the modularity chapter, extended to twenty seeds.

These are the analogues of the figures the package leads with, the ones where modularity is something you
look at rather than something you read off a number: the spring layout with coloured communities, the
community-ordered adjacency block matrix, the MI heatmaps, and the community-by-class tuning grids.

They matter here for a specific reason. The block matrix is the most persuasive single image in the package,
because a modular graph shows visible blocks on the diagonal and a diffuse one does not, and on seed 1001 the
contrast between the two arms is stark. The claim under test is that the contrast does not generalise, and
the honest way to test a visual claim is with the same visual, drawn for every seed. That is what the montage
in S9 is: forty block matrices, the same colour scale, the pair the package used marked.

Everything is drawn from the arrays stored by run_structure_matrices_n20.py, which reproduce the reported
Q/Q_rand exactly, so these pictures and the paired tables describe one computation.

Run: python code/supervisor_extension/make_structure_figures.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sup_common as SC
import style as S
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from run_modularity_n20 import full_graph, graph_at_density

S.apply()
FIGS = SC.FIGS
ARCHS = ("het", "hom")
CLASSES = SC.CLASSES
LANG_C, PAR_C = "#7b3fa0", "#1b9e77"
Z = np.load(SC.DATA / "structure_matrices_n20.npz")
SCAL = json.load(open(SC.DATA / "structure_scalars_n20.json"))
MOD = json.load(open(SC.DATA / "modularity_n20.json"))


def g(seed, arch, field):
    return Z[f"{seed}_{arch}__{field}"]


def tun(seed, arch, kind, field):
    return Z[f"{seed}_{arch}__tun_{kind}_{field}"]


def order_by_module(modid):
    """Node order that groups communities together, largest first, and the block boundaries."""
    order, bounds, b = [], [], 0
    for ci in range(modid.max() + 1):
        idx = np.where(modid == ci)[0]
        order.extend(idx.tolist())
        b += len(idx)
        bounds.append(b)
    return np.array(order, int), bounds[:-1]


def block_panel(ax, W, modid, vmax=None, boundcol="#00e5ff", lw=0.8):
    """Adjacency reordered so communities sit in contiguous blocks. Blocks on the diagonal are modularity."""
    order, bounds = order_by_module(modid)
    A = W[np.ix_(order, order)]
    vmax = vmax if vmax is not None else np.percentile(A[A > 0], 99) if (A > 0).any() else 1.0
    im = ax.imshow(A, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    for b in bounds:
        ax.axhline(b - 0.5, color=boundcol, lw=lw)
        ax.axvline(b - 0.5, color=boundcol, lw=lw)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# ---------------------------------------------------------------- S7: the package's own figure, reproduced
def fig_seed1001_reproduction():
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 11.0))
    for c, arch in enumerate(ARCHS):
        W, modid = g(1001, arch, "W"), g(1001, arch, "modid")
        sc = SCAL[f"1001_{arch}"]
        G = full_graph(W)
        ncomm = int(modid.max() + 1)
        cmap = plt.cm.tab20(np.linspace(0, 1, max(ncomm, 3)))
        pos = nx.spring_layout(G, seed=1001, weight="weight")
        ax = axes[0, c]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.08, width=0.4)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[cmap[m] for m in modid],
                               node_size=110, edgecolors="k", linewidths=0.4)
        ax.set_title(f"{S.LABEL[arch]}: {ncomm} communities | Q={sc['Q']:.3f} | "
                     f"$Q/Q_{{rand}}$={sc['Qnorm']:.2f}", fontsize=10)
        ax.axis("off")
        ax = axes[1, c]
        im = block_panel(ax, W, modid)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{S.LABEL[arch]}: adjacency reordered by community", fontsize=10)
    fig.suptitle("Seed 1001 alone, the package's figure reproduced\n"
                 "the homogeneous block structure is visible, and this is the pair the original analysis used",
                 fontweight="bold", y=1.005, fontsize=11.5)
    S.save(fig, FIGS / "S7_seed1001_communities.png")


# ---------------------------------------------------------------- S8: 1001 against typical seeds
def fig_contrast_seeds():
    homq = {s: SCAL[f"{s}_hom"]["Qnorm"] for s in SC.SEEDS}
    med = min(SC.SEEDS, key=lambda s: abs(homq[s] - np.median(list(homq.values()))))
    low = min(SC.SEEDS, key=lambda s: homq[s])
    picks = [(1001, "the seed the package used"), (med, "a median seed"), (low, "the least modular seed")]
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 9.6))
    for c, (seed, note) in enumerate(picks):
        for r, arch in enumerate(ARCHS):
            W, modid = g(seed, arch, "W"), g(seed, arch, "modid")
            ax = axes[r, c]
            block_panel(ax, W, modid)
            sc = SCAL[f"{seed}_{arch}"]
            ax.set_title(f"seed {seed} {arch}   $Q/Q_{{rand}}$ = {sc['Qnorm']:.2f}", fontsize=10)
            if r == 0:
                ax.text(0.5, 1.22, note, transform=ax.transAxes, ha="center", fontsize=10,
                        fontweight="bold", color=S.ACCENT)
    for r, arch in enumerate(ARCHS):
        axes[r, 0].set_ylabel(S.LABEL[arch], fontsize=11, fontweight="bold")
    fig.suptitle("The same picture at three seeds. Seed 1001's homogeneous network scores highest, but\n"
                 "seed 210's scores lowest of all forty while showing the brightest block: on these dense "
                 "graphs\n$Q/Q_{rand}$ does not track what the block matrices display (see S16)",
                 fontweight="bold", y=1.045, fontsize=11)
    S.save(fig, FIGS / "S8_block_contrast.png")


# ---------------------------------------------------------------- S9: all forty block matrices
def fig_block_montage(key="Qnorm", field="W", modfield="modid", tag="S9", what="binned"):
    """Forty block matrices, each seed's two networks side by side.

    Pairing is the point of the design: the two networks of a seed share initial weights, data order and
    training schedule, so the only thing that differs between two adjacent panels is the membrane time
    constant. Any other arrangement asks the reader to hold one picture in mind while finding its partner
    somewhere else on the page.

    Each panel is scaled to its own 99th percentile because absolute edge MI differs between networks, the
    heterogeneous arm firing more. Brightness is therefore not comparable across panels; the presence and
    sharpness of blocks on the diagonal is, which is what modularity means.
    """
    ncol, nrow = 10, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(16.8, 7.6))
    fig.subplots_adjust(top=0.92, bottom=0.06, hspace=0.45, wspace=0.12)
    for i, seed in enumerate(SC.SEEDS):
        r, c0 = divmod(i, 5)
        for j, arch in enumerate(ARCHS):
            ax = axes[r, c0 * 2 + j]
            block_panel(ax, g(seed, arch, field), g(seed, arch, modfield), lw=0.5)
            ax.grid(False)
            q = SCAL[f"{seed}_{arch}"][key]
            # the panel carries only the statistic; the frame colour and the legend name the network type
            ax.set_title(f"$Q/Q_{{\\mathrm{{rand}}}}$ = {q:.2f}", fontsize=7.5, pad=2.5, color=S.ACCENT,
                         fontweight="normal", loc="center")
            if arch == "het":
                ax.text(1.06, -0.08, f"Seed {seed}", transform=ax.transAxes, ha="center", va="top",
                        fontsize=8.5, fontweight="bold", color=S.ACCENT)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(S.COL[arch])
                sp.set_linewidth(1.6)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="s", ls="", color=S.COL[a], label=S.LABEL[a]) for a in ARCHS],
               loc="lower center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle(f"Community-ordered functional coupling of all forty networks, {what} graph",
                 fontweight="bold", y=0.97, fontsize=11, x=0.5)
    S.save(fig, FIGS / f"{tag}_block_montage.png")


def fig_block_montage_sorted(key="Qnorm", field="W", modfield="modid", tag="S9b", what="binned"):
    """The same forty panels ordered by modularity instead of by seed.

    This arrangement answers a different question. If the two arms differed on this statistic the title
    colours would segregate, cool at one end and warm at the other. Where they interleave, they do not
    differ, which is the result the seed-paired version cannot show at a glance.
    """
    items = sorted([(SCAL[f"{s}_{a}"][key], s, a) for s in SC.SEEDS for a in ARCHS])
    ncol, nrow = 10, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(16.8, 8.2))
    for i, (q, seed, arch) in enumerate(items):
        ax = axes[i // ncol, i % ncol]
        block_panel(ax, g(seed, arch, field), g(seed, arch, modfield), lw=0.5)
        ax.set_title(f"{arch} {seed}\n{q:.2f}", fontsize=7.2, pad=2.5,
                     color=S.COL[arch], fontweight="bold", linespacing=1.15)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="s", ls="", color=S.COL[a], label=S.LABEL[a]) for a in ARCHS],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"The same forty matrices, {what} graph, sorted by $Q/Q_{{rand}}$ from least to most "
                 "modular\nthe two arms interleave across the whole range rather than separating",
                 fontweight="bold", y=1.05, fontsize=11)
    S.save(fig, FIGS / f"{tag}_block_montage_sorted.png")


# ---------------------------------------------------------------- S10: MI matrices and edge statistics
def fig_mi_matrices():
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8))
    for r, arch in enumerate(ARCHS):
        MI, W = g(1001, arch, "MI_obs"), g(1001, arch, "W")
        iu = np.triu_indices(MI.shape[0], 1)
        vmax = np.percentile(MI[iu], 99)
        im = axes[r, 0].imshow(MI, cmap="magma", vmin=0, vmax=vmax)
        axes[r, 0].set_title(f"{S.LABEL[arch]}: observed MI (bits)", fontsize=10)
        plt.colorbar(im, ax=axes[r, 0], fraction=0.046)
        im = axes[r, 1].imshow(W, cmap="magma", vmin=0, vmax=vmax)
        axes[r, 1].set_title(f"{S.LABEL[arch]}: null-subtracted weight", fontsize=10)
        plt.colorbar(im, ax=axes[r, 1], fraction=0.046)
        for a in (axes[r, 0], axes[r, 1]):
            a.set_xlabel("unit")
            a.set_ylabel("unit")

    # edge-weight distribution, pooled over all twenty seeds rather than shown for one
    ax = axes[0, 2]
    for arch in ARCHS:
        v = np.concatenate([g(s, arch, "W")[np.triu_indices(SCAL[f'{s}_{arch}']['n'], 1)] for s in SC.SEEDS])
        ax.hist(v, bins=80, histtype="step", lw=1.8, color=S.COL[arch], density=True, label=S.LABEL[arch])
    ax.set_yscale("log")
    ax.set_xlabel("null-subtracted MI (bits)")
    ax.set_ylabel("density")
    ax.set_title("Edge weights, all twenty seeds pooled", fontsize=10)
    ax.legend()

    # mean edge MI per network, paired
    ax = axes[1, 2]
    v = {a: np.array([[MOD["per_network"][f"{s}_{a}"]["edge_mi_mean"] for s in SC.SEEDS]], float)
         for a in ARCHS}
    S.paired_bars(ax, ["mean edge MI"], v, [MOD["paired"]["edge_mi_mean"]])
    ax.set_ylabel("mean edge MI (bits)")
    ax.set_title("Mean edge MI per network", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Edge statistics: the matrices for seed 1001, the distributions for all twenty",
                 fontweight="bold", y=1.01)
    S.save(fig, FIGS / "S10_mi_matrices.png")


# ---------------------------------------------------------------- S11: the full density sweep
def fig_density_sweep_full():
    """The sweep as a paired within-seed difference, which is the quantity the tests are actually about.

    Overlaying forty trajectories was unreadable, and it was also the wrong picture: in a paired design the
    two arms' absolute levels vary together across seeds, so most of the visible scatter is between-seed
    variation that the paired test already removes. Differencing within a seed cancels it. One line per panel,
    a 95% CI band, a zero reference, and a filled marker wherever the sign-flip test clears 0.05.
    """
    from stats_core import paired_triple
    d = MOD["densities"]
    panels = [("weighted", "Qnorm", "$\\Delta\\;Q/Q_{rand}$", "Normalised modularity, weighted"),
              ("binary", "Qnorm", "$\\Delta\\;Q/Q_{rand}$", "Normalised modularity, binary"),
              ("binary", "sigma", "$\\Delta\\;\\sigma$", "Small-worldness, binary")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    for ax, (wt, key, ylab, title) in zip(axes, panels):
        V = {}
        for arch in ARCHS:
            rows = []
            for sd in SC.SEEDS:
                recs = [r for r in MOD["per_network"][f"{sd}_{arch}"]["sweep"] if r["weighting"] == wt]
                recs.sort(key=lambda r: r["target_density"])
                rows.append([r[key] for r in recs])
            V[arch] = np.array(rows, float)
        D = V["het"] - V["hom"]                       # (n_seeds, n_densities)
        m = D.mean(0)
        h = np.array([S.ci95_mean(D[:, j]) for j in range(D.shape[1])])
        stats = [paired_triple(V["het"][:, j], V["hom"][:, j]) for j in range(D.shape[1])]
        sig = np.array([r["perm_p"] < 0.05 for r in stats])

        ax.axhline(0, color=S.CHANCE, ls=":", lw=1.2)
        ax.fill_between(d, m - h, m + h, color=S.HOM, alpha=0.20, lw=0)
        ax.plot(d, m, color=S.HOM, lw=2.4, zorder=3)
        ax.scatter(np.array(d)[~sig], m[~sig], s=46, facecolors="white", edgecolors=S.HOM,
                   lw=1.8, zorder=4)
        ax.scatter(np.array(d)[sig], m[sig], s=46, color=S.HOM, zorder=4)
        for j, r in enumerate(stats):                 # how many seeds move with the mean
            n_hom = r["n"] - r["wins"]
            ax.annotate(f"{n_hom}/{r['n']}", (d[j], m[j]), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=7.2, color="#3a3a3a")
        ax.set_xlabel("matched edge density")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.margins(y=0.22)

    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=S.HOM, lw=2.4, marker="o", ms=7,
                               label="mean het $-$ hom difference, 95% CI band"),
                        Line2D([], [], color=S.HOM, lw=0, marker="o", ms=7, mfc="white", mew=1.8,
                               label="sign-flip p $\\geq$ 0.05"),
                        Line2D([], [], color="none", label="labels: seeds with hom higher")],
               loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("The matched-density sweep as a within-seed difference "
                 "(below zero means the homogeneous network is higher)", fontweight="bold", y=1.02)
    S.save(fig, FIGS / "S11_density_sweep_full.png")


# ---------------------------------------------------------------- S12: community by class tuning
def fig_tuning(kind="binned", tag="S12", label="binned functional graph"):
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8))
    for r, arch in enumerate(ARCHS):
        mat = tun(1001, arch, kind, "mat")
        sizes = tun(1001, arch, kind, "sizes").astype(int)
        matz = (mat - mat.mean(1, keepdims=True)) / (mat.std(1, keepdims=True) + 1e-9)
        nc = mat.shape[0]
        ylab = [f"c{i} (n={sizes[i]})" for i in range(nc)]
        im = axes[r, 0].imshow(mat, cmap="magma", aspect="auto")
        axes[r, 0].set_title(f"{S.LABEL[arch]}: mean per-unit count", fontsize=10)
        plt.colorbar(im, ax=axes[r, 0], fraction=0.046)
        im = axes[r, 1].imshow(matz, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        axes[r, 1].set_title(f"{S.LABEL[arch]}: z-scored across classes", fontsize=10)
        plt.colorbar(im, ax=axes[r, 1], fraction=0.046)
        for c in (0, 1):
            axes[r, c].set_xticks(range(4))
            axes[r, c].set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=8)
            axes[r, c].set_yticks(range(nc))
            axes[r, c].set_yticklabels(ylab, fontsize=8)
            axes[r, c].set_ylabel("community")
        for i in range(nc):
            for j in range(4):
                axes[r, 1].text(j, i, f"{matz[i, j]:+.1f}", ha="center", va="center", fontsize=7.5,
                                color="k" if abs(matz[i, j]) < 1.0 else "w")

    # across seeds: every community's eta2, pooled, rather than one seed's bars
    ax = axes[0, 2]
    x = np.arange(2)
    w = 0.36
    for i, (fld, lbl, col) in enumerate([("eta2_lang", "language", LANG_C),
                                         ("eta2_par", "parity", PAR_C)]):
        means, errs = [], []
        for arch in ARCHS:
            v = np.array([tun(s, arch, kind, fld).mean() for s in SC.SEEDS])
            means.append(v.mean())
            errs.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, color=col, label=lbl, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([S.LABEL[a] for a in ARCHS])
    ax.set_ylabel(r"mean community $\eta^2$")
    ax.set_title("Which factor, all twenty seeds", fontsize=10)
    ax.legend()

    ax = axes[1, 2]
    for i, (fld, lbl, col) in enumerate([("eta2_4c", "raw", "#444444"),
                                         ("eta2_4c_frac", "fraction of network", "#aaaaaa")]):
        means, errs = [], []
        for arch in ARCHS:
            v = np.array([tun(s, arch, kind, fld).mean() for s in SC.SEEDS])
            means.append(v.mean())
            errs.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, color=col, label=lbl, capsize=3)
    ax.axhline(3 / (SC.N_TRIALS - 1), color=S.CHANCE, ls=":", lw=1.2, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([S.LABEL[a] for a in ARCHS])
    ax.set_ylabel(r"mean 4-class $\eta^2$")
    ax.set_title("Not a global-activity artefact", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle(f"Community by class tuning, {label}: heatmaps for seed 1001, summaries for all twenty",
                 fontweight="bold", y=1.01)
    S.save(fig, FIGS / f"{tag}_tuning_{kind}.png")


# ---------------------------------------------------------------- S14: the cumulative-count graph
def fig_cumulative():
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
    for r, arch in enumerate(ARCHS):
        cW, modid_c = g(1001, arch, "cW"), g(1001, arch, "modid_c")
        sc = SCAL[f"1001_{arch}"]
        im = axes[r, 0].imshow(cW, cmap="magma")
        axes[r, 0].set_title(f"{S.LABEL[arch]}: cumulative-count MI (bits)", fontsize=10)
        axes[r, 0].set_xlabel("unit")
        axes[r, 0].set_ylabel("unit")
        plt.colorbar(im, ax=axes[r, 0], fraction=0.046)
        ax = axes[r, 1]
        G = full_graph(cW)
        ncomm = int(modid_c.max() + 1)
        cmap = plt.cm.tab20(np.linspace(0, 1, max(ncomm, 3)))
        pos = nx.spring_layout(G, seed=1001, weight="weight")
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.06, width=0.4)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[cmap[m] for m in modid_c],
                               node_size=95, edgecolors="k", linewidths=0.3)
        ax.set_title(f"{S.LABEL[arch]}: {ncomm} communities | $Q/Q_{{rand}}$={sc['cQnorm']:.2f}", fontsize=10)
        ax.axis("off")
        im = block_panel(axes[r, 2], cW, modid_c)
        plt.colorbar(im, ax=axes[r, 2], fraction=0.046)
        axes[r, 2].set_title(f"{S.LABEL[arch]}: block matrix", fontsize=10)
    fig.suptitle("The cumulative-count graph at seed 1001, where the package's confirmation came from\n"
                 "across twenty seeds this comparison runs the other way (het 1.251, hom 1.018)",
                 fontweight="bold", y=1.01, fontsize=11.5)
    S.save(fig, FIGS / "S14_cumulative_structure.png")


def fig_cumulative_montage():
    """The same montage for the cumulative-count graph, where the comparison runs the other way."""
    fig_block_montage(key="cQnorm", field="cW", modfield="modid_c", tag="S15",
                      what="cumulative-count")
    fig_block_montage_sorted(key="cQnorm", field="cW", modfield="modid_c", tag="S15b",
                             what="cumulative-count")


# ---------------------------------------------------------------- S16: the statistic against the structure
def fig_block_concentration():
    """Why the headline statistic and the block matrices disagree, and what happens when the null is dropped.

    Panel (a) is the diagnostic: if Q/Q_rand measured what the block matrices show, the points would lie on a
    rising line. They do not, and the homogeneous networks sit high on the concentration axis across the whole
    range of Q/Q_rand. The remaining panels are the comparison the concentration measure supports, on three
    partitions, including one with the community count fixed so it cannot drive the result.
    """
    B = json.load(open(SC.DATA / "block_concentration_n20.json"))
    per, pa = B["per_network"], B["paired"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    ax = axes[0]
    for arch in ARCHS:
        x = [MOD["per_network"][f"{s}_{arch}"]["full"]["Qnorm"] for s in SC.SEEDS]
        y = [per[arch][str(s)]["louvain"] for s in SC.SEEDS]
        ax.scatter(x, y, s=42, color=S.COL[arch], label=S.LABEL[arch], zorder=3,
                   edgecolors="white", lw=0.6)
    ax.axvline(1.0, color=S.CHANCE, ls=":", lw=1.2)
    ax.set_xlabel("$Q/Q_{rand}$, the package's statistic")
    ax.set_ylabel("within / between weight ratio")
    ax.set_title(f"(a) The two disagree (r = {B['corr_ratio_vs_Qnorm']:+.2f})", loc="left", fontsize=10)

    keys = ["louvain", "fixedK", "louvain_cum"]
    names = ["Louvain\n(binned graph)", f"fixed K = {B['k_fixed']}\n(binned graph)",
             "Louvain\n(cumulative graph)"]
    ax = axes[1]
    V = {a: np.array([[per[a][str(s)][k] for s in SC.SEEDS] for k in keys], float) for a in ARCHS}
    S.paired_bars(ax, names, V, [pa[k] for k in keys])
    ax.set_ylabel("within / between weight ratio")
    ax.set_title("(b) The same measure on three partitions", loc="left", fontsize=10)

    S.arch_legend(fig)
    fig.suptitle("Dropping the randomisation null: homogeneous networks concentrate coupling within "
                 "communities, on every partition and both graphs", fontweight="bold", y=1.03)
    S.save(fig, FIGS / "S16_block_concentration.png")


def main():
    fig_seed1001_reproduction()
    fig_contrast_seeds()
    fig_block_montage()
    fig_block_montage_sorted()
    fig_mi_matrices()
    fig_density_sweep_full()
    fig_tuning("binned", "S12", "binned functional graph")
    fig_tuning("cumulative", "S13", "cumulative-count graph")
    fig_cumulative()
    fig_cumulative_montage()
    fig_block_concentration()


if __name__ == "__main__":
    main()

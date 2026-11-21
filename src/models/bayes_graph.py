"""Gambar graf Bayesian Network (bonus spesifikasi). Hanya untuk visualisasi, tidak dipakai oleh fit/predict.

Struktur graf (kelas -> fitur dan parent fitur -> child) mengikuti model TAN:
N. Friedman, D. Geiger, M. Goldszmidt, "Bayesian Network Classifiers", Machine Learning 29(2-3), 131-163, 1997.
https://doi.org/10.1023/A:1007465528199
"""
import numpy as np


def graph_structure(model, feature_names):
    d = len(feature_names)
    parents = [[] if p < 0 else [int(p)] for p in model.parent]

    order, placed = [], set()
    while len(order) < d:
        ready = [i for i in range(d) if i not in placed and all(p in placed for p in parents[i])]
        i = min(ready, key=lambda k: (len(parents[k]), k))
        order.append(i)
        placed.add(i)
    edges = [(p, i, r + 1) for i in range(d) for r, p in enumerate(parents[i])]
    return order, edges


def plot_bayes_graph(model, feature_names, class_name="is_canceled", path=None, title=None):
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    ink, muted, accent, paper = "#1c2430", "#8a94a3", "#1f5f8b", "#fbfbfa"
    order, edges = graph_structure(model, feature_names)
    d = len(order)

    ang = np.pi / 2 - 2 * np.pi * np.arange(d) / d
    R = 1.0
    pos = {f: (R * np.cos(a), R * np.sin(a)) for f, a in zip(order, ang)}
    center = (0.0, 0.0)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor=paper)
    ax.set_facecolor(paper)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.45, 1.55)

    def arrow(p, q, rad, shrink_a, shrink_b, **kw):
        ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", shrinkA=shrink_a, shrinkB=shrink_b,
                                     mutation_scale=13, zorder=1, **kw))

    for f in range(d):
        arrow(center, pos[f], 0.0, 42, 14, arrowstyle="-|>", color=muted, lw=0.7, ls=(0, (3, 3)), alpha=0.7)

    width = {1: 2.4, 2: 1.5, 3: 0.9}
    for p, c, r in edges:
        arrow(pos[p], pos[c], 0.18, 14, 14, arrowstyle="-|>", color=accent, lw=width.get(r, 0.8),
              alpha={1: 0.9, 2: 0.6, 3: 0.4}.get(r, 0.35))

    ax.scatter(*center, s=5200, color=accent, zorder=3)
    ax.text(*center, class_name.replace("_", "\n"), color="white", ha="center", va="center",
            fontsize=11, fontweight="bold", zorder=4)

    for f in range(d):
        x, y = pos[f]
        ax.scatter(x, y, s=520, color=paper, edgecolor=ink, linewidth=1.4, zorder=3)
        ax.text(x, y, str(order.index(f) + 1), ha="center", va="center", fontsize=10, color=ink, zorder=4)
        side = "left" if x > 0.08 else "right" if x < -0.08 else "center"
        ax.text(x * 1.1 + (0.03 if x > 0.08 else -0.03 if x < -0.08 else 0), y * 1.1 + (0.07 * np.sign(y) if abs(x) <= 0.08 else 0),
                feature_names[f].replace("_", " "), ha=side, va="center", fontsize=11, color=ink, zorder=4)

    kind = "TAN" if len(edges) else "Naive Bayes"
    ax.set_title(title or f"Struktur Bayesian Network ({kind}): kelas dan {d} fitur",
                 fontsize=14, color=ink, pad=8, loc="left", x=0.02)
    ax.text(0.02, 0.01,
            f"{len(edges)} edge antarfitur. Garis putus-putus: kelas ke fitur. Panah biru: parent ke child.\n"
            "Angka di node = urutan topologis (parent selalu bernomor lebih kecil).",
            transform=ax.transAxes, fontsize=9, color=muted, va="bottom")
    if path:
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=paper)
        plt.close(fig)
    return fig

"""Figures of the experiments: the quantity of each panel is its title, no y-axis label; every
method has one color, marker and line in every figure, KP drawn thick and on top; posterior mean and 95% band with the
observations and the truth, draws, and metrics against a size (mean +- sd over random replicates, log axes, reference
slopes); one figure per kind (posterior, accuracy metrics, cost)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white", "savefig.facecolor": "white"})
STYLE = {"KP": dict(color="#1c5cab", marker="o", ls="-", lw=2.0, zorder=3),
         "Cholesky": dict(color="#e07b00", marker="s", ls="--", lw=1.3, zorder=2)}
TRUTH = dict(color=INK, lw=1.2, ls=":", label="ground truth")
TRAIN = dict(ls="", marker="o", ms=3.5, color="#d62728", zorder=4, label="observations")


def posterior_plot(ax, xs, truth, X, y, fits, title):
    """fits: {method: (mean, var)} on the plotting inputs xs: each method's mean, the KP mean with its 95% band (one
    legend entry), the observations and the truth; returns the legend entries (bottom_legend)."""
    ax.plot(xs, truth, **TRUTH)
    handles, labels = [], []
    for name, (mean, var) in fits.items():
        st, mean, sd = STYLE[name], np.asarray(mean), np.sqrt(np.clip(np.asarray(var), 0, None))
        line, = ax.plot(xs, mean, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=st["zorder"])
        if name == "KP":
            band = ax.fill_between(xs, mean - 1.96 * sd, mean + 1.96 * sd, color=st["color"], alpha=0.18, lw=0)
            handles.append((band, line)), labels.append("KP mean and 95% interval")
        else:
            handles.append(line), labels.append(f"{name} mean")
    ax.plot(X, y, **TRAIN)
    ax.set_xlabel("$x$")
    ax.set_title(title)
    return entries(ax, handles, labels)


def band_plot(ax, xs, mean, var, label):
    """A mean and its 95% band in ink (a reference distribution), one legend entry; returns its handle."""
    sd = np.sqrt(np.clip(np.asarray(var), 0, None))
    band = ax.fill_between(xs, mean - 1.96 * sd, mean + 1.96 * sd, color=GRID, alpha=0.9, lw=0, zorder=0)
    line, = ax.plot(xs, mean, color=INK2, lw=0.8, zorder=1)
    return (band, line), label


def entries(ax, handles=(), labels=()):
    """Legend entries of ax: the given (mean line, band) entries first, then the labelled artists."""
    h, lab = ax.get_legend_handles_labels()
    return list(handles) + h, list(labels) + lab


def bottom_legend(fig, handles, labels):
    """One legend for the figure, below its panels (a (band, line) entry drawn as the line over the band)."""
    from matplotlib.legend_handler import HandlerTuple
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=len(labels), fontsize=10,
               handler_map={tuple: HandlerTuple(ndivide=1)}, handlelength=2.5)


def draws_plot(ax, xs, draws, title, X=None, y=None, truth=None, band=None):
    """draws: {method: F (m, p)}, one legend entry per method; band: (mean, var, label) of the target distribution,
    its mean and 95% band as one legend entry; returns the legend entries (bottom_legend)."""
    extra = [band_plot(ax, xs, *band)] if band is not None else []
    if truth is not None:
        ax.plot(xs, truth, **TRUTH)
    for name, F in draws.items():
        st = STYLE[name]
        F = np.asarray(F)
        for i in range(F.shape[1]):
            ax.plot(xs, F[:, i], color=st["color"], ls=st["ls"], lw=0.9, alpha=0.8, zorder=st["zorder"],
                    label=f"{name} draws" if i == 0 else None)
    if X is not None:
        ax.plot(X, y, **TRAIN)
    ax.set_xlabel("$x$")
    ax.set_title(title)
    return entries(ax, [e[0] for e in extra], [e[1] for e in extra])


GUIDES = {1: (0, (1, 2)), 2: (0, (4, 2)), 3: (0, (6, 2, 1, 2))}   # dotted O(n), dashed O(n^2), dash-dot O(n^3)


def metric_plot(ax, x, series, xlabel, title, slopes=(), logx=True, logy=True):
    """series: {method: values at x, or (mean, sd) at x}, the quantity named by the title; slopes: reference powers p
    drawn as c x^p through the first point of the first method. The legend lists the methods, then the guides."""
    handles = []
    for name, v in series.items():
        st = STYLE[name]
        mean, sd = (np.asarray(v[0], float), np.asarray(v[1], float)) if isinstance(v, tuple) else (np.asarray(v, float), None)
        handles.append(ax.errorbar(x, mean, yerr=sd, color=st["color"], marker=st["marker"], ls=st["ls"], lw=st["lw"],
                                   ms=5, capsize=2, zorder=st["zorder"], label=name))
    first = next(iter(series.values()))
    x0, v0 = np.asarray(x, float), np.asarray(first[0] if isinstance(first, tuple) else first, float)
    for p in slopes:
        handles += ax.plot(x0, v0[0] * (x0 / x0[0]) ** p, color=INK2, lw=0.9, ls=GUIDES.get(p, ":"),
                           label=f"$O(n)$" if p == 1 else f"$O(n^{p})$")
    ax.set(xlabel=xlabel, xscale="log" if logx else "linear", yscale="log" if logy else "linear")
    ax.set_title(title)
    ax.legend(handles=handles, fontsize=8)


def save_figure(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

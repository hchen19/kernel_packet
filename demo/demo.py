"""Animated GIFs of 1-D kernel packets (``kernel_packets``) and of a 2-D Smolyak sparse grid."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = ROOT / "python" / "src"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

from kernel_packets import KernelPacket, SparseGrid, hc, matern  # noqa: E402

FIGURES = Path(__file__).resolve().parent / "figures"


class KernelPacketDemo:
    """Computes normalized 1-D KP matrices/basis and renders the demo animations."""

    def __init__(self, x, nu=1.5, rho=None, x_eval=None):
        self.x = np.asarray(x, dtype=np.float64).ravel()
        self.nu = float(nu)
        self.k = int(round(2.0 * self.nu + 2.0))
        self.n, self.q = self.x.size, (self.k - 1) // 2
        self.rho = float(np.sqrt(2.0 * self.nu) if rho is None else rho)
        self.x_eval = None if x_eval is None else np.asarray(x_eval, dtype=np.float64).ravel()

        kp = KernelPacket(self.x, k=self.k, rho=self.rho, method="original")
        phi = self._eval_basis(kp, self.x)
        A = np.asarray(kp.solve(phi), dtype=np.float64)
        scale = np.linalg.norm(A, axis=0) * np.where(np.diag(phi) < 0.0, -1.0, 1.0)
        self.A = A / scale
        self.Phi = phi / scale
        self.A[np.abs(self.A) < 1e-12] = 0.0
        self.Phi[np.abs(self.Phi) < 1e-12] = 0.0
        self.K = np.asarray(matern(self.x, self.x, self.k, self.rho), dtype=np.float64)
        self.Phi_eval = None if self.x_eval is None else np.maximum(self._eval_basis(kp, self.x_eval) / scale, 0.0)

    @staticmethod
    def _eval_basis(kp, pts):
        start, vals = kp.basis(pts)
        out = np.zeros((pts.size, kp.n), dtype=np.float64)
        out[np.arange(pts.size)[:, None], start[:, None] + np.arange(vals.shape[1])] = vals
        return out

    def basis_outputs(
        self,
        gif_path=FIGURES / "kernel_packets_basis.gif",
    ):
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        cats = ["left" if i < self.q else ("intermediate" if i < self.n - self.q else "right") for i in range(self.n)]
        cmap = {"left": "#1f77b4", "intermediate": "#d62728", "right": "#2ca02c"}

        fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
        ax.grid(True, alpha=0.3, color="gray", linewidth=0.8)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(1.5)
        ax.set_xlabel(r"$x$", fontsize=28, weight="bold", labelpad=10)
        ax.tick_params(labelsize=20, width=1.5, length=6)
        nu_str = f"{int(round(2 * self.nu))}/2"
        ax.set_title(rf"Kernel Packet Basis ($\nu={nu_str}$, $n={self.n}$)", fontsize=28, weight="bold", pad=20)

        y_max = float(np.max(self.Phi_eval)) * 1.2
        ax.set_xlim(self.x_eval[0], self.x_eval[-1])
        ax.set_ylim(-y_max * 0.1, y_max)
        for xi in self.x[self.q : self.n - self.q]:
            ax.axvline(xi, color="gray", linestyle="--", alpha=0.3, linewidth=1)
        for i, xi in enumerate(self.x):
            ax.scatter([xi], [0.0], s=120, color="#707070", zorder=200, edgecolors="black", linewidths=2)
            ax.text(xi, -y_max * 0.05, rf"$x_{{{i + 1}}}$", fontsize=22, ha="center", va="top", color="#707070", weight="bold")

        lines = [ax.plot(self.x_eval, self.Phi_eval[:, i], color=cmap[cats[i]], visible=False)[0] for i in range(self.n)]
        badge = ax.text(
            0.0, 0.0, "", fontsize=20, ha="center", va="bottom", weight="bold", visible=False,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", linewidth=2, alpha=0.9),
        )

        def support(i):
            if cats[i] == "left":
                return rf"Support: $(-\infty, x_{{{i + self.q + 1}}}]$"
            if cats[i] == "right":
                return rf"Support: $[x_{{{i - self.q + 1}}}, \infty)$"
            return rf"Support: $[x_{{{i - self.q + 1}}}, x_{{{i + self.q + 1}}}]$"

        def update(f):
            if f < self.n:
                for i, ln in enumerate(lines):
                    ln.set_visible(i <= f)
                    ln.set_linewidth(4.0 if i == f else 2.5)
                    ln.set_alpha(1.0 if i == f else 0.3)
                    ln.set_zorder(10 if i == f else 5)
                col = cmap[cats[f]]
                y = self.Phi_eval[:, f]
                tx = float(np.clip(self.x_eval[int(np.argmax(y))], self.x_eval[0] + 0.65, self.x_eval[-1] - 0.65))
                ty = min(float(np.max(y)) + y_max * 0.07, y_max * 0.86)
                badge.set_position((tx, ty))
                badge.set_text(support(f))
                badge.set_color(col)
                badge.get_bbox_patch().set_edgecolor(col)
                badge.set_visible(True)
                ax.legend([lines[f]], [rf"{cats[f].capitalize()} KP: $\phi_{{{f + 1}}}(x)$"],
                          loc="upper center", fontsize=24, framealpha=0.95, edgecolor="black", fancybox=True, shadow=True)
            else:
                badge.set_visible(False)
                for ln in lines:
                    ln.set_visible(True)
                    ln.set_linewidth(3.0)
                    ln.set_alpha(0.8)
                    ln.set_zorder(5)
                if f == self.n:
                    if ax.get_legend():
                        ax.get_legend().set_visible(False)
                else:
                    ax.legend(
                        [lines[0], lines[self.q], lines[self.n - self.q]],
                        ["Left-sided KPs", "Intermediate KPs", "Right-sided KPs"],
                        loc="upper center", fontsize=24, framealpha=0.95, edgecolor="black", fancybox=True, shadow=True,
                    )

        anim = FuncAnimation(fig, update, frames=self.n + 2, interval=800, blit=False, repeat=True)
        plt.tight_layout()
        anim.save(str(gif_path), writer=PillowWriter(fps=1.25), dpi=100)
        plt.close(fig)
        return gif_path

    def curves_outputs(
        self,
        col=None,
        gif_path=FIGURES / "kernel_packets_curves.gif",
    ):
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        col = self.n // 2 if col is None else col
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        K_eval = np.asarray(matern(self.x_eval, self.x, self.k, self.rho), dtype=np.float64)
        a_col = self.A[:, col]
        comps = K_eval * a_col
        phi_curve = self.Phi_eval[:, col]
        comp_y = self.K.diagonal() * a_col
        phi_y = self.Phi[:, col]
        z_max = float(max(np.max(np.abs(phi_curve)), np.max(np.abs(comps)))) * 1.15
        labels = [rf"$A_{{{i + 1},{col + 1}}}k(\cdot, x_{{{i + 1}}})$" for i in range(self.n)]

        fig, ax = plt.subplots(figsize=(14, 9), facecolor="white")
        ax.grid(True, alpha=0.4, color="gray", linewidth=0.8)
        for s in ("left", "bottom"):
            ax.spines[s].set_linewidth(1.5)
        ax.set_xlabel(r"$x$", fontsize=26, weight="bold", labelpad=10)
        ax.tick_params(labelsize=18, width=1.5, length=6)
        ax.set_title(rf"Kernel Packet $\phi_{{{col + 1}}}(\cdot) = \sum_{{i=1}}^{{{self.n}}} A_{{i, {col + 1}}}k(\cdot, x_{{i}})$",
                     fontsize=26, weight="bold", pad=20)
        ax.set_xlim(self.x_eval[0], self.x_eval[-1])
        ax.set_ylim(-0.5, 0.85)
        for i, xi in enumerate(self.x):
            ax.scatter([xi], [0.0], s=120, color="#707070", zorder=200, edgecolors="black", linewidths=2)
            ax.text(xi, -z_max * 0.08, rf"$x_{{{i + 1}}}$", fontsize=24, ha="center", va="top", color="#707070", weight="bold", zorder=200)

        c_lines = [ax.plot(self.x_eval, comps[:, i], color=colors[i], label=labels[i], visible=False)[0] for i in range(self.n)]
        c_vlines = [ax.plot([self.x[i], self.x[i]], [0.0, comp_y[i]], "k--", lw=2.5, alpha=0.7, zorder=50, visible=False)[0] for i in range(self.n)]
        c_dots = [ax.scatter([self.x[i]], [comp_y[i]], s=150, color=colors[i], zorder=100, edgecolors="black", linewidths=2, visible=False) for i in range(self.n)]
        phi_line = ax.plot(self.x_eval, phi_curve, color="black", lw=5, label=rf"$\phi_{{{col + 1}}}(\cdot) = \Phi[:,{col + 1}]$", zorder=20, visible=False)[0]
        p_vlines = [ax.plot([self.x[i], self.x[i]], [0.0, phi_y[i]], "k--", lw=2.5, alpha=0.7, zorder=50, visible=False)[0] for i in range(self.n)]
        p_dots = [ax.scatter([self.x[i]], [phi_y[i]], s=150, color="black", zorder=100, edgecolors="red", linewidths=2, visible=False) for i in range(self.n)]

        c_badge = ax.text(0.0, 0.0, "", fontsize=20, weight="bold", ha="center", zorder=100, visible=False,
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="white", linewidth=2, alpha=0.9))
        s_badge = ax.text(0.0, 0.52, "Σ", fontsize=60, weight="bold", color="black", ha="center", va="center", zorder=100, visible=False,
                          bbox=dict(boxstyle="circle,pad=0.3", facecolor="yellow", edgecolor="black", linewidth=3, alpha=0.85))
        p_badge = ax.text(self.x_eval[int(np.argmax(np.abs(phi_curve)))], float(np.max(phi_curve)) + z_max * 0.1,
                          rf"$\phi_{{{col + 1}}}(\cdot)$", fontsize=26, weight="bold", color="black", ha="center", va="bottom", zorder=100, visible=False,
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", linewidth=3, alpha=0.95))

        def update(f):
            for artist in (*c_vlines, *c_dots, *p_vlines, *p_dots, phi_line, c_badge, s_badge, p_badge):
                artist.set_visible(False)
            if f < self.n:
                for i, ln in enumerate(c_lines):
                    ln.set_visible(i <= f)
                    ln.set_linewidth(4.0 if i == f else 3.0)
                    ln.set_alpha(1.0 if i == f else 0.3)
                    ln.set_zorder(10 if i == f else 5)
                c_vlines[f].set_visible(True)
                c_dots[f].set_visible(True)
                yv = float(comp_y[f])
                lx, ly, va = (
                    (float(self.x[f]) - 1.35, min(yv - 0.04, 0.72), "top") if yv >= 0.55
                    else (float(self.x[f]), yv + 0.08, "bottom") if yv >= 0.0
                    else (float(self.x[f]), max(yv - 0.05, -0.44), "top")
                )
                c_badge.set_position((lx, ly))
                c_badge.set_verticalalignment(va)
                c_badge.set_text(labels[f])
                c_badge.set_color(colors[f])
                c_badge.get_bbox_patch().set_edgecolor(colors[f])
                c_badge.set_visible(True)
                ax.legend(handles=c_lines[: f + 1], loc="upper right", fontsize=20, framealpha=0.95, edgecolor="black", fancybox=True, shadow=True)
            elif f == self.n:
                for ln in c_lines:
                    ln.set_visible(True)
                    ln.set_linewidth(3.0)
                    ln.set_alpha(0.7)
                    ln.set_zorder(10)
                s_badge.set_visible(True)
                ax.legend(handles=c_lines, loc="upper right", fontsize=20, framealpha=0.95, edgecolor="black", fancybox=True, shadow=True)
            else:
                for ln in c_lines:
                    ln.set_visible(True)
                    ln.set_linewidth(2.0)
                    ln.set_alpha(0.15)
                    ln.set_zorder(5)
                for artist in (*p_vlines, *p_dots, phi_line, p_badge):
                    artist.set_visible(True)
                ax.legend(handles=[*c_lines, phi_line], loc="upper right", fontsize=20, framealpha=0.95, edgecolor="black", fancybox=True, shadow=True)

        anim = FuncAnimation(fig, update, frames=self.n + 2, interval=1000, blit=False, repeat=True)
        plt.tight_layout()
        anim.save(str(gif_path), writer=PillowWriter(fps=1), dpi=100)
        plt.close(fig)
        return gif_path

    def matrices_outputs(
        self,
        gif_path=FIGURES / "kernel_packets_matrices.gif",
    ):
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        size = 0.4 * self.n
        total_w = 0.4 + size * 3 + 1.6
        cy = 0.8 + size / 2
        fig, ax = plt.subplots(figsize=(max(12.0, total_w * 1.2), max(5.5, (size + 1.2) * 1.1)), facecolor="white")
        ax.set_xlim(-0.2, total_w + 0.2)
        ax.set_ylim(0.2, size + 1.4)
        ax.axis("off")

        fk, fap = max(9, int(9 * 5 / self.n)), max(9, int(8 * 5 / self.n))
        pos = [(0.2, 0.8, size, size), (0.8 + size, 0.8, size, size), (1.4 + 2 * size, 0.8, size, size)]

        def build_mat(mat, p, title, sub, sub_col, fs, vmax=None):
            vmax = float(np.max(np.abs(mat))) if vmax is None else vmax
            cw, ch = p[2] / self.n, p[3] / self.n
            artists = [
                ax.imshow(mat, cmap="RdBu_r", aspect="equal", extent=[p[0], p[0] + p[2], p[1], p[1] + p[3]], vmin=-vmax, vmax=vmax, zorder=1),
                ax.text(p[0] + p[2] / 2, p[1] + p[3] + 0.3, title, fontsize=22, weight="bold", ha="center", va="bottom"),
                ax.text(p[0] + p[2] / 2, p[1] - 0.2, sub, fontsize=16, ha="center", va="top", style="italic", color=sub_col, weight="bold"),
            ]
            for i in range(self.n):
                for j in range(self.n):
                    v = mat[i, j]
                    artists.append(ax.text(
                        p[0] + (j + 0.5) * cw, p[1] + (self.n - i - 0.5) * ch,
                        "0" if abs(v) < 1e-10 else f"{v:.2f}",
                        ha="center", va="center", fontsize=fs, weight="bold",
                        color="white" if abs(v) > 0.5 * vmax else "black", zorder=15,
                    ))
            return artists

        def build_boxes(mat, p):
            cw, ch = p[2] / self.n, p[3] / self.n
            boxes = []
            for i in range(self.n):
                for j in range(self.n):
                    if abs(mat[i, j]) > 1e-10:
                        r = Rectangle((p[0] + j * cw, p[1] + (self.n - 1 - i) * ch), cw, ch, lw=2.5, edgecolor="yellow", facecolor="none", alpha=0.5, zorder=10)
                        ax.add_patch(r)
                        boxes.append(r)
            return boxes

        k_grp = build_mat(self.K, pos[0], rf"$k(x_{{1:{self.n}}}, x_{{1:{self.n}}})$", f"[{self.n}×{self.n}] Dense", "darkgreen", fk, 1.0)
        mult_txt = ax.text(0.4 + size, cy, r"$\times$", fontsize=40, weight="bold", ha="center", va="center")
        a_grp = build_mat(self.A, pos[1], r"$\mathbf{A}$", f"[{self.n}×{self.n}] Banded (BW={self.q})", "darkblue", fap)
        eq_txt = ax.text(1.0 + 2 * size, cy, "=", fontsize=40, weight="bold", ha="center", va="center")
        p_grp = build_mat(self.Phi, pos[2], r"$\mathbf{\Phi}$", f"[{self.n}×{self.n}] Banded (BW={self.q - 1})", "darkred", fap)
        a_boxes = build_boxes(self.A, pos[1])
        p_boxes = build_boxes(self.Phi, pos[2])

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        for b in a_boxes:
            b.set_alpha(0.9)
        for b in p_boxes:
            b.set_edgecolor("lime")
            b.set_alpha(0.9)

        def update(f):
            for a in k_grp:
                a.set_visible(True)
            mult_txt.set_visible(f >= 1)
            for a in a_grp:
                a.set_visible(f >= 2)
            eq_txt.set_visible(f >= 3)
            for a in p_grp:
                a.set_visible(f >= 4)
            for b in a_boxes:
                b.set_visible(f == 5)
            for b in p_boxes:
                b.set_visible(f == 7)

        anim = FuncAnimation(fig, update, frames=9, interval=600, blit=False, repeat=True)
        anim.save(str(gif_path), writer=PillowWriter(fps=1.67), dpi=100)
        plt.close(fig)
        return gif_path


class SparseGridDemo:
    """Animates hierarchical 2-D Smolyak sparse grid construction and its 1-D KP combination tableau."""

    def __init__(self, eta=5, nu=1.5):
        self.d, self.eta, self.nu = 2, eta, float(nu)
        self.k = int(round(2.0 * self.nu + 2.0))
        self.L = eta - 1
        self.sgs = {e: SparseGrid(d=2, eta=e, k=self.k) for e in range(2, eta + 1)}
        seen = set()
        self.level_pts = {}
        for e in range(2, eta + 1):
            pts = self.sgs[e].X
            keys = [tuple(np.round(p, 12)) for p in pts]
            self.level_pts[e] = np.array([p for p, key in zip(pts, keys) if key not in seen])
            seen.update(keys)

    def outputs(
        self,
        gif_path=FIGURES / "sparse_grid.gif",
    ):
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.2, 6.4), facecolor="white")

        ax_l.set_xlim(0.0, float(self.L))
        ax_l.set_ylim(0.0, float(self.L))
        ax_l.set_aspect("equal")
        ticks = [i + 0.5 for i in range(self.L)]
        ax_l.set_xticks(ticks)
        ax_l.set_xticklabels([str(i + 1) for i in range(self.L)])
        ax_l.set_yticks(ticks)
        ax_l.set_yticklabels([str(i + 1) for i in range(self.L)])
        ax_l.tick_params(labelsize=14)
        ax_l.set_xlabel(r"1-D level $j_1$ along $x_1$", fontsize=17, weight="bold")
        ax_l.set_ylabel(r"1-D level $j_2$ along $x_2$", fontsize=17, weight="bold")

        ax_r.set_xlim(0.0, 1.0)
        ax_r.set_ylim(0.0, 1.0)
        ax_r.set_aspect("equal")
        ax_r.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_r.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_r.tick_params(labelsize=14)
        ax_r.set_xlabel(r"$x_1$", fontsize=17, weight="bold")
        ax_r.set_ylabel(r"$x_2$", fontsize=17, weight="bold")
        ax_r.grid(True, alpha=0.25, color="gray", linewidth=0.7)

        fig.suptitle(rf"Hierarchical Smolyak Sparse Grid ($d=2$, $\eta=2 \to {self.eta}$)", fontsize=22, weight="bold", y=0.965)
        plt.subplots_adjust(left=0.075, right=0.97, top=0.84, bottom=0.19, wspace=0.22)

        lvl_style = {
            2: ("#1f77b4", 100, r"$\eta=2$ (1)"),
            3: ("#ff7f0e", 85, r"$\eta=3$ (+4)"),
            4: ("#2ca02c", 70, r"$\eta=4$ (+12)"),
            5: ("#9467bd", 58, r"$\eta=5$ (+32)"),
        }
        tableau_legend = [
            Patch(facecolor="#eaf3fb", edgecolor="#1f77b4", linewidth=1.6, label=r"$c_{\mathbf{j}}=+1$ ($j_1+j_2=\eta$)"),
            Patch(facecolor="#fdecec", edgecolor="#d62728", linewidth=1.6, label=r"$c_{\mathbf{j}}=-1$ ($j_1+j_2=\eta-1$)"),
            Patch(facecolor="#f5f5f5", edgecolor="#999999", linewidth=1.2, label=r"$c_{\mathbf{j}}=0$ (canceled)"),
        ]
        ax_l.legend(handles=tableau_legend, loc="upper right", fontsize=11.5, framealpha=0.95, edgecolor="#cccccc")

        artists = []

        def update(f):
            for a in artists:
                a.remove()
            artists.clear()

            e_cur = min(2 + f, self.eta)
            final = f == (self.eta - 1)

            for j1 in range(1, self.L + 1):
                for j2 in range(1, self.L + 1):
                    if j1 + j2 > e_cur:
                        continue
                    if j1 + j2 == e_cur:
                        col, fill, lbl_c, lw = "#1f77b4", "#eaf3fb", "+1", 1.8
                    elif j1 + j2 == e_cur - 1:
                        col, fill, lbl_c, lw = "#d62728", "#fdecec", "−1", 1.8
                    else:
                        col, fill, lbl_c, lw = "#999999", "#f5f5f5", "0", 1.1

                    x0, y0, w, h = j1 - 0.88, j2 - 0.94, 0.76, 0.72
                    rect = Rectangle((x0, y0), w, h, facecolor=fill, edgecolor=col, linewidth=lw, zorder=2)
                    ax_l.add_patch(rect)
                    artists.append(rect)

                    u1, u2 = hc(j1), hc(j2)
                    g1, g2 = np.meshgrid(x0 + w * u1, y0 + h * u2, indexing="ij")
                    artists.append(ax_l.vlines(x0 + w * u1, y0, y0 + h, colors=col, alpha=0.32, lw=0.8, zorder=3))
                    artists.append(ax_l.hlines(y0 + h * u2, x0, x0 + w, colors=col, alpha=0.32, lw=0.8, zorder=3))
                    artists.append(ax_l.scatter(g1.ravel(), g2.ravel(), s=16, color=col, zorder=4))
                    artists.append(
                        ax_l.text(
                            j1 - 0.50,
                            j2 - 0.19,
                            f"{lbl_c} ({u1.size}×{u2.size})",
                            ha="center",
                            va="bottom",
                            fontsize=11.5,
                            weight="bold",
                            color=col,
                            zorder=5,
                        )
                    )

            ax_l.set_title(rf"Smolyak Components ($\eta={e_cur}$)", fontsize=17, weight="bold", pad=10)

            r_handles = []
            for e in range(2, e_cur + 1):
                col, sz, lab = lvl_style[e]
                pts = self.level_pts[e]
                sc = ax_r.scatter(pts[:, 0], pts[:, 1], s=sz, color=col, edgecolors="black", lw=0.7, zorder=4 + e, label=lab)
                artists.append(sc)
                r_handles.append(sc)
                if not final and e == e_cur:
                    ring = ax_r.scatter(pts[:, 0], pts[:, 1], s=sz * 2.2, facecolors="none", edgecolors="#222222", lw=1.4, zorder=10)
                    artists.append(ring)

            ax_r.legend(
                handles=r_handles,
                loc="upper center",
                bbox_to_anchor=(0.38, -0.18),
                ncol=4,
                fontsize=11.0,
                framealpha=0.95,
                edgecolor="#cccccc",
                columnspacing=0.65,
                handletextpad=0.3,
            )
            ax_r.set_title(
                rf"Sparse Grid $Z$ ($\eta={e_cur}$, $m={len(self.sgs[e_cur].X)}$ pts)",
                fontsize=17,
                weight="bold",
                pad=10,
            )

        n_frames = self.eta  # e_cur = 2, 3, 4, 5 + final summary frame
        anim = FuncAnimation(fig, update, frames=n_frames, interval=1200, blit=False, repeat=True)
        anim.save(str(gif_path), writer=PillowWriter(fps=0.85), dpi=100)
        plt.close(fig)
        return gif_path


if __name__ == "__main__":
    d11 = KernelPacketDemo(np.linspace(-2.0, 2.0, 11), x_eval=np.linspace(-2.1, 2.1, 400))
    d11.basis_outputs()
    d11.matrices_outputs()
    KernelPacketDemo(np.linspace(-2.0, 2.0, 5), x_eval=np.linspace(-5.0, 5.0, 200)).curves_outputs()
    SparseGridDemo(eta=5).outputs()

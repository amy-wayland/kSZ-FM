"""
Plot C_ell(baryonified) / C_ell(DMO) from the .npz saved by check_bfc_output.py,
in the default plot style (no need to recompute the power spectra).

Usage:
    python plot_cl_ratio.py plots/bfc_check_shell6_nside4096_cl.npz --out plots/bfc_shell6_cl_ratio.pdf
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import apply_style, style_axes, LINE_COLOUR, LABEL_SIZE


def plot_cl_ratio(ell, c_bfc, c_dmo, out, ell_min=10, title=None):
    apply_style()
    good = (ell >= ell_min) & (c_dmo > 0)
    l = ell[good]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(l, c_bfc[good] / c_dmo[good], color=LINE_COLOUR)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xlim(l[0], l[-1])
    ax.set_xlabel(r"$\ell$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"$C_\ell^{\rm bfc} / C_\ell^{\rm DMO}$", fontsize=LABEL_SIZE)
    if title:
        ax.set_title(title)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    d = np.load(a.npz)
    out = a.out or a.npz.replace("_cl.npz", "_cl_ratio.pdf")
    plot_cl_ratio(d["ell"], d["cl_bfc"], d["cl_dmo"], out, title=a.title)
    print("wrote", out)

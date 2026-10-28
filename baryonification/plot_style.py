"""Amy's default plot style. Use apply_style() once, then style_axes(ax) on every axis."""
import matplotlib.pyplot as plt

LINE_COLOUR = "navy"
LABEL_SIZE = 18


def apply_style():
    plt.rcParams.update({
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "font.size": 14,
        "axes.linewidth": 2.0,
    })


def style_axes(ax):
    ax.tick_params(which="major", direction="in", length=5, width=1.2, top=True, right=True)
    ax.tick_params(which="minor", direction="in", length=4, width=1.0, top=True, right=True)

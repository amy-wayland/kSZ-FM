import argparse
import numpy as np

# --------------------------------------------------------------------------
# Plotting colours
# --------------------------------------------------------------------------

PALETTE = ["#0072B2", "#009E73", "#D55E00", "#7A5195"]
INK, MUTED, RULE = "#1A1A1A", "#5A5A55", "#D8D8D4"

# --------------------------------------------------------------------------
# Geometry of the CMASS shell from the FLAMINGO shell attributes
# --------------------------------------------------------------------------

D_C_MPC = 2102.2204637  # comoving distance at z = 0.55
H_LITTLE = 0.681  # D3A cosmology
ARCMIN = np.pi / (180.0 * 60.0)

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def arcmin_to_mpch(theta_arcmin):
    return np.asarray(theta_arcmin) * ARCMIN * D_C_MPC * H_LITTLE


def mpch_to_arcmin(r):
    return np.asarray(r) / (ARCMIN * D_C_MPC * H_LITTLE)

# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cap", action="append", required=True,
                   help="cap_*.npz from 04_stack_cap.py; repeatable")
    p.add_argument("--label", action="append", default=None)
    p.add_argument("--out", default="ksz_datavector")
    p.add_argument("--data", action="append", default=None,
                   help="observational points to overlay: a text/CSV file with "
                        "columns theta_arcmin, T_kSZ, [sigma], one row per "
                        "aperture, '#' for comments. Repeatable. Plotted as "
                        "neutral points with error bars, the usual convention "
                        "for measurements against model curves.")
    p.add_argument("--data-label", action="append", default=None)
    p.add_argument("--yscale", choices=["log", "linear"], default="log")
    p.add_argument("--beam", type=float, default=1.3,
                   help="beam FWHM in arcmin, shaded as the resolution limit")
    p.add_argument("--pixel", type=float, default=0.86,
                   help="map pixel scale in arcmin, for the caption")
    p.add_argument("--ratio-to", type=int, default=None, metavar="N",
                   help="show a ratio panel instead of the null panel: divide "
                        "every profile by the Nth --cap (1-based, so --ratio-to 2 "
                        "makes the second run the reference). Null chi2/dof is "
                        "still printed for each run.")
    args = p.parse_args()

    labels = args.label or []
    while len(labels) < len(args.cap):
        labels.append(f"run {len(labels) + 1}")

    ratio_mode = args.ratio_to is not None
    if ratio_mode and not (1 <= args.ratio_to <= len(args.cap)):
        raise SystemExit(f"--ratio-to {args.ratio_to}: only {len(args.cap)} "
                         "--cap arguments were given (index is 1-based)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "xtick.major.size": 5.0, "ytick.major.size": 5.0,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    })

    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(5.4, 5.8 if ratio_mode else 5.4), dpi=200,
        gridspec_kw=dict(height_ratios=[3, 1.6] if ratio_mode else [3, 1],
                         hspace=0.07))

    runs = []
    for path, label, colour in zip(args.cap, labels, PALETTE):
        d = np.load(path)
        runs.append((label, colour, d))

        ax.errorbar(d["theta_arcmin"], d["T_ksz"], yerr=d["T_err"],
                    color=colour, lw=1.6, marker="o", ms=4.5,
                    mec="white", mew=0.8, elinewidth=1.2, capsize=0,
                    label=label, zorder=3)

        if len(args.cap) == 1 and not args.data:
            ax.text(0.03, 0.95, label, transform=ax.transAxes, color=colour,
                    fontsize=8.5, ha="left", va="top", zorder=4)

        if not ratio_mode:
            axn.plot(d["theta_arcmin"], d["T_null"] / d["T_err"], color=colour,
                     lw=1.2, marker="o", ms=3.5, mec="white", mew=0.7, zorder=3)

    # --- observational points --------------------------------------
    dlabels = args.data_label or []
    for i, path in enumerate(args.data or []):
        arr = np.atleast_2d(np.loadtxt(path, delimiter=None, comments="#",
                                       ndmin=2))
        if arr.shape[1] < 2:
            raise SystemExit(f"{path}: need at least two columns "
                             "(theta_arcmin, T_kSZ)")
        th_d, T_d = arr[:, 0], arr[:, 1]
        e_d = arr[:, 2] if arr.shape[1] > 2 else None
        lab = dlabels[i] if i < len(dlabels) else "measurement"
        ax.errorbar(th_d, T_d, yerr=e_d, fmt="o", ms=4.5, color=INK,
                    mfc="white", mew=1.1, elinewidth=1.0, capsize=2.5,
                    zorder=5, label=lab)

    theta = runs[0][2]["theta_arcmin"]

    # --- main panel ---------------------------------------------------------
    ax.set_xscale("log")
    ax.set_yscale(args.yscale)
    ax.set_ylabel(r"$\hat{T}_{\rm kSZ}(\theta_{\rm d})$ [$\mu$K arcmin$^2$]")
    ax.grid(False)
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.2)
    if len(runs) > 1 or args.data:
        ax.legend(frameon=False, fontsize=8, loc="upper left")

    if args.yscale == "log":
        lo, hi = runs[0][2]["T_ksz"].min(), runs[0][2]["T_ksz"].max()
        ticks = [t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10)
                 if lo / 1.6 <= t <= hi * 1.6]
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xticks(np.round(np.geomspace(theta.min(), theta.max(), 5), 1))
    ax.set_xlim(theta.min() * 0.85, theta.max() * 1.05)
    ax.minorticks_off()
    ax.tick_params(which="major", direction="in", color="black",
                   labelcolor=INK, labelsize=8,
                   left=True, right=True, top=True, bottom=True,
                   labelbottom=False)

    # --- lower panel: ratio, or null ----------------------------------------
    ratios = {}
    if ratio_mode:
        ref_label, _, ref = runs[args.ratio_to - 1]
        for label, _, d in runs:
            if not np.allclose(d["theta_arcmin"], ref["theta_arcmin"]):
                raise SystemExit(f"'{label}' uses a different aperture grid "
                                 f"from the reference '{ref_label}' - rerun "
                                 "04_stack_cap.py with matching --theta-list")

        ref_frac = ref["T_err"] / ref["T_ksz"]
        axn.fill_between(ref["theta_arcmin"], 1 - ref_frac, 1 + ref_frac,
                         color="#EAF0F7", zorder=0, lw=0)
        axn.axhline(1.0, color=INK, lw=0.7, zorder=1)

        for label, colour, d in runs:
            r = d["T_ksz"] / ref["T_ksz"]
            re = r * np.hypot(d["T_err"] / d["T_ksz"], ref_frac)
            ratios[label] = (r, re)
            if d is ref:
                continue
            axn.errorbar(d["theta_arcmin"], r, yerr=re, color=colour,
                         lw=1.4, marker="o", ms=3.8, mec="white", mew=0.7,
                         elinewidth=1.0, capsize=0, zorder=3)

        axn.set_ylabel(f"ratio to\n{ref_label}", fontsize=8.5)
        lo = min(float(r.min()) for r, _ in ratios.values())
        hi = max(float(r.max()) for r, _ in ratios.values())
        pad = 0.08 * (hi - lo)
        axn.set_ylim(lo - pad, hi + pad)
    else:
        axn.axhspan(-1, 1, color="#EAF0F7", zorder=0)
        axn.axhline(0, color=INK, lw=0.7, zorder=1)
        axn.set_ylim(-3.2, 3.2)
        axn.set_yticks([-2, 0, 2])
        axn.set_ylabel("null [$\\sigma$]", fontsize=8.5)

    axn.set_xlabel(r"aperture radius $\theta_{\rm d}$ [arcmin]")
    axn.grid(False)
    for s in ("top", "right", "bottom", "left"):
        axn.spines[s].set_visible(True)
        axn.spines[s].set_color("black")
        axn.spines[s].set_linewidth(1.2)

    axn.set_xscale("log")
    axn.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axn.set_xticks(np.round(np.geomspace(theta.min(), theta.max(), 5), 1))
    axn.set_xlim(theta.min() * 0.85, theta.max() * 1.05)
    axn.minorticks_off()
    axn.tick_params(which="major", direction="in", color="black",
                    labelcolor=INK, labelsize=8,
                    left=True, right=True, top=True, bottom=True)

    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    fig.savefig(f"{args.out}.png", bbox_inches="tight")
    print(f"wrote {args.out}.pdf and {args.out}.png")

    # --- numbers for the caption -------------------------------------------
    for label, _, d in runs:
        chi2 = float(np.sum((d["T_null"] / d["T_err"]) ** 2))
        n = len(d["theta_arcmin"])
        print(f"\n{label}: {int(d['n_obj'])} objects, "
              f"beam {float(d['beam_fwhm_arcmin'])}' FWHM")
        print(f"  S/N per bin: {np.min(d['T_ksz']/d['T_err']):.0f} - "
              f"{np.max(d['T_ksz']/d['T_err']):.0f}")
        print(f"  null chi2/dof = {chi2/n:.2f}, "
              f"max |null|/sigma = {np.max(np.abs(d['T_null']/d['T_err'])):.2f}")
        if ratio_mode:
            r, _ = ratios[label]
            print(f"  ratio to {runs[args.ratio_to - 1][0]}: "
                  f"{r[0]:.3f} at {theta.min():.2f}' -> "
                  f"{r[-1]:.3f} at {theta.max():.2f}'")
        print(f"  {theta.min():.2f}' - {theta.max():.2f}' = "
              f"{arcmin_to_mpch(theta.min()):.2f} - "
              f"{arcmin_to_mpch(theta.max()):.2f} h^-1 Mpc comoving")
    print(f"\n  map pixel scale {args.pixel}' - apertures below "
          f"~{2*args.pixel:.1f}' are pixelisation limited")
    if ratio_mode:
        print("  ratio error bars assume independent runs and are therefore "
              "conservative:\n  the variants share initial conditions, so much "
              "of the uncertainty is common\n  and cancels in the ratio.")


if __name__ == "__main__":
    main()

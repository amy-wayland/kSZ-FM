"""
Validation 1: does the halo and subhalo lightcone reproduce the snapshot mass functions?

For each chosen lightcone shell i (which is filled from snapshot i only):
    n_lc(M)   = N_lightcone(M, shell i) / V_shell,  V_shell = 4pi/3 (r_out^3 - r_in^3)
    n_snap(M) = N_snapshot_i(M) / Lbox^3            (same catalogue, same cuts)
and their ratio should be 1. Centrals are binned in M200c and satellites in bound mass.
"""
import argparse
import sys
import numpy as np
import h5py
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prep_halocone import build_default_param, load_snapshot_halos


def param_from_lightcone(attrs):
    """Rebuild the param object with the same cuts that were used to build the lightcone."""
    param = build_default_param()
    Om, Ob, h0 = attrs["cosmo"]
    param.cosmo.Om, param.cosmo.Ob, param.cosmo.h0 = float(Om), float(Ob), float(h0)
    param.sim.Lbox = float(attrs["Lbox"])
    param.sim.Nmin_per_halo = int(attrs["Nmin_per_halo"])
    param.code.Mhalo_min = float(attrs["Mhalo_min"])
    param.files.halofile_format = attrs["halofile_format"]
    if isinstance(param.files.halofile_format, bytes):
        param.files.halofile_format = param.files.halofile_format.decode()
    if "include_satellites" in attrs:
        param.code.include_satellites = bool(attrs["include_satellites"])
        param.code.Msub_min = float(attrs["Msub_min"])
        param.sim.Nmin_per_subhalo = int(attrs["Nmin_per_subhalo"])
        param.code.require_host_above_cut = bool(attrs.get("require_host_above_cut", False))
    return param


def dn_dlog10m(mass, bins, volume):
    counts, _ = np.histogram(np.log10(mass), bins=bins)
    dlog = np.diff(bins)
    return counts / volume / dlog, counts


plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 16,
    "axes.linewidth": 2.0,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lightcone")
    ap.add_argument("--shells", type=int, nargs="+", default=None,
                    help="shell ids to check (default: ~6 spread over the range)")
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--nside-sky", type=int, default=8)
    ap.add_argument("--out", default="hmf_validation")
    args = ap.parse_args()

    with h5py.File(args.lightcone, "r") as f:
        halos = f["halos"][:]
        shells = f["shells"][:]
        halo_files = [s.decode() for s in f["halo_files"][:]]
        attrs = dict(f.attrs)

    param = param_from_lightcone(attrs)
    Lbox = param.sim.Lbox
    has_sats = "is_central" in halos.dtype.names
    n_shell = len(shells)
    shell_ids = args.shells if args.shells is not None else \
        np.unique(np.linspace(0, n_shell - 1, 6).astype(int)).tolist()

    # host linkage check (whole lightcone)
    if has_sats:
        cen = halos[halos["is_central"] == 1]
        sat = halos[halos["is_central"] == 0]
        key_c = cen["ID"].astype(np.int64) * 100000 + cen["replica"]
        key_s = sat["IDhost"].astype(np.int64) * 100000 + sat["replica"]
        found = np.isin(key_s, key_c)
        print(f"Satellites with host (IDhost, replica) present in lightcone: "
              f"{found.mean()*100:.1f}% of {len(sat)}")
        print("  (misses are hosts below Mhalo_min / Nmin cuts, or hosts across a shell boundary)")

    # per-shell HMF comparison
    mcol_c = "Mvir"                                    # M200c for centrals
    mcol_s = "Mbound" if has_sats else None            # bound mass for satellites
    ncol = len(shell_ids)
    fig, axes = plt.subplots(2, ncol, figsize=(3.4 * ncol, 6), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1]), squeeze=False)
    summary = []

    for j, sid in enumerate(shell_ids):
        sh = shells[shells["shell_id"] == sid][0]
        r_in, r_out = float(sh["lower_com"]), float(sh["upper_com"])
        V_shell = 4.0 / 3.0 * np.pi * (r_out**3 - r_in**3)
        lc = halos[(halos["shell_id"] == sid) & (halos["halo_buffer"] == 0)]
        print(f"\nShell {sid}: z={sh['lower_z']:.3f}-{sh['upper_z']:.3f}, "
              f"r={r_in:.1f}-{r_out:.1f} Mpc/h, V_shell/Lbox^3 = {V_shell/Lbox**3:.3f}, "
              f"snapshot file: {halo_files[sid]}")

        snap = load_snapshot_halos(param, halo_files[sid])

        groups = [("centrals", mcol_c, 1)]
        if has_sats:
            groups.append(("satellites", mcol_s, 0))

        ax, axr = axes[0, j], axes[1, j]
        row = dict(shell=sid, zmid=0.5 * (sh["lower_z"] + sh["upper_z"]), vfrac=V_shell / Lbox**3)
        for (label, mcol, flag), col in zip(groups, ["C0", "C3"]):
            if has_sats:
                m_lc = lc[mcol][lc["is_central"] == flag]
                m_sn = snap[mcol][snap["is_central"] == flag]
            else:
                m_lc, m_sn = lc[mcol], snap[mcol]
            if len(m_sn) == 0:
                continue
            lo = np.log10(m_sn.min())
            bins = np.linspace(lo, max(np.log10(m_sn.max()), lo + 0.5) + 1e-6, args.nbins + 1)
            mid = 0.5 * (bins[1:] + bins[:-1])
            n_lc, c_lc = dn_dlog10m(m_lc, bins, V_shell)
            n_sn, c_sn = dn_dlog10m(m_sn, bins, Lbox**3)
            good = (c_sn > 0) & (c_lc > 0)
            ratio = np.where(good, n_lc / np.where(n_sn > 0, n_sn, 1), np.nan)
            # Poisson errors on both samples
            err = ratio * np.sqrt(1.0 / np.maximum(c_lc, 1) + 1.0 / np.maximum(c_sn, 1))

            ax.plot(mid[c_sn > 0], n_sn[c_sn > 0], "-", color=col, label=f"{label} snapshot")
            ax.plot(mid[c_lc > 0], n_lc[c_lc > 0], "o", ms=3, color=col, label=f"{label} lightcone")
            axr.errorbar(mid[good], ratio[good], err[good], fmt="o", ms=3, color=col)

            tot_ratio = (len(m_lc) / V_shell) / (len(m_sn) / Lbox**3)
            row[label] = tot_ratio
            print(f"  {label:10s}: N_lc={len(m_lc):9d}  N_snap={len(m_sn):9d}  "
                  f"n_lc/n_snap = {tot_ratio:.4f}")

        ax.set_yscale("log")
        ax.set_title(f"shell {sid}, z={row['zmid']:.2f}", fontsize=9)
        ax.tick_params(which='major', direction='in', length=5, width=1.2, top=True, right=True)
        ax.tick_params(which='minor', direction='in', length=3, width=1.0, top=True, right=True)
        axr.axhline(1, color="k", lw=0.8)
        axr.axhspan(0.95, 1.05, color="0.9", zorder=0)
        axr.set_ylim(0.8, 1.2)
        axr.set_xlabel(r"$\log_{10} M\,[M_\odot/h]$  (cen: $M_{200c}$, sat: $M_{\rm bound}$)", fontsize=8)
        axr.tick_params(which='major', direction='in', length=5, width=1.2, top=True, right=True)
        axr.tick_params(which='minor', direction='in', length=3, width=1.0, top=True, right=True)
        if j == 0:
            ax.set_ylabel(r"$dn/d\log_{10}M\ [h^3\,{\rm Mpc}^{-3}]$")
            axr.set_ylabel("lightcone / snapshot")
            ax.legend(fontsize=7)
        summary.append(row)

        # sky completeness of this shell
        nside = args.nside_sky
        pix = hp.vec2pix(nside, lc["x"], lc["y"], lc["z"])
        cmap = np.bincount(pix, minlength=hp.nside2npix(nside)).astype(float)
        frac_empty = np.mean(cmap == 0)
        print(f"  sky: mean {cmap.mean():.1f} halos/pix (nside={nside}), "
              f"empty pixels {frac_empty*100:.2f}%, rms/mean {cmap.std()/max(cmap.mean(),1e-30):.3f} "
              f"(Poisson: {1/np.sqrt(max(cmap.mean(),1e-30)):.3f})")
        hp.mollview(cmap, title=f"shell {sid} halo counts (nside {nside})", unit="N")
        plt.savefig(f"{args.out}_sky_shell{sid:03d}.pdf", bbox_inches="tight")
        plt.close()

    fig.tight_layout()
    fig.savefig(f"{args.out}_hmf.pdf", bbox_inches="tight")

    # integrated ratio vs z
    fig2, ax2 = plt.subplots(figsize=(5, 3.2))
    for label, col in [("centrals", "C0"), ("satellites", "C3")]:
        zz = [r["zmid"] for r in summary if label in r]
        rr = [r[label] for r in summary if label in r]
        if zz:
            ax2.plot(zz, rr, "o-", color=col, label=label)
    ax2.axhline(1, color="k", lw=0.8)
    ax2.set_xlabel("z (shell)")
    ax2.set_ylabel(r"$n_{\rm lc}/n_{\rm snap}$ (all masses)")
    ax2.tick_params(which='major', direction='in', length=5, width=1.2, top=True, right=True)
    ax2.tick_params(which='minor', direction='in', length=3, width=1.0, top=True, right=True)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{args.out}_ratio_vs_z.pdf", bbox_inches="tight")

    print(f"\nwrote {args.out}_hmf.pdf, {args.out}_ratio_vs_z.pdf and {args.out}_sky_shell*.pdf")


if __name__ == "__main__":
    sys.exit(main())

"""
Validation 2: do halo and subhalo positions from the lightcone sit on the structures
of a DMO healpix (mass/density) map covering the same radial range?
"""
import argparse
import itertools
import sys
import numpy as np
import h5py
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


# IO helpers
def read_map(path, dataset=None, nest=False):
    if path.endswith((".fits", ".fits.gz", ".fit")):
        m = hp.read_map(path, nest=nest, dtype=np.float64)
    else:
        with h5py.File(path, "r") as f:
            print("map file attrs:", dict(f.attrs))
            if dataset is None:
                raise ValueError("HDF5 map: pass --dataset (e.g. DarkMatterMass / TotalMass)")
            d = f[dataset]
            print(f"dataset {dataset} attrs:", dict(d.attrs))
            m = d[:].astype(np.float64)
    if nest:
        m = hp.reorder(m, n2r=True)
    return m


def to_delta(m, nside_out=None, is_delta=False, smooth_arcmin=0.0):
    m = np.where(np.isfinite(m) & (m != hp.UNSEEN), m, 0.0)
    if nside_out is not None and nside_out != hp.npix2nside(len(m)):
        m = hp.ud_grade(m, nside_out)  # mean over sub-pixels: fine for delta
    if not is_delta:
        m = m / m.mean() - 1.0
    if smooth_arcmin > 0:
        m = hp.smoothing(m, fwhm=np.radians(smooth_arcmin / 60.0))
    return m


# Geometry
def circle_lonlat(vec, radius_rad, n=60):
    """Points on a small circle of angular radius radius_rad around unit vector vec."""
    vec = vec / np.linalg.norm(vec)
    a = np.array([1.0, 0, 0]) if abs(vec[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(vec, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(vec, e1)
    phi = np.linspace(0, 2 * np.pi, n)
    pts = (np.cos(radius_rad) * vec[:, None]
           + np.sin(radius_rad) * (np.cos(phi) * e1[:, None] + np.sin(phi) * e2[:, None]))
    theta, ph = hp.vec2ang(pts.T)
    return np.degrees(ph), 90.0 - np.degrees(theta)


def signed_permutations():
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            yield perm, signs


# Main
plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "font.size": 16,
    "axes.linewidth": 2.0,
})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lightcone")
    ap.add_argument("--map", required=True)
    ap.add_argument("--dataset", default=None, help="dataset name for HDF5 maps")
    ap.add_argument("--nest", action="store_true", help="map is in NESTED ordering")
    ap.add_argument("--is-delta", action="store_true", help="map is already an overdensity")
    ap.add_argument("--zmin", type=float); ap.add_argument("--zmax", type=float)
    ap.add_argument("--rmin", type=float, help="comoving Mpc/h"); ap.add_argument("--rmax", type=float)
    ap.add_argument("--nside-out", type=int, default=None, help="downgrade map to this nside")
    ap.add_argument("--smooth-arcmin", type=float, default=0.0)
    ap.add_argument("--n-zoom", type=int, default=6, help="number of zoom patches on massive centrals")
    ap.add_argument("--wide-deg", type=float, default=4.0, help="side of the wide-field patches [deg]")
    ap.add_argument("--n-stack", type=int, default=2000, help="max objects per stacked profile")
    ap.add_argument("--out", default="overlay")
    args = ap.parse_args()

    # Halos
    with h5py.File(args.lightcone, "r") as f:
        halos = f["halos"][:]
        attrs = dict(f.attrs)
    print("halo lightcone observer_pos [Mpc/h]:", attrs.get("observer_pos"),
          "<- must be the map's observer")
    Om, Ob, h0 = attrs["cosmo"] if "cosmo" in attrs else (0.306, 0.0486, 0.681)
    cosmo = FlatLambdaCDM(H0=100 * h0, Om0=Om, Ob0=Ob)

    if args.rmin is None:
        if args.zmin is None or args.zmax is None:
            raise ValueError("give either --rmin/--rmax [Mpc/h] or --zmin/--zmax")
        args.rmin = cosmo.comoving_distance(args.zmin).to(u.Mpc).value * h0
        args.rmax = cosmo.comoving_distance(args.zmax).to(u.Mpc).value * h0
    print(f"radial range: {args.rmin:.1f} - {args.rmax:.1f} Mpc/h")

    pos = np.stack([halos["x"], halos["y"], halos["z"]], axis=1) / 1000.0  # Mpc/h
    r = np.linalg.norm(pos, axis=1)
    sel = (r >= args.rmin) & (r < args.rmax) & (halos["halo_buffer"] == 0)
    halos, pos, r = halos[sel], pos[sel], r[sel]
    is_cen = halos["is_central"] == 1 if "is_central" in halos.dtype.names else np.ones(len(halos), bool)
    print(f"objects in slice: {len(halos)} ({is_cen.sum()} centrals, {(~is_cen).sum()} satellites); "
          f"shell_ids contributing: {np.unique(halos['shell_id'])}")

    # Map
    m = read_map(args.map, args.dataset, args.nest)
    delta = to_delta(m, args.nside_out, args.is_delta, args.smooth_arcmin)
    nside = hp.npix2nside(len(delta))
    pix_arcmin = hp.nside2resol(nside, arcmin=True)
    print(f"map nside={nside} (pixel ~{pix_arcmin:.2f} arcmin), delta rms={delta.std():.3f}")

    ipix = hp.vec2pix(nside, pos[:, 0], pos[:, 1], pos[:, 2])
    theta200 = np.full(len(halos), np.nan)
    theta200[is_cen] = (halos["rvir"][is_cen] / 1000.0) / r[is_cen]  # rad

    # Mean delta at halo pixels
    print("\nMean map overdensity at object pixels (random positions: 0):")
    for lab, msk in [("centrals", is_cen), ("satellites", ~is_cen)]:
        if msk.sum() == 0:
            continue
        d = delta[ipix[msk]]
        print(f"  {lab:10s}: <delta> = {d.mean():8.3f}  (S/N {d.mean()/(delta.std()/np.sqrt(len(d))):7.1f}),"
              f"  frac above map median = {np.mean(d > np.median(delta)):.3f} (random: 0.5)")

    # Axis-convention scan
    pc = pos[is_cen]
    scan = []
    for perm, signs in signed_permutations():
        q = pc[:, perm] * np.array(signs)
        scan.append((delta[hp.vec2pix(nside, q[:, 0], q[:, 1], q[:, 2])].mean(), perm, signs))
    scan.sort(key=lambda t: -t[0])
    print("\nAxis-convention scan (stacked delta at centrals; identity should be first, by a lot):")
    for val, perm, signs in scan[:5]:
        tag = "  <-- identity" if perm == (0, 1, 2) and signs == (1, 1, 1) else ""
        print(f"  perm={perm} signs={signs}: {val:8.3f}{tag}")
    print(f"  median over the 48: {np.median([s[0] for s in scan]):.3f}")

    # Stacked radial profiles
    def stack(vecs, rad_max, bins, scale=None):
        prof = np.zeros(len(bins) - 1); cnt = np.zeros(len(bins) - 1)
        for i, v in enumerate(vecs):
            rmax_i = rad_max if scale is None else rad_max * scale[i]
            pl = hp.query_disc(nside, v, rmax_i)
            pv = np.array(hp.pix2vec(nside, pl)).T
            ang = np.arccos(np.clip(pv @ v, -1, 1))
            x = ang if scale is None else ang / scale[i]
            k = np.digitize(x, bins) - 1
            ok = (k >= 0) & (k < len(prof))
            np.add.at(prof, k[ok], delta[pl[ok]]); np.add.at(cnt, k[ok], 1)
        return prof / np.maximum(cnt, 1)

    unit = pos / r[:, None]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
    if is_cen.sum():
        idx = np.flatnonzero(is_cen)
        idx = idx[np.argsort(-halos["Mvir"][idx])][: args.n_stack]
        sc = np.maximum(theta200[idx], np.radians(pix_arcmin / 60.0))
        b = np.linspace(0, 5, 26)
        p = stack(unit[idx], 5.0, b, scale=sc)
        axs[0].plot(0.5 * (b[1:] + b[:-1]), p, "o-")
        axs[0].set_xlabel(r"$\theta/\theta_{200c}$"); axs[0].set_ylabel(r"stacked $\delta$")
        axs[0].set_title(f"{len(idx)} most massive centrals", fontsize=9)
        axs[0].tick_params(which='major', direction='in', length=5, width=1.2, top=True, right=True)
        axs[0].tick_params(which='minor', direction='in', length=3, width=1.0, top=True, right=True)
    if (~is_cen).sum():
        idx = np.flatnonzero(~is_cen)
        idx = idx[np.argsort(-halos["Mbound"][idx])][: args.n_stack] if "Mbound" in halos.dtype.names \
            else idx[: args.n_stack]
        rmax_arc = max(10 * pix_arcmin, 5.0)
        b = np.linspace(0, rmax_arc, 21)
        p = stack(unit[idx], np.radians(rmax_arc / 60.0), np.radians(b / 60.0))
        axs[1].plot(0.5 * (b[1:] + b[:-1]), p, "o-", color="C3")
        axs[1].set_xlabel("θ [arcmin]"); axs[1].set_title(f"{len(idx)} most massive satellites", fontsize=9)
        axs[1].tick_params(which='major', direction='in', length=5, width=1.2, top=True, right=True)
        axs[1].tick_params(which='minor', direction='in', length=3, width=1.0, top=True, right=True)
    for a in axs:
        a.axhline(0, color="k", lw=0.8)
    fig.tight_layout(); fig.savefig(f"{args.out}_stacked_profiles.pdf", bbox_inches="tight"); plt.close(fig)

    # Patch images
    lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
    lat = np.degrees(np.arcsin(pos[:, 2] / r))
    vmin, vmax = np.percentile(delta, [1, 99.7])
    norm_kw = dict(min=max(vmin, -1), max=vmax, cmap="magma", notext=True, cbar=False)

    def overlay(center_lonlat, half_deg):
        c = hp.ang2vec(center_lonlat[0], center_lonlat[1], lonlat=True)
        near = (unit @ c) > np.cos(np.radians(1.5 * half_deg))
        s = near & ~is_cen
        if s.any():
            hp.projscatter(lon[s], lat[s], lonlat=True, s=6, c="cyan", marker="o", lw=0)
        for i in np.flatnonzero(near & is_cen):
            cl, cb = circle_lonlat(unit[i], theta200[i])
            hp.projplot(cl, cb, lonlat=True, color="lime", lw=0.8)

    # Wide fields centred on the two most massive centrals
    order = np.flatnonzero(is_cen)[np.argsort(-halos["Mvir"][is_cen])]
    _, first = np.unique(halos["ID"][order], return_index=True)
    order = order[np.sort(first)]
    fig = plt.figure(figsize=(11, 6.0))
    for k, i in enumerate(order[:2]):
        reso = args.wide_deg * 60.0 / 600
        hp.gnomview(delta, rot=(lon[i], lat[i]), xsize=600, reso=reso, sub=(1, 2, k + 1),
                    margins=(0.01, 0.06, 0.01, 0.08),
                    title=f"{args.wide_deg:.0f} deg field", fig=fig.number, **norm_kw)
        overlay((lon[i], lat[i]), args.wide_deg / 2)
    fig.text(0.5, 0.01, "green circles: centrals (radius $\\theta_{200c}$)   cyan dots: satellites",
             ha="center", fontsize=10)
    fig.savefig(f"{args.out}_wide.pdf", bbox_inches="tight"); plt.close(fig)

    # Zooms on the most massive centrals (side = 8 theta200)
    n = min(args.n_zoom, len(order))
    if n:
        ncol = 3; nrow = int(np.ceil(n / ncol))
        fig = plt.figure(figsize=(4 * ncol, 4.1 * nrow))
        for k, i in enumerate(order[:n]):
            side_arcmin = max(8 * np.degrees(theta200[i]) * 60, 40 * pix_arcmin)
            hp.gnomview(delta, rot=(lon[i], lat[i]), xsize=300, reso=side_arcmin / 300,
                        sub=(nrow, ncol, k + 1), fig=fig.number, margins=(0.02, 0.03, 0.02, 0.13),
                        title="", **norm_kw)
            mant, expo = f"{halos['Mvir'][i]:.1e}".split("e")
            plt.gca().set_title(rf"$M_{{200c}}={mant}\times10^{{{int(expo)}}}\,M_\odot/h$" "\n"
                                rf"$r={r[i]:.0f}\,{{\rm Mpc}}/h$", fontsize=11, pad=4)
            overlay((lon[i], lat[i]), side_arcmin / 120.0)
        fig.text(0.5, 0.0, "green circles: centrals (radius $\\theta_{200c}$)   cyan dots: satellites",
                 ha="center", va="top", fontsize=11)
        fig.savefig(f"{args.out}_zooms.pdf", bbox_inches="tight"); plt.close(fig)

    print(f"\nwrote {args.out}_wide.pdf, {args.out}_zooms.pdf, {args.out}_stacked_profiles.pdf")


if __name__ == "__main__":
    sys.exit(main())

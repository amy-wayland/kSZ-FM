"""
Halo-map position test in bins of mass, for one FLAMINGO map shell.

For every central (or satellite) in the map shell's radial range, in mass bins:
  1. Offset between the catalogue position and the mass centre in the map, found with a
     shrinking-circle centroid (start at max(2 theta200c, 4 pix), shrink by 20% per step
     until 2 pixels, recentring each time). Reported in arcmin, in pixels, in units of
     theta200c and in comoving kpc/h.
  2. Stacked delta profile around the catalogue position (in theta/theta200c and in arcmin).
  3. Timing check: the map records particles when they cross the lightcone, and the
     lightcone places each object at its crossing. Had it been left at its position in the
     snapshot at the shell's lower edge, it would be offset by
         dx = v_pec * (t_cross - t_snap) / a      (comoving)
     We report the predicted |dx_perp| (transverse part, 'pred') and the component of the
     measured offset along that direction ('along'). For correctly placed objects 'along'
     is ~0; if it is ~pred, the objects sit at their snapshot positions instead.

Usage:
    python -u halo_map_offsets.py halo_lightcone_flamingo_crossing_obs750_hostcut.h5 \
        --map maps/L1_m9_DMO_lightcone0_shell2_DarkMatterMass_nside4096.hdf5 --out plots/offsets_shell2

Velocities are SOAP peculiar velocities in km/s (physical).
"""
import argparse

import numpy as np
import h5py
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u

KM_PER_MPC = 3.0857e19
S_PER_GYR = 3.15576e16

COLS = ["x", "y", "z", "vx", "vy", "vz", "Mvir", "rvir", "is_central", "shell_id", "halo_buffer"]


def read_shell_objects(path, rmin, rmax, chunk=5_000_000):
    """Objects with rmin <= r < rmax [Mpc/h], read in chunks to keep memory low."""
    out = []
    with h5py.File(path, "r") as f:
        ds = f["halos"]
        cols = [c for c in COLS if c in ds.dtype.names]
        for s in range(0, ds.shape[0], chunk):
            b = ds[s:s + chunk]
            r = np.sqrt(b["x"] ** 2 + b["y"] ** 2 + b["z"] ** 2) / 1000.0
            m = (r >= rmin) & (r < rmax)
            if "halo_buffer" in cols:
                m &= b["halo_buffer"] == 0
            if m.any():
                out.append(b[m][cols].copy())
        attrs = dict(f.attrs)
        zsnap = f["z_snapshot"][:]
    return np.concatenate(out), attrs, zsnap


def drift_shift(h, zsnap, cosmo, h0):
    """Comoving shift [kpc/h] from the snapshot time to the lightcone-crossing time."""
    pos = np.c_[h["x"], h["y"], h["z"]] / 1000.0                          # Mpc/h
    r = np.linalg.norm(pos, axis=1)
    zgrid = np.linspace(0, 3, 3001)
    rgrid = cosmo.comoving_distance(zgrid).to(u.Mpc).value * h0
    z_cross = np.interp(r, rgrid, zgrid)
    z_snap = zsnap[h["shell_id"]]
    tgrid = cosmo.age(zgrid).to(u.Gyr).value
    dt = np.interp(z_cross, zgrid, tgrid) - np.interp(z_snap, zgrid, tgrid)   # Gyr, >0 if crossing is later
    a = 1.0 / (1.0 + z_cross)
    v = np.c_[h["vx"], h["vy"], h["vz"]]                                    # km/s physical peculiar
    dx = v * (dt * S_PER_GYR)[:, None] / KM_PER_MPC / a[:, None] * h0       # comoving Mpc/h
    return dx * 1000.0, dt                                                  # kpc/h, Gyr


def shrinking_centroid(delta, nside, vec0, r0, rmin):
    """Shrinking-circle centroid of the map mass around vec0. Returns unit vector."""
    c = vec0 / np.linalg.norm(vec0)
    rad = r0
    while True:
        pix = hp.query_disc(nside, c, rad, inclusive=False)
        if len(pix) < 3:
            break
        w = delta[pix] + 1.0                                   # mass in each pixel (relative)
        w = np.clip(w - np.median(w), 0, None)                 # remove local background
        if w.sum() <= 0:
            break
        v = np.array(hp.pix2vec(nside, pix)).T
        c_new = (w[:, None] * v).sum(0)
        c = c_new / np.linalg.norm(c_new)
        if rad <= rmin:
            break
        rad = max(rad * 0.8, rmin)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lightcone")
    ap.add_argument("--map", required=True)
    ap.add_argument("--dataset", default="DarkMatterMass")
    ap.add_argument("--rmin", type=float, default=None, help="Mpc/h (default: from the map file)")
    ap.add_argument("--rmax", type=float, default=None)
    ap.add_argument("--bins", type=float, nargs="+", default=[1e13, 3e13, 1e14, 3e14, 1e16],
                    help="mass bin edges [Msun/h]: M200c for centrals, Mbound for satellites")
    ap.add_argument("--satellites", action="store_true", help="use satellites instead of centrals")
    ap.add_argument("--nmax", type=int, default=2000, help="max objects per bin (random subset)")
    ap.add_argument("--out", default="offsets")
    args = ap.parse_args()

    # ---- map ----
    with h5py.File(args.map, "r") as f:
        ds = f[args.dataset]
        m = ds[:].astype(np.float64)
        a = {**dict(f.attrs), **dict(ds.attrs)}
    nside = hp.npix2nside(len(m))
    delta = m / m.mean() - 1.0
    del m
    pix_arcmin = hp.nside2resol(nside, arcmin=True)

    with h5py.File(args.lightcone, "r") as f:
        Om, Ob, h0 = f.attrs["cosmo"]
    cosmo = FlatLambdaCDM(H0=100 * h0, Om0=Om, Ob0=Ob)
    def radius(key_h, key_noh, given):
        if given is not None:
            return given
        if key_h in a:                      # written by fetch_healpix_shell.py
            return float(a[key_h])
        return float(np.ravel(a[key_noh])[0]) * h0   # FLAMINGO attrs, comoving Mpc (no h)
    rmin = radius("rmin_Mpch", "comoving_inner_radius", args.rmin)
    rmax = radius("rmax_Mpch", "comoving_outer_radius", args.rmax)
    zlo = z_at_value(cosmo.comoving_distance, rmin / h0 * u.Mpc).value if rmin > 0 else 0.0
    zhi = z_at_value(cosmo.comoving_distance, rmax / h0 * u.Mpc).value
    print(f"map nside {nside} ({pix_arcmin:.2f}' pixels), shell r = {rmin:.1f}-{rmax:.1f} Mpc/h, z = {zlo:.3f}-{zhi:.3f}")

    h, attrs, zsnap = read_shell_objects(args.lightcone, rmin, rmax)
    flag = 0 if args.satellites else 1
    if "is_central" in h.dtype.names:
        h = h[h["is_central"] == flag]
    print(f"{len(h)} {'satellites' if args.satellites else 'centrals'} in the shell; "
          "from our shells " + ", ".join(f"{int(k)}: {int(n)}" for k, n in zip(*np.unique(h['shell_id'], return_counts=True))))

    dx, dt = drift_shift(h, zsnap, cosmo, h0)
    pos = np.c_[h["x"], h["y"], h["z"]]
    r_kpc = np.linalg.norm(pos, axis=1)
    unit = pos / r_kpc[:, None]
    # predicted transverse shift (whether or not it was applied)
    dx_perp = dx - (dx * unit).sum(1)[:, None] * unit
    theta200 = np.where(np.isfinite(h["rvir"]), h["rvir"] / r_kpc, np.nan)        # rad (comoving/comoving)

    rng = np.random.default_rng(1)
    edges = np.asarray(args.bins)
    rows, prof_th, prof_am = [], [], []
    tb = np.linspace(0, 3, 16)                       # theta / theta200c
    ab = np.linspace(0, 12 * pix_arcmin, 13)         # arcmin
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero((h["Mvir"] >= lo) & (h["Mvir"] < hi))
        if len(idx) == 0:
            continue
        if len(idx) > args.nmax:
            idx = rng.choice(idx, args.nmax, replace=False)
        off_am, off_t200, off_kpc, along = [], [], [], []
        P_th = np.zeros(len(tb) - 1); N_th = np.zeros(len(tb) - 1)
        P_am = np.zeros(len(ab) - 1); N_am = np.zeros(len(ab) - 1)
        for i in idx:
            v0 = unit[i]
            th = theta200[i] if np.isfinite(theta200[i]) else np.radians(2 * pix_arcmin / 60)
            r0 = max(2 * th, np.radians(4 * pix_arcmin / 60))
            c = shrinking_centroid(delta, nside, v0, r0, np.radians(2 * pix_arcmin / 60))
            ang = np.arccos(np.clip(np.dot(c, v0), -1, 1))
            off_am.append(np.degrees(ang) * 60)
            off_t200.append(ang / th)
            off_kpc.append(ang * r_kpc[i])
            # measured offset vector (tangent plane) projected on the predicted shift direction
            d_meas = (c - v0) * r_kpc[i]
            pp = dx_perp[i]
            n = np.linalg.norm(pp)
            along.append(np.dot(d_meas, pp) / n if n > 0 else np.nan)
            # profiles
            rad = max(3 * th, np.radians(ab[-1] / 60))
            pix = hp.query_disc(nside, v0, rad)
            vv = np.array(hp.pix2vec(nside, pix)).T
            sep = np.arccos(np.clip(vv @ v0, -1, 1))
            k = np.digitize(sep / th, tb) - 1; ok = (k >= 0) & (k < len(tb) - 1)
            np.add.at(P_th, k[ok], delta[pix][ok]); np.add.at(N_th, k[ok], 1)
            k = np.digitize(np.degrees(sep) * 60, ab) - 1; ok = (k >= 0) & (k < len(ab) - 1)
            np.add.at(P_am, k[ok], delta[pix][ok]); np.add.at(N_am, k[ok], 1)
        off_am, off_t200, off_kpc, along = map(np.asarray, (off_am, off_t200, off_kpc, along))
        pred = np.linalg.norm(dx_perp[idx], axis=1)
        row = dict(lo=lo, hi=hi, n=len(idx),
                   theta200_am=np.nanmedian(np.degrees(theta200[idx]) * 60),
                   off_med_am=np.median(off_am), off_med_pix=np.median(off_am) / pix_arcmin,
                   frac_gt_1pix=np.mean(off_am > pix_arcmin), off_med_t200=np.nanmedian(off_t200),
                   off_med_kpc=np.median(off_kpc), pred_med_kpc=np.median(pred),
                   along_med_kpc=np.nanmedian(along))
        rows.append(row)
        prof_th.append((lo, hi, P_th / np.maximum(N_th, 1)))
        prof_am.append((lo, hi, P_am / np.maximum(N_am, 1)))

    tag = "satellites (Mbound)" if args.satellites else "centrals (M200c)"
    print(f"\n{tag}; pixel = {pix_arcmin:.2f}'")
    hdr = ("   mass bin [Msun/h]       N  th200[']  off[']  off[pix]  >1pix  off/th200  off[kpc/h]  pred[kpc/h]  along[kpc/h]")
    print(hdr)
    for r in rows:
        print(f"{r['lo']:9.1e}-{r['hi']:9.1e} {r['n']:5d} {r['theta200_am']:8.2f} {r['off_med_am']:7.2f} "
              f"{r['off_med_pix']:8.2f} {r['frac_gt_1pix']:6.2f} {r['off_med_t200']:9.3f} {r['off_med_kpc']:10.1f} "
              f"{r['pred_med_kpc']:11.1f} {r['along_med_kpc']:12.1f}")
    print("off = median offset catalogue position -> map mass centre; pred = median transverse shift a\n"
          "snapshot-based placement would introduce; along = median component of the measured offset\n"
          "along it (~0 for objects placed at their lightcone crossing).")

    np.savetxt(args.out + "_summary.txt",
               np.array([[r[k] for k in ("lo", "hi", "n", "theta200_am", "off_med_am", "off_med_pix",
                                          "frac_gt_1pix", "off_med_t200", "off_med_kpc", "pred_med_kpc",
                                          "along_med_kpc")] for r in rows]),
               header="lo hi n theta200_arcmin off_med_arcmin off_med_pix frac_gt_1pix off_med_t200 "
                      "off_med_kpch pred_med_kpch along_med_kpch", fmt="%.4g")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lo, hi, p in prof_th:
        ax[0].plot(0.5 * (tb[1:] + tb[:-1]), p / p[0], "o-", ms=3, label=f"{lo:.0e}-{hi:.0e}")
    for lo, hi, p in prof_am:
        ax[1].plot(0.5 * (ab[1:] + ab[:-1]), p / p[0], "o-", ms=3)
    ax[0].set_xlabel(r"$\theta/\theta_{200c}$"); ax[1].set_xlabel("separation [arcmin]")
    ax[0].set_ylabel(r"$\langle\delta\rangle / \langle\delta\rangle_{\rm first\ bin}$")
    for x in ax:
        x.set_yscale("log"); x.axhline(1, color="0.7", lw=0.8)
    ax[1].axvline(pix_arcmin, color="0.6", ls=":", label="1 pixel"); ax[1].legend(fontsize=8)
    ax[0].legend(title=r"$M\ [M_\odot/h]$", fontsize=8)
    fig.suptitle(f"z = {zlo:.2f}-{zhi:.2f}, {tag}")
    fig.tight_layout()
    fig.savefig(args.out + "_profiles.pdf")
    print(f"\nwrote {args.out}_summary.txt and {args.out}_profiles.pdf")


if __name__ == "__main__":
    main()

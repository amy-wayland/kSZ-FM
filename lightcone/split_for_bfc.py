"""
Split the halo lightcone into one file per FLAMINGO map shell, in the format read by the
baryonification code (euclid_fs2 branch, halolc_format = 'FLAMINGO').

Why this is needed
  - The FLAMINGO HEALPix map shell i runs between consecutive snapshot redshifts,
    z_i < z < z_{i+1} (z = 0.05 i for lightcone0, e.g. shell 6 = 0.30-0.35), and the
    crossing-based lightcone uses the same shells. The baryonification reader pairs halo
    file i with map shell i, so each shell's centrals are written to their own file
    (objects are assigned by comoving distance, which for this lightcone reproduces its
    shell_id).
  - The reader expects one HDF5 file per shell with datasets 'shells' (one row:
    shell_id, lower_z, upper_z, lower_com, upper_com, shell_com [Mpc/h]) and 'halos'
    (x, y, z relative to the observer and rvir in comoving kpc/h, Mvir = M200c [Msun/h],
    cvir). It treats every object as a host halo (IDhost = -1), so only centrals are
    written (satellites have no R200c / c200c).

Usage
    python -u split_for_bfc.py halo_lightcone_flamingo_crossing_obs750_hostcut.h5 \
        --outdir bfc_halos --shells 0 39
    # optional: take the shell radii from the map files themselves
    #   --map-pattern "maps/lightcone0.shell_{shell_id}.0.hdf5"

Then, in the baryonification run script:
    par.files.halolc_format = 'FLAMINGO'
    par.files.halolc_in     = 'bfc_halos/halo_lightcone_shell_{shell_id}.h5'
"""
import argparse
import os

import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from numpy.lib.recfunctions import repack_fields

SHELL_DTYPE = np.dtype([("shell_id", np.int32), ("lower_z", np.float32), ("upper_z", np.float32),
                        ("lower_com", np.float32), ("upper_com", np.float32), ("shell_com", np.float32)])
KEEP = ["ID", "IDhost", "Mvir", "Nvir", "x", "y", "z", "vx", "vy", "vz", "rvir", "cvir", "Mbound",
        "is_central", "shell_id", "replica"]


def map_shell_radii(i, dz, cosmo, h0, map_pattern):
    """Comoving inner/outer radius [Mpc/h] of map shell i, from the map file if given."""
    zlo, zhi = i * dz, (i + 1) * dz
    if map_pattern:
        with h5py.File(map_pattern.format(shell_id=i), "r") as f:
            # stored in comoving Mpc (no h); look in the file and in the first dataset
            a = dict(f.attrs)
            for k in f:
                a.update(dict(f[k].attrs))
            r_in = float(np.ravel(a["comoving_inner_radius"])[0]) * h0
            r_out = float(np.ravel(a["comoving_outer_radius"])[0]) * h0
    else:
        r_in = cosmo.comoving_distance(zlo).to(u.Mpc).value * h0
        r_out = cosmo.comoving_distance(zhi).to(u.Mpc).value * h0
    r_mid = cosmo.comoving_distance(0.5 * (zlo + zhi)).to(u.Mpc).value * h0
    return zlo, zhi, r_in, r_out, r_mid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lightcone")
    ap.add_argument("--outdir", default="bfc_halos")
    ap.add_argument("--shells", type=int, nargs=2, default=[0, 39], metavar=("FIRST", "LAST"),
                    help="map shells to write, inclusive (39 = 1.95 < z < 2.00, the last one fully covered)")
    ap.add_argument("--dz", type=float, default=0.05, help="map shell width in z")
    ap.add_argument("--map-pattern", default=None,
                    help='map file path with "{shell_id}" placeholder, to read the shell radii from the maps')
    ap.add_argument("--chunk", type=int, default=5_000_000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with h5py.File(args.lightcone, "r") as f:
        Om, Ob, h0 = f.attrs["cosmo"]
        halos = f["halos"]
        n = halos.shape[0]
        names = halos.dtype.names
        keep = [k for k in KEEP if k in names]
        out_dtype = repack_fields(np.zeros(0, dtype=halos.dtype)[keep]).dtype
        cosmo = FlatLambdaCDM(H0=100 * h0, Om0=Om, Ob0=Ob)

        ids = list(range(args.shells[0], args.shells[1] + 1))
        info = {i: map_shell_radii(i, args.dz, cosmo, h0, args.map_pattern) for i in ids}
        edges = np.array([[info[i][2], info[i][3]] for i in ids])
        pieces = {i: [] for i in ids}

        for s in range(0, n, args.chunk):
            blk = halos[s:s + args.chunk]
            sel = np.ones(len(blk), bool)
            if "is_central" in names:
                sel &= blk["is_central"] == 1
            if "halo_buffer" in names:
                sel &= blk["halo_buffer"] == 0
            blk = blk[sel]
            r = np.sqrt(blk["x"] ** 2 + blk["y"] ** 2 + blk["z"] ** 2) / 1000.0   # kpc/h -> Mpc/h
            for k, i in enumerate(ids):
                m = (r >= edges[k, 0]) & (r < edges[k, 1])
                if m.any():
                    pieces[i].append(repack_fields(blk[m][keep]))
            print(f"read {min(s + args.chunk, n)}/{n} rows")

    for i in ids:
        zlo, zhi, r_in, r_out, r_mid = info[i]
        h = np.concatenate(pieces[i]) if pieces[i] else np.zeros(0, dtype=out_dtype)
        shell = np.array([(i, zlo, zhi, r_in, r_out, r_mid)], dtype=SHELL_DTYPE)
        out = os.path.join(args.outdir, f"halo_lightcone_shell_{i}.h5")
        with h5py.File(out, "w") as g:
            g.create_dataset("shells", data=shell)
            g.create_dataset("halos", data=h, compression="gzip")
            g.attrs["source"] = os.path.basename(args.lightcone)
            g.attrs["note"] = "centrals only, regrouped into FLAMINGO map shell edges"
        src = ""
        if len(h) and "shell_id" in h.dtype.names:
            u_, c_ = np.unique(h["shell_id"], return_counts=True)
            src = "  from our shells " + ", ".join(f"{a}:{b}" for a, b in zip(u_, c_))
        print(f"map shell {i:2d}: z {zlo:.2f}-{zhi:.2f}, r {r_in:.1f}-{r_out:.1f} Mpc/h, "
              f"{len(h)} centrals{src}")


if __name__ == "__main__":
    main()

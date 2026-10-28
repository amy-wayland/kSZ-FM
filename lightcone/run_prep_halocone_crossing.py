"""
Build the crossing-based FLAMINGO L1_m9_DMO halo + subhalo lightcone, one shell per
FLAMINGO map shell (shell i: 0.05 i < z < 0.05 (i+1), i = 0..39).

Needs soap_subset files with TrackId and Tracks/ (run add_track_info.py first), and
flamingo_halo_files.txt from fetch_flamingo_soap.py.

Usage:
    python -u run_prep_halocone_crossing.py --out $X/halo_lightcone_flamingo_crossing_obs750_hostcut.h5
    python -u run_prep_halocone_crossing.py --shells 2 2 --out test_crossing_shell2.h5   # quick test
"""
import argparse
import os

import numpy as np
import h5py

from prep_halocone_crossing import build_default_param, prep_lightcone_halos_crossing

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="halo_lightcone_flamingo_crossing_obs750_hostcut.h5")
ap.add_argument("--shells", type=int, nargs=2, default=[0, 39], metavar=("FIRST", "LAST"))
ap.add_argument("--dz", type=float, default=0.05)
ap.add_argument("--max-repli", type=int, default=9)
args = ap.parse_args()

param = build_default_param()
param.cosmo.Om, param.cosmo.Ob, param.cosmo.h0 = 0.306, 0.0486, 0.681
param.cosmo.s8, param.cosmo.ns = 0.807, 0.967
param.sim.Lbox = 681.0000003288388
param.sim.Nmin_per_halo = 100
param.code.halo_lc_buffer = 1e-9
param.files.halofile_format = "FLAMINGO-SOAP-HBT"
param.code.Mhalo_min = 8e12
param.code.include_satellites = True
param.code.Msub_min = 1e12
param.sim.Nmin_per_subhalo = 100
param.code.require_host_above_cut = True
param.code.observer_pos = tuple(750.0 * param.cosmo.h0 for _ in range(3))   # lightcone0

snapnum = np.loadtxt("flamingo_halo_files.txt", dtype=int, usecols=0)
paths = np.loadtxt("flamingo_halo_files.txt", dtype=str, usecols=1)
file_of = dict(zip(snapnum, paths))
z_of = {}
for s, p in file_of.items():
    with h5py.File(p, "r") as f:
        z_of[s] = float(f["Header"].attrs["Redshift"])
snap_at = lambda z: min(z_of, key=lambda s: abs(z_of[s] - z))

ids = list(range(args.shells[0], args.shells[1] + 1))
pairs, shells_z, z_in = [], [], []
for i in ids:
    zlo, zhi = i * args.dz, (i + 1) * args.dz
    s_in, s_out = snap_at(zlo), snap_at(zhi)
    assert abs(z_of[s_in] - zlo) < 1e-3 and abs(z_of[s_out] - zhi) < 1e-3, \
        f"no snapshots at the edges of shell {i}: {z_of[s_in]:.4f}, {z_of[s_out]:.4f}"
    pairs.append((file_of[s_in], file_of[s_out]))
    shells_z.append((zlo, zhi))
    z_in.append(z_of[s_in])

halo_data, shell_data = prep_lightcone_halos_crossing(param, pairs, shells_z, shell_ids=ids,
                                                      max_repli=args.max_repli)

# the validation scripts index halo_files / z_snapshot by shell_id -> pad up to the first id
n_pad = ids[0]
inner_files = [""] * n_pad + [p[0] for p in pairs]
outer_files = [""] * n_pad + [p[1] for p in pairs]
z_inner = [np.nan] * n_pad + z_in

tmp = args.out + ".tmp"
with h5py.File(tmp, "w") as f:
    f.create_dataset("shells", data=shell_data)
    f.create_dataset("halos", data=halo_data, compression="gzip")
    f.create_dataset("halo_files", data=np.array(inner_files, dtype="S"))       # selection snapshot
    f.create_dataset("halo_files_outer", data=np.array(outer_files, dtype="S"))
    f.create_dataset("z_snapshot", data=np.asarray(z_inner, dtype=float))
    f.attrs["placement"] = "crossing"
    f.attrs["n_snapshots"] = len(shell_data)
    f.attrs["n_halos"] = len(halo_data)
    f.attrs["Lbox"] = param.sim.Lbox
    f.attrs["Nmin_per_halo"] = param.sim.Nmin_per_halo
    f.attrs["Nmin_per_subhalo"] = param.sim.Nmin_per_subhalo
    f.attrs["Mhalo_min"] = param.code.Mhalo_min
    f.attrs["Msub_min"] = param.code.Msub_min
    f.attrs["include_satellites"] = param.code.include_satellites
    f.attrs["require_host_above_cut"] = param.code.require_host_above_cut
    f.attrs["halo_lc_buffer"] = param.code.halo_lc_buffer
    f.attrs["max_repli"] = args.max_repli
    f.attrs["halofile_format"] = param.files.halofile_format
    f.attrs["observer_pos"] = np.asarray(param.code.observer_pos, dtype=float)
    f.attrs["cosmo"] = np.array([param.cosmo.Om, param.cosmo.Ob, param.cosmo.h0])
os.replace(tmp, args.out)
print("wrote", args.out, "n_halos =", len(halo_data))

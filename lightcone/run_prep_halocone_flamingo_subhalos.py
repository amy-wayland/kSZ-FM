"""
Build the FLAMINGO L1_m9_DMO halo + subhalo lightcone (0 < z < 2).

Centrals  : M200c  >= Mhalo_min and N200c  >= Nmin_per_halo
Satellites: Mbound >= Msub_min  and Nbound >= Nmin_per_subhalo  (include_satellites = True)

Output columns: ID, IDhost, Mvir, Nvir, x, y, z, vx, vy, vz, rvir, cvir, Mbound,
is_central, shell_id, halo_buffer, replica. For satellites Mvir is the bound mass,
rvir/cvir are NaN (no SO quantities) and IDhost is the ID of their top-level central;
(IDhost, replica) identifies the host copy in the lightcone.

Requirements:
  - prep_halocone.py, and the project's params.py and constants.py
  - flamingo_shells_z.txt and flamingo_halo_files.txt, written by fetch_flamingo_soap.py
"""
import os

import numpy as np
import h5py

from prep_halocone import build_default_param, prep_lightcone_halos

OUT = "halo_lightcone_flamingo_subhalos_obs750.h5"  # output file
MAX_REPLI = 9

param = build_default_param()

param.cosmo.Om = 0.306
param.cosmo.Ob = 0.0486
param.cosmo.h0 = 0.681
param.cosmo.s8 = 0.807
param.cosmo.ns = 0.967

param.sim.Lbox = 681.0000003288388  # 1000 Mpc (comoving, h-free) * h -- FLAMINGO L1000N1800 box
param.sim.Nmin_per_halo = 100
param.code.halo_lc_buffer = 1e-9
param.files.halofile_format = "FLAMINGO-SOAP-HBT"

# mass cuts
param.code.Mhalo_min = 8e12          # centrals, M200c [Msun/h]
param.code.include_satellites = True
param.code.Msub_min = 1e12           # satellites, bound mass [Msun/h]
param.sim.Nmin_per_subhalo = 100     # satellites, bound DM particles

# FLAMINGO's lightcone0 maps (L1_m9_DMO) have their observer 
# at (750,750,750) comoving Mpc (h-free) -> 750*h Mpc/h.
param.code.observer_pos = tuple(750.0 * param.cosmo.h0 for _ in range(3))
print("observer_pos [Mpc/h]:", param.code.observer_pos)

# both files are written by fetch_flamingo_soap.py
snapnum, z_lower, z_upper, z_snapshot = np.loadtxt("flamingo_shells_z.txt", unpack=True)
halo_files = np.loadtxt("flamingo_halo_files.txt", dtype=str, usecols=1)

zmax = 2.00
keep = z_snapshot <= zmax
halo_files = halo_files[keep].tolist()
shells_z = np.column_stack([z_lower[keep], z_upper[keep]])
z_snap = z_snapshot[keep]
print(f"{keep.sum()} snapshots kept, z_snapshot in "
      f"[{z_snapshot[keep].min():.3f}, {z_snapshot[keep].max():.3f}]")

# ascending snapshot number = descending redshift
# shell_id ordering (z=0 first)
halo_files = halo_files[::-1]
shells_z = shells_z[::-1]
z_snap = z_snap[::-1]

halo_data, shell_data = prep_lightcone_halos(param, halo_files, shells_z, max_repli=MAX_REPLI)

tmp = OUT + ".tmp"
with h5py.File(tmp, "w") as f:
    f.create_dataset("shells", data=shell_data, compression="gzip")
    f.create_dataset("halos", data=halo_data, compression="gzip")
    f.create_dataset("halo_files", data=np.array(halo_files, dtype="S"))
    f.create_dataset("z_snapshot", data=np.asarray(z_snap, dtype=float))

    f.attrs["n_snapshots"] = len(shell_data)
    f.attrs["n_halos"] = len(halo_data)
    f.attrs["Lbox"] = param.sim.Lbox
    f.attrs["Nmin_per_halo"] = param.sim.Nmin_per_halo
    f.attrs["Nmin_per_subhalo"] = param.sim.Nmin_per_subhalo
    f.attrs["Mhalo_min"] = param.code.Mhalo_min
    f.attrs["Msub_min"] = param.code.Msub_min
    f.attrs["include_satellites"] = param.code.include_satellites
    f.attrs["halo_lc_buffer"] = param.code.halo_lc_buffer
    f.attrs["max_repli"] = MAX_REPLI
    f.attrs["halofile_format"] = param.files.halofile_format
    f.attrs["observer_pos"] = np.asarray(param.code.observer_pos, dtype=float)  # comoving Mpc/h
    f.attrs["cosmo"] = np.array([param.cosmo.Om, param.cosmo.Ob, param.cosmo.h0])

os.replace(tmp, OUT)
print("wrote", OUT, "n_halos =", len(halo_data))
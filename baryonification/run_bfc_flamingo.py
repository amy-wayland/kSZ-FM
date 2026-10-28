"""
Shell baryonification of the FLAMINGO L1_m9_DMO lightcone maps
(baryonification code, euclid_fs2 branch, FLAMINGO format).

Inputs
  --halos : per-map-shell halo files written by split_for_bfc.py, with "{shell_id}" placeholder
  --maps  : FLAMINGO DarkMatterMass maps written by fetch_healpix_shell.py, with "{shell_id}" placeholder

The code uses an MPI pool (1 master + workers), so it must be launched with mpiexec and at
least 2 processes, e.g.
    mpiexec -n 9 python -u run_bfc_flamingo.py --first 6 --last 6
Shells --first..--last are inclusive here (the code's max_shell is exclusive; handled below).
"""
import argparse
import os
from time import time

import baryonification as bfc

# Baryonic parameters fitted to FLAMINGO in Schneider et al. 2025 (arXiv:2507.07892),
# as used in Jozef's CosmoGrid run script (examples/run_shell_bfc_cosmogrid_euclidFS2_branch.py)
MODELS = {
    "fiducial": dict(Mc=10**13.08268916, thco=0.31867105, mu=0.553033, delta=5.76996342,
                     eta=0.0378558, deta=0.22472725, Nstar=0.02718398, ciga=0.1037475),
    "fgas_8sigma": dict(Mc=10**13.45878413, thco=0.31916443, mu=0.51973802, delta=4.71687645,
                        eta=0.06044311, deta=0.23963549, Nstar=0.02687355, ciga=0.25362647),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=6, help="first map shell (inclusive)")
    ap.add_argument("--last", type=int, default=6, help="last map shell (inclusive)")
    ap.add_argument("--halos", default="bfc_halos/halo_lightcone_shell_{shell_id}.h5")
    ap.add_argument("--maps", default="maps/L1_m9_DMO_lightcone0_shell{shell_id}_DarkMatterMass_nside4096.hdf5")
    ap.add_argument("--map-field", default="DarkMatterMass")
    ap.add_argument("--nside", type=int, default=1024, help="working resolution (input maps are degraded to this)")
    ap.add_argument("--nside-out", type=int, default=1024)
    ap.add_argument("--model", default="fiducial", choices=sorted(MODELS))
    ap.add_argument("--mhalo-min", type=float, default=1e13, help="M200c cut used for displacement [Msun/h]")
    ap.add_argument("--transfct", required=True, help="path to baryonification/files/CDM_PLANCK_tk.dat in the repo")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tmp", default="tmp_bfc")
    ap.add_argument("--ncpu", type=int, default=int(os.environ.get("BFC_NCPU", 8)))
    args = ap.parse_args()

    par = bfc.par()

    # FLAMINGO L1_m9 cosmology
    par.cosmo.Om, par.cosmo.Ob, par.cosmo.h0 = 0.306, 0.0486, 0.681
    par.cosmo.s8, par.cosmo.ns = 0.807, 0.967

    # model options as in Jozef's CosmoGrid run
    par.code.Mhalo_min = args.mhalo_min
    par.code.beta_model = 1
    par.code.multicomp = True          # separate gas / dm / star output maps
    par.code.eps1 = 0.5
    par.code.q1 = 0.25
    par.code.q2 = 0.7
    par.code.halo_excl = 0.4

    for k, v in MODELS[args.model].items():
        setattr(par.baryon, k, v)

    par.shell.min_shell = args.first
    par.shell.max_shell = args.last + 1   # exclusive in the code
    par.shell.nside = args.nside
    par.shell.nside_out = args.nside_out
    par.shell.N_cpu = args.ncpu

    out = args.out or f"bfc_flamingo_{args.model}_shells{args.first}-{args.last}_nside{args.nside_out}.h5"
    os.makedirs(args.tmp, exist_ok=True)
    par.files.halolc_format = "FLAMINGO"
    par.files.halolc_in = args.halos
    par.files.shellfile_format = "FLAMINGO"
    par.files.shellfile_in = args.maps
    par.files.flamingo_map_field = args.map_field
    par.files.shellfile_out = out
    par.files.transfct = args.transfct
    par.files.cosmofct = os.path.join(args.tmp, "cosmofct.dat")
    par.files.tmp_files = args.tmp

    t0 = time()
    bfc.ShellDisplacer(par).perform_shell_displacement()
    print(f"done in {time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()

"""
Step 1 of the baryonification setup: does the package import and compute profiles?
(The first README example, saving the figure instead of showing it.)

Usage (from anywhere):
    python test_profiles.py --transfct ~/baryonification/baryonification/files/CDM_PLANCK_tk.dat
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

import baryonification as bfc

ap = argparse.ArgumentParser()
ap.add_argument("--transfct", required=True)
ap.add_argument("--mvir", type=float, default=1e14, help="halo mass [Msun/h]")
ap.add_argument("--cvir", type=float, default=5.0)
args = ap.parse_args()

par = bfc.par()
par.files.transfct = args.transfct
par.files.cosmofct = "cosmofct_test.dat"
par.cosmo.Om, par.cosmo.Ob, par.cosmo.h0 = 0.306, 0.0486, 0.681   # FLAMINGO

rbin = np.logspace(np.log10(0.005), np.log10(50), 100)

cc = bfc.cosmo.CosmoCalculator(par)
vc_r, vc_m, vc_var, vc_bias, vc_corr = cc.compute_cosmology()
cosmo_var = splev(args.mvir, splrep(vc_m, vc_var, s=0))
cosmo_bias = splev(args.mvir, splrep(vc_m, vc_bias, s=0))
cosmo_corr = splev(rbin, splrep(vc_r, vc_corr, s=0))

prof = bfc.Profiles(rbin, args.mvir, args.cvir, cosmo_corr, cosmo_bias, cosmo_var, par)
frac, den, mass, press, temp = prof.calc_profiles()

print("component fractions:", {k: round(float(v), 4) for k, v in frac.items() if np.ndim(v) == 0})
plt.figure(dpi=120)
plt.loglog(rbin, frac["HGA"] * den["HGA"], label="hot gas")
plt.loglog(rbin, den["DMO"], label="dark matter only")
plt.loglog(rbin, frac["CGA"] * den["CGA"] + frac["SGA"] * den["SGA"], label="stars (central + satellites)")
plt.loglog(rbin, den["DMB"] + den["BG"], "k", label="total baryonified (incl. 2-halo)")
plt.xlabel(r"$r$ [cMpc/$h$]")
plt.ylabel(r"$\rho$ [$h^2\,M_\odot$/cMpc$^3$]")
plt.title(f"M = {args.mvir:.0e} Msun/h, c = {args.cvir}")
plt.legend(fontsize=8)
plt.savefig("test_profiles.pdf", bbox_inches="tight")
print("wrote test_profiles.pdf")

"""
Sanity checks of a baryonified FLAMINGO shell against the DMO input map.

  1. Mass conservation: sum(dm + gas + star) should equal the input mass.
  2. Component fractions: gas + star should add up to Omega_b/Omega_m (0.159), stars ~1 per cent.
  3. Power spectrum ratio C_ell(baryonified) / C_ell(DMO): ~1 on large scales and suppressed
     on small scales (feedback pushes gas out of haloes).
  4. Side-by-side gnomonic zoom on the most massive halo in the shell: DMO vs baryonified vs gas.

Usage:
    python check_bfc_output.py bfc_flamingo_fiducial_shells6-6_nside1024.h5 --shell 6 \
        --map maps/L1_m9_DMO_lightcone0_shell6_DarkMatterMass_nside4096.hdf5 \
        --halos bfc_halos/halo_lightcone_shell_6.h5
"""
import argparse

import numpy as np
import h5py
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_cl_ratio import plot_cl_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bfc_file")
    ap.add_argument("--shell", type=int, required=True)
    ap.add_argument("--map", required=True, help="the DMO input map of this shell")
    ap.add_argument("--map-field", default="DarkMatterMass")
    ap.add_argument("--halos", default=None, help="per-shell halo file, to centre the zoom on the biggest halo")
    ap.add_argument("--out", default=None)
    ap.add_argument("--output-ring", action="store_true", help="output maps are already RING (default: NEST)")
    ap.add_argument("--fb", type=float, default=0.0486 / 0.306, help="Omega_b / Omega_m")
    args = ap.parse_args()
    out = args.out or f"bfc_check_shell{args.shell}"

    with h5py.File(args.bfc_file, "r") as f:
        comps = {g: f[g][f"shell_{args.shell}"][:].astype(float) for g in f if f"shell_{args.shell}" in f[g]}
    print("components:", list(comps))
    # the code writes its output maps in NESTED ordering (input FLAMINGO maps are RING)
    if not args.output_ring:
        comps = {k: hp.reorder(v, n2r=True) for k, v in comps.items()}
    nside = hp.npix2nside(len(next(iter(comps.values()))))
    with h5py.File(args.map, "r") as f:
        dmo = f[args.map_field][:].astype(float)
    if hp.npix2nside(len(dmo)) != nside:
        dmo = hp.ud_grade(dmo, nside_out=nside, power=-2)   # keep total mass
    # In multicomp mode every output map carries the full particle mass: 'dm' is the whole shell
    # displaced as dark matter, 'gas' + 'star' the whole shell displaced as baryons. The physical
    # maps are therefore weighted by the cosmic fractions (check this convention with Jozef):
    #   rho_total = (1 - fb) * dm + fb * (gas + star),   fb = Omega_b / Omega_m
    fb = args.fb
    if "dm" in comps:
        phys = {"dm": (1 - fb) * comps["dm"], "gas": fb * comps.get("gas", 0), "star": fb * comps.get("star", 0)}
        print(f"raw map sums / input: dm {comps['dm'].sum() / 1:.4e}, gas+star "
              f"{(comps.get('gas', 0) + comps.get('star', 0)).sum():.4e}  (each should equal the input mass)")
    else:
        phys = {"dmb": comps["dmb"]}
    comps = phys
    tot = sum(comps.values())

    print(f"\nnside {nside}")
    print(f"input DMO mass  : {dmo.sum():.6e}")
    print(f"output total    : {tot.sum():.6e}   (ratio {tot.sum() / dmo.sum():.5f}, expect ~1)")
    for k, v in comps.items():
        print(f"  {k:5s} fraction: {v.sum() / tot.sum():.4f}")
    print(f"  (expect gas + star = fb = {fb:.3f}; stars ~1 per cent of the total)")

    # power spectra of the overdensity
    def cl(m):
        d = m / m.mean() - 1
        return hp.anafast(d, lmax=3 * nside - 1)
    c0, c1 = cl(dmo), cl(tot)
    ell = np.arange(len(c0))
    np.savez(out + "_cl.npz", ell=ell, cl_bfc=c1, cl_dmo=c0)   # replot later with plot_cl_ratio.py
    plot_cl_ratio(ell, c1, c0, out + "_cl_ratio.pdf")
    ratio = c1 / np.where(c0 > 0, c0, 1)
    for L in (100, 500, 1000, 2000):
        if L < len(ratio):
            print(f"C_ell ratio at ell={L}: {ratio[L]:.3f}")

    # zoom on the biggest halo
    if args.halos:
        with h5py.File(args.halos, "r") as f:
            h = f["halos"][:]
        i = np.argmax(h["Mvir"])
        lon, lat = hp.vec2ang(np.array([h["x"][i], h["y"][i], h["z"][i]]), lonlat=True)
        rot = (float(lon[0]), float(lat[0]))
    else:
        rot = hp.pix2ang(nside, int(np.argmax(dmo)), lonlat=True)
    fig = plt.figure(figsize=(12, 4))
    kw = dict(rot=rot, xsize=200, reso=60 * hp.nside2resol(nside, arcmin=True) / 60 * 1.0,
              fig=fig.number, cbar=True, notext=True)
    vmin, vmax = np.log10(max(dmo.mean() * 0.3, 1e-30)), np.log10(dmo.max())
    for k, (m, t) in enumerate([(dmo, "DMO"), (tot, "baryonified"), (comps.get("gas", tot), "gas")]):
        hp.gnomview(np.log10(np.maximum(m, 1e-30)), sub=(1, 3, k + 1), title=t, min=vmin, max=vmax, **kw)
    fig.savefig(out + "_zoom.pdf", bbox_inches="tight")
    print(f"\nwrote {out}_cl_ratio.pdf, {out}_cl.npz and {out}_zoom.pdf")


if __name__ == "__main__":
    main()

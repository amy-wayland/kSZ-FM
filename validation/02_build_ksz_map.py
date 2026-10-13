import argparse
import os
import numpy as np

from ksz_utils import T_CMB_K, find_shell_files

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BOX = "L1_m9"               # box/resolution grouping directory
SIM = "L1_m9"               # run inside it; set by --sim
LIGHTCONE = "lightcone0"
MAP_NAME = "DopplerB"
PREFER_SWIFT = False        # set by --prefer
REF_NSIDE = 16384           # resolution that expected_sum refers to

# --------------------------------------------------------------------------
# Open shell directory
# --------------------------------------------------------------------------

def open_shell(shell_nr, basedir, remote, nside_dir):
    """
    Return (file object supporting f[MAP_NAME][...], its path).

    The archive stores a directory per shell rather than the single file the
    docs describe, and a shell directory can hold more than one HDF5 file, so
    the path is resolved at runtime and the file carrying MAP_NAME is chosen.
    """
    if remote:
        import hdfstream
        root = hdfstream.open("cosma", "/")
        base = f"FLAMINGO/{BOX}/{SIM}"
        ls = lambda p: sorted(list(root[p]))
        opener = lambda p: root[p]
    else:
        import h5py
        base = basedir.rstrip("/")
        ls = lambda p: sorted(os.listdir(p))
        opener = lambda p: h5py.File(p, "r")

    candidates = find_shell_files(ls, base, shell_nr, LIGHTCONE, nside_dir)
    # A shell directory holds two files with the same maps:
    #   lightcone0.shell_N.0.hdf5        X-ray maps tagged _FrozenUVB
    #   swift_lightcone0.shell_N.0.hdf5  SWIFT's on-the-fly version
    # DopplerB is expected to be identical in both; the sort below makes the
    # choice deterministic rather than accidental. Use --prefer to flip it.
    if PREFER_SWIFT:
        candidates = sorted(candidates, key=lambda p: not os.path.basename(p).startswith("swift_"))
    else:
        candidates = sorted(candidates, key=lambda p: os.path.basename(p).startswith("swift_"))
    tried = []
    for path in candidates:
        f = opener(path)
        if MAP_NAME in list(f):
            return f, path
        tried.append((path, list(f)))
        if hasattr(f, "close"):
            f.close()

    msg = "\n".join(f"  {p}: {names}" for p, names in tried)
    raise KeyError(f"{MAP_NAME} not found in any file for shell {shell_nr}:\n{msg}")

# --------------------------------------------------------------------------
# Construct kSZ map
# --------------------------------------------------------------------------

def main():
    global PREFER_SWIFT, SIM
    p = argparse.ArgumentParser()
    p.add_argument("--shell", type=int, default=None,
                   help="central shell number")
    p.add_argument("--pad", type=int, default=1,
                   help="number of padding shells on each side (McCarthy+24: 1)")
    p.add_argument("--shells", type=int, nargs="+", default=None,
                   help="explicit shell list, overriding --shell/--pad. A galaxy "
                        "bin of 0.525<z<0.575 straddles shells 10 and 11, so it "
                        "wants --shells 9 10 11 12")
    p.add_argument("--prefer", choices=["standard", "swift"], default="standard",
                   help="which of the two files in a shell directory to read")
    p.add_argument("--sim", default=SIM,
                   help="run inside FLAMINGO/L1_m9/: L1_m9 (fiducial), "
                        "fgas+2sigma, fgas-2sigma, fgas-4sigma, fgas-8sigma, "
                        "Jet, Jet_fgas-4sigma, NoCooling, L1_m9_DMO")
    p.add_argument("--nside-dir", default="nside_4096",
                   choices=["nside_4096", "nside_16384"])
    p.add_argument("--basedir", default=None,
                   help="local FLAMINGO/L1_m9/L1_m9 directory")
    p.add_argument("--remote", action="store_true",
                   help="stream from the Durham server instead")
    p.add_argument("--out", default=None)
    p.add_argument("--extra-1pz", action="store_true",
                   help="multiply each DopplerB shell by a further 1+z_mid, on "
                        "top of the released correction. Consistency with the "
                        "DM maps suggests the released maps are low by this "
                        "factor (provisional)")
    args = p.parse_args()

    if not args.remote and args.basedir is None:
        p.error("give either --remote or --basedir")
    if args.shells is None and args.shell is None:
        p.error("give either --shell or --shells")

    PREFER_SWIFT = (args.prefer == "swift")
    SIM = args.sim
    print(f"simulation: FLAMINGO/{BOX}/{SIM}")

    shells = (args.shells if args.shells is not None
              else list(range(args.shell - args.pad, args.shell + args.pad + 1)))
    print(f"summing shells {list(shells)}")
    total = None
    checksum_ok = True

    for s in shells:
        if s < 0:
            continue
        f, path = open_shell(s, args.basedir, args.remote, args.nside_dir)
        print(f"shell {s}: {path}", flush=True)
        try:
            print(f"  Shell attrs: {dict(f['Shell'].attrs)}", flush=True)
        except Exception as exc:
            print(f"  (no Shell group: {exc})", flush=True)

        dset = f[MAP_NAME]
        attrs = dict(getattr(dset, "attrs", {}))
        b = np.asarray(dset[...], dtype=np.float64)
        print(f"  read {b.size} pixels, "
              f"mean={b.mean():.3e} rms={b.std():.3e}", flush=True)

        # The file carries expected_sum, a checksum over all pixels. Two known
        # offsets have to be undone before comparing:
        #
        #  * expected_sum refers to the full-resolution nside 16384 map. b is a
        #    per-solid-angle quantity, so downsampling averages rather than adds
        #    and the sum scales as (nside/16384)^2 - a factor 1/16 at nside 4096.
        #  * expected_sum was computed before the (1+z_centre) bug correction
        #    that the released data already has applied, so the stored data is
        #    larger than the checksum by "Correction factor".
        #
        # sum_here = expected_sum * correction_factor * (nside/16384)^2
        #
        this_nside = int(round(np.sqrt(b.size / 12)))
        cf = float(np.atleast_1d(attrs.get("Correction factor", [1.0]))[0])
        if "expected_sum" in attrs:
            raw = float(np.atleast_1d(attrs["expected_sum"])[0])
            want = raw * cf * (this_nside / REF_NSIDE) ** 2
            got = float(b.sum())
            rel = abs(got - want) / max(abs(want), 1e-30)
            flag = "OK" if rel < 1e-5 else "MISMATCH"
            print(f"  checksum: sum={got:.8f} predicted={want:.8f} "
                  f"rel={rel:.2e} [{flag}]", flush=True)
            print(f"    (raw expected_sum {raw:.8f} x correction {cf} "
                  f"x (nside {this_nside}/{REF_NSIDE})^2)")
            if rel >= 1e-5:
                checksum_ok = False
        else:
            print("  (no expected_sum attribute to check against)")

        for key in ("Correction applied",
                    "Central redshift assumed for correction"):
            if key in attrs:
                print(f"  {key}: {np.atleast_1d(attrs[key])[0]}")
                
        # Provisional extra (1+z): applied after the checksum, so the checksum
        # still validates the released data as read.
        if args.extra_1pz:
            z_mid = float(np.atleast_1d(attrs.get(
                "Central redshift assumed for correction", [np.nan]))[0])
            if not np.isfinite(z_mid):
                raise SystemExit("--extra-1pz: this shell has no 'Central "
                                 "redshift assumed for correction' attribute")
            b *= 1.0 + z_mid
            print(f"  DopplerB multiplied by a further 1+z = {1 + z_mid:.3f}")
            
        if total is None:
            total = b
        else:
            np.add(total, b, out=total)
            del b
        if hasattr(f, "close"):
            f.close()

    # Convert dT/T = -b to dT in micro-kelvin
    total *= -T_CMB_K * 1e6
    dT_uK = total.astype(np.float32)
    del total

    npix = dT_uK.size
    nside = int(round(np.sqrt(npix / 12)))
    print(f"\nnside {nside}, {npix} pixels")
    print(f"dT_kSZ: mean {dT_uK.mean():.4f} uK, rms {dT_uK.std():.4f} uK, "
          f"min {dT_uK.min():.2f}, max {dT_uK.max():.2f}")

    tag = (f"shells{'-'.join(str(s) for s in shells)}" if args.shells
           else f"shell{args.shell}")
    suffix = "_x1pz" if args.extra_1pz else ""
    out = args.out or f"dT_ksz_{SIM}_{LIGHTCONE}_{tag}_{args.nside_dir}{suffix}.npy"
    np.save(out, dT_uK)
    print(f"wrote {out}  ({os.path.getsize(out)/1e9:.2f} GB)")

    # Sanity check: the monopole should be tiny compared to the rms, because
    # kSZ is a velocity-weighted (signed) effect.
    if abs(dT_uK.mean()) > 0.1 * dT_uK.std():
        print("WARNING: large monopole - check you read DopplerB and not ComptonY")
    if not checksum_ok:
        print("WARNING: at least one shell failed its expected_sum checksum. ")


if __name__ == "__main__":
    main()

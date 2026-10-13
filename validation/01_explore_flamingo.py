import sys
import traceback
import numpy as np
import hdfstream

from ksz_utils import find_shell_files

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BOX = "L1_m9"             # the box/resolution grouping directory
SIM = "L1_m9"             # the run inside it: L1_m9, fgas-4sigma, Jet, etc
NSIDE_DIR = "nside_4096"
LIGHTCONE = "lightcone0"  # observer 0 has 60 shells out to z=3
SHELL = 10                # 0.50 < z < 0.55

BASE = f"FLAMINGO/{BOX}/{SIM}"

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def show(title):
    """
    Print a section banner.
    """
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


def safe(title, fn, *args, **kwargs):
    """
    Run one section at a time. 
    Teport a failure but keep going as sections are independent.
    """
    show(title)
    try:
        return fn(*args, **kwargs)
    except Exception:
        print("--- Section failed, continuing ---")
        traceback.print_exc(limit=3)
        return None


def dump_attrs(obj, indent="    "):
    """
    Print every HDF5 attribute of a given dataset.
    """
    try:
        for k, v in dict(obj.attrs).items():
            print(f"{indent}{k} = {v}")
    except Exception as exc:
        print(f"{indent}<could not read attrs: {exc}>")


def walk(g, depth=0, max_depth=3):
    """
    Recursively print an HDF5 tree: datasets with their shape and dtype.
    """
    if depth > max_depth:
        return
    for k in list(g):
        child = g[k]
        try:
            print("  " * depth + f"{k}  shape={child.shape} dtype={child.dtype}")
        except AttributeError:
            print("  " * depth + f"{k}/")
            walk(child, depth + 1, max_depth)

# --------------------------------------------------------------------------
# Inspect metadata
# --------------------------------------------------------------------------

def main():
    import argparse
    global SIM, BASE, NSIDE_DIR, SHELL
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default=SIM,
                    help="run inside FLAMINGO/L1_m9/, e.g. fgas-4sigma, Jet")
    ap.add_argument("--nside-dir", default=NSIDE_DIR)
    ap.add_argument("--shell", type=int, default=SHELL)
    a = ap.parse_args()
    SIM, NSIDE_DIR, SHELL = a.sim, a.nside_dir, a.shell
    BASE = f"FLAMINGO/{BOX}/{SIM}"
    print(f"exploring {BASE}")

    root = hdfstream.open("cosma", "/")
    ls = lambda p: sorted(list(root[p]))

    safe("Top-level FLAMINGO directories",
         lambda: print(ls("FLAMINGO")))

    safe(f"Simulations under FLAMINGO/{BOX}",
         lambda: print(ls(f"FLAMINGO/{BOX}")))

    safe(f"Data products for {SIM}",
         lambda: print(ls(BASE)))

    # ------------------------------------------------------------------
    # HEALPix shells
    # ------------------------------------------------------------------
    def shells():
        """
        Determine which resolutions are available for a given run.
        """
        print("resolutions:", ls(f"{BASE}/healpix_maps"))
        sd = f"{BASE}/healpix_maps/{NSIDE_DIR}/{LIGHTCONE}_shells"
        entries = ls(sd)
        print(f"{len(entries)} entries under {LIGHTCONE}_shells, "
              f"first few: {entries[:4]}")
        files = find_shell_files(ls, BASE, SHELL, LIGHTCONE, NSIDE_DIR)
        print(f"\nshell {SHELL} resolves to:")
        for f in files:
            print("   ", f)
        return files

    files = safe("HEALPix shell layout", shells)

    def shell_contents(files):
        """
        Print what is inside each file in a shell and the required metadata.
        """
        for path in files:
            f = root[path]
            names = list(f)
            print(f"\n{path}")
            print("  datasets/groups:", names)
            if "DopplerB" in names:
                d = f["DopplerB"]
                print("\n  DopplerB attributes:")
                dump_attrs(d, "      ")
                npix = int(np.prod(d.shape))
                nside = int(round(np.sqrt(npix / 12)))
                print(f"      shape {d.shape} dtype {d.dtype}")
                print(f"      -> nside {nside}, "
                      f"{np.degrees(np.sqrt(4*np.pi/npix))*60:.3f} arcmin pixels, "
                      f"{npix*8/1e9:.2f} GB as float64")
            for grp in ("Shell", "Units"):
                if grp in names:
                    print(f"\n  {grp} attributes:")
                    dump_attrs(f[grp], "      ")

    if files:
        safe(f"Inside the shell {SHELL} file(s)", shell_contents, files)

    # ------------------------------------------------------------------
    # Halo lightcone
    # ------------------------------------------------------------------
    def halos():
        hl = ls(f"{BASE}/halo_lightcone/{LIGHTCONE}")
        print(f"{len(hl)} files: {hl[:3]} ... {hl[-3:]}")
        for snap in (66, 67):
            name = f"lightcone_halos_{snap:04d}.hdf5"
            if name not in hl:
                print(f"  {name} NOT FOUND - real names above")
                continue
            f = root[f"{BASE}/halo_lightcone/{LIGHTCONE}/{name}"]
            print(f"\n--- {name} ---")
            walk(f)
            try:
                z = f["Lightcone/Redshift"]
                zs = z[:200000]
                print(f"  n halos: {z.shape[0]}, "
                      f"z in first chunk: {zs.min():.4f} - {zs.max():.4f}")
            except Exception as exc:
                print("  could not read Lightcone/Redshift:", exc)

    safe("Halo lightcone files", halos)

    # ------------------------------------------------------------------
    # SOAP names
    # ------------------------------------------------------------------
    def soap():
        sf_names = ls(f"{BASE}/SOAP-HBT")
        print(f"{len(sf_names)} files: {sf_names[:3]} ... {sf_names[-3:]}")
        target = "halo_properties_0067.hdf5"
        name = target if target in sf_names else sf_names[len(sf_names) // 2]
        sf = root[f"{BASE}/SOAP-HBT/{name}"]
        print(f"\n--- {name} ---")
        print("top-level groups:", list(sf))
        for grp in ("InputHalos", "BoundSubhalo", "ExclusiveSphere",
                    "SO", "SphericalOverdensity"):
            if grp in list(sf):
                print(f"\n{grp}/ ->", list(sf[grp])[:30])
                # one level deeper for the aperture/overdensity containers
                for sub in list(sf[grp])[:6]:
                    try:
                        child = sf[grp][sub]
                        if not hasattr(child, "shape"):
                            print(f"   {grp}/{sub}/ ->", list(child)[:30])
                    except Exception:
                        pass
        for grp in ("Header", "Cosmology", "Parameters"):
            if grp in list(sf):
                print(f"\n[{grp}]")
                dump_attrs(sf[grp])

    safe("SOAP halo catalogue", soap)


if __name__ == "__main__":
    sys.exit(main())

"""
Download one FLAMINGO HEALPix lightcone shell map (streamed with hdfstream) to a
small local HDF5 file that overlay_halos_on_map.py can read.

Layout (FLAMINGO data release docs):
  FLAMINGO/<box>/<run>/healpix_maps/nside_<N>/lightcone<X>_shells/lightcone<X>_shell_<Y>.0.hdf5
  one file per shell; each map is a 1D float64 dataset in the root group, RING ordering;
  attrs comoving_inner_radius / comoving_outer_radius are comoving Mpc (no h);
  mass maps are in 1e10 Msun per pixel.
Shells are dz = 0.05 wide: shell Y covers 0.05*Y < z < 0.05*(Y+1) (shell 0 starts at 0.001).

Usage:
    python fetch_healpix_shell.py --list                       # the DMO healpix directory
    python fetch_healpix_shell.py --observer 0 --shell 6       # z = 0.30-0.35
"""
import argparse
import os
import numpy as np
import h5py

H = 0.681

def show_attrs(obj, label):
    try:
        attrs = dict(obj.attrs)
    except Exception as e:  # remote object without attrs support
        print(f"  ({label}: could not read attrs: {e})")
        return {}
    for k, v in attrs.items():
        print(f"  {label}.attrs[{k!r}] = {v}")
    return attrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", default="L1_m9")
    ap.add_argument("--run", default="L1_m9_DMO")
    ap.add_argument("--nside", type=int, default=4096)
    ap.add_argument("--observer", type=int, default=0)
    ap.add_argument("--shell", type=int, default=6)
    ap.add_argument("--map", default="DarkMatterMass", help="falls back to TotalMass if missing")
    ap.add_argument("--outdir", default="maps")
    ap.add_argument("--list", action="store_true", help="list the healpix directory and exit")
    ap.add_argument("--file", default=None, help="file name inside shell_<Y>/ (if there are several)")
    args = ap.parse_args()

    import hdfstream
    root = hdfstream.open("cosma", "/")
    base = f"FLAMINGO/{args.box}/{args.run}/healpix_maps/nside_{args.nside}"

    if args.list:
        d = root[base]
        print(base, "->", list(d.keys()) if hasattr(d, "keys") else d)
        sub = f"{base}/lightcone{args.observer}_shells"
        d = root[sub]
        print(sub, "->", sorted(d.keys())[:10] if hasattr(d, "keys") else d, "...")
        sh = f"{sub}/shell_{args.shell}"
        d = root[sh]
        print(sh, "->", sorted(d.keys()) if hasattr(d, "keys") else d)
        return

    # The docs give lightcone<X>_shells/lightcone<X>_shell_<Y>.0.hdf5, but on the server
    # each shell is a subdirectory (lightcone<X>_shells/shell_<Y>/...).
    shells_dir = f"{base}/lightcone{args.observer}_shells"
    path = None
    try:
        sdir = root[f"{shells_dir}/shell_{args.shell}"]
        names = sorted(sdir.keys()) if hasattr(sdir, "keys") else []
        print(f"{shells_dir}/shell_{args.shell} ->", names)
        h5s = [n for n in names if n.endswith(".hdf5")]
        if not h5s and any(n in names for n in (args.map, "TotalMass", "Shell")):
            h5s = [None]  # shell_<Y> is itself the HDF5 file
        if len(h5s) != 1 and not args.file:
            raise SystemExit(f"expected one .hdf5 file in shell_{args.shell}, found {h5s}: "
                             "pass the right one with --file")
        path = (f"{shells_dir}/shell_{args.shell}" if h5s == [None] else
                f"{shells_dir}/shell_{args.shell}/{h5s[0]}" if h5s else None)
    except KeyError:
        path = f"{shells_dir}/lightcone{args.observer}_shell_{args.shell}.0.hdf5"
    if args.file:
        path = f"{shells_dir}/shell_{args.shell}/{args.file}"
    print("opening", path)
    f = root[path]

    print("file contents:", list(f.keys()) if hasattr(f, "keys") else "?")
    show_attrs(f, "file")
    for grp in ("Shell", "Header", "Lightcone", "Units"):
        try:
            show_attrs(f[grp], grp)
        except KeyError:
            pass

    name = args.map
    try:
        ds = f[name]
    except KeyError:
        print(f"{name} not in file, using TotalMass")
        name = "TotalMass"
        ds = f[name]
    dattrs = show_attrs(ds, name)

    print(f"downloading {name} ({ds.shape[0]} pixels, ~{ds.shape[0]*8/1e9:.1f} GB) ...", flush=True)
    m = np.asarray(ds[...], dtype=np.float64)

    r_in = float(np.atleast_1d(dattrs.get("comoving_inner_radius", np.nan))[0])
    r_out = float(np.atleast_1d(dattrs.get("comoving_outer_radius", np.nan))[0])
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{args.run}_lightcone{args.observer}_shell{args.shell}_{name}_nside{args.nside}.hdf5")
    with h5py.File(out, "w") as g:
        d = g.create_dataset(name, data=m)
        for k, v in dattrs.items():
            d.attrs[k] = v
        g.attrs["source_path"] = path
        g.attrs["rmin_Mpch"] = r_in * H
        g.attrs["rmax_Mpch"] = r_out * H

    print(f"\nwrote {out}")
    print(f"shell radii: {r_in:.2f} - {r_out:.2f} Mpc  =  {r_in*H:.2f} - {r_out*H:.2f} Mpc/h")
    print(f"map: mean {m.mean():.4g}, min {m.min():.4g}, max {m.max():.4g}, zero pixels {np.mean(m==0)*100:.2f}%")
    print("\noverlay command:")
    print(f"  python overlay_halos_on_map.py <halo_lightcone.h5> --map {out} --dataset {name} "
          f"--rmin {r_in*H:.2f} --rmax {r_out*H:.2f} --out plots/overlay_lc{args.observer}_shell{args.shell}")


if __name__ == "__main__":
    main()

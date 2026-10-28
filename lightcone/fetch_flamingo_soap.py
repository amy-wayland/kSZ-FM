"""
Fetch the parts of the public FLAMINGO SOAP-HBT catalogues that the lightcone
builder needs, and write the snapshot list read by run_prep_halocone_crossing.py.

Writes:
  flamingo_halo_files.txt : snapnum  local_path
Then run add_track_info.py to add the HBT track information.

Usage (z <= 2 is snapshots 37..77 for L1_m9):
    pip install hdfstream
    python fetch_flamingo_soap.py --snaps 37-77 --outdir soap_subset \
        --mhalo-min 8e12 --msub-min 1e12 --nmin 100
"""
import argparse
import os
import time
import numpy as np
import h5py

H = 0.681  # FLAMINGO D3A

CUT_COLS = {
    "InputHalos/IsCentral": None,
    "SO/200_crit/TotalMass": None,
    "SO/200_crit/NumberOfDarkMatterParticles": None,
    "BoundSubhalo/TotalMass": "BoundSubhaloProperties/TotalMass",
    "BoundSubhalo/NumberOfDarkMatterParticles": "BoundSubhaloProperties/NumberOfDarkMatterParticles",
    "SOAP/HostHaloIndex": None,
}
OTHER_COLS = {
    "InputHalos/HaloCatalogueIndex": None,
    "InputHalos/HaloCentre": None,
    "SO/200_crit/CentreOfMassVelocity": None,
    "SO/200_crit/SORadius": None,
    "SO/200_crit/Concentration": None,
    "BoundSubhalo/CentreOfMassVelocity": "BoundSubhaloProperties/CentreOfMassVelocity",
}


def get_ds(f, name, alt):
    for n in (name, alt):
        if n is None:
            continue
        try:
            return f[n]
        except KeyError:
            pass
    raise KeyError(f"{name} not found (also tried {alt})")


def read_rows(ds, rows, max_slices=20000):
    """Read the given (sorted, unique) rows of a remote dataset, as few contiguous slices."""
    if len(rows) == 0:
        return np.zeros((0,) + tuple(ds.shape[1:]), dtype=ds.dtype)
    breaks = np.flatnonzero(np.diff(rows) > 1) + 1
    starts = np.r_[rows[0], rows[breaks]]
    stops = np.r_[rows[breaks - 1] + 1, rows[-1] + 1]
    ndim = len(ds.shape)
    if not hasattr(ds, "request_slices"):  # plain h5py (testing) or old hdfstream
        return ds[...][rows]
    out = []
    for i in range(0, len(starts), max_slices):
        sl = [np.s_[a:b] if ndim == 1 else np.s_[a:b, :]
              for a, b in zip(starts[i:i + max_slices], stops[i:i + max_slices])]
        out.append(np.asarray(ds.request_slices(sl)))
    return np.concatenate(out, axis=0)


def snapshot_redshift(f):
    for grp in ("Header", "Cosmology"):
        try:
            z = f[grp].attrs["Redshift"]
            return float(np.atleast_1d(z)[0])
        except (KeyError, AttributeError):
            pass
    raise KeyError("no Redshift attribute in Header/Cosmology")


def fetch_one(f, out_path, mhalo_min, msub_min, nmin):
    cut = {k: np.asarray(get_ds(f, k, alt)[...]) for k, alt in CUT_COLS.items()}
    isc = cut["InputHalos/IsCentral"] == 1
    m200 = cut["SO/200_crit/TotalMass"] * 1e10 * H
    n200 = cut["SO/200_crit/NumberOfDarkMatterParticles"]
    mb = cut["BoundSubhalo/TotalMass"] * 1e10 * H
    nb = cut["BoundSubhalo/NumberOfDarkMatterParticles"]
    host = cut["SOAP/HostHaloIndex"]

    sel_c = isc & (m200 >= mhalo_min) & (n200 >= nmin)
    sel_s = (~isc) & (mb >= msub_min) & (nb >= nmin)
    keep = sel_c | sel_s
    keep[host[sel_s]] = True  # hosts of kept satellites, needed for the remap
    rows = np.flatnonzero(keep)

    newidx = np.full(len(isc), -1, dtype=np.int64)
    newidx[rows] = np.arange(len(rows))
    host_new = np.where(host[rows] >= 0, newidx[np.maximum(host[rows], 0)], -1)

    z = snapshot_redshift(f)
    tmp = out_path + ".tmp"
    with h5py.File(tmp, "w") as g:
        g.create_group("Header").attrs["Redshift"] = z
        for k in CUT_COLS:
            g[k] = host_new if k == "SOAP/HostHaloIndex" else cut[k][rows]
        for k, alt in OTHER_COLS.items():
            g[k] = read_rows(get_ds(f, k, alt), rows)
        g.attrs["n_rows_full"] = len(isc)
        g.attrs["fetch_cuts"] = np.array([mhalo_min, msub_min, nmin])
    os.replace(tmp, out_path)
    return z, len(isc), int(sel_c.sum()), int(sel_s.sum()), len(rows)


def parse_snaps(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", default="37-77")
    ap.add_argument("--remote-dir", default="FLAMINGO/L1_m9/L1_m9_DMO/SOAP-HBT",
                    help="check this path in the FLAMINGO data browser")
    ap.add_argument("--server", default="cosma")
    ap.add_argument("--outdir", default="soap_subset")
    ap.add_argument("--mhalo-min", type=float, default=8e12, help="M200c [Msun/h]")
    ap.add_argument("--msub-min", type=float, default=1e12, help="bound mass [Msun/h]")
    ap.add_argument("--nmin", type=int, default=100)
    ap.add_argument("--local-dir", default=None,
                    help="read SOAP files from a local directory with h5py instead of streaming")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.local_dir is None:
        import hdfstream
        root = hdfstream.open(args.server, "/")
        opener = lambda p: root[f"{args.remote_dir}/{p}"]
    else:
        opener = lambda p: h5py.File(os.path.join(args.local_dir, p), "r")

    snaps, zs, paths = [], [], []
    for s in parse_snaps(args.snaps):
        name = f"halo_properties_{s:04d}.hdf5"
        out = os.path.abspath(os.path.join(args.outdir, name))
        t0 = time.time()
        if os.path.exists(out):  # resume
            with h5py.File(out, "r") as g:
                z = float(g["Header"].attrs["Redshift"])
            print(f"snap {s:3d}: exists, z={z:.4f}")
        else:
            z, nfull, nc, ns, nrow = fetch_one(opener(name), out, args.mhalo_min, args.msub_min, args.nmin)
            print(f"snap {s:3d}: z={z:.4f}  rows {nfull} -> {nrow}  ({nc} centrals, {ns} satellites)  "
                  f"{time.time()-t0:.0f}s", flush=True)
        snaps.append(s); zs.append(z); paths.append(out)

    o = np.argsort(snaps)
    snaps = np.array(snaps)[o]; paths = [paths[i] for i in o]
    np.savetxt("flamingo_halo_files.txt", np.c_[snaps, paths], fmt="%s", header="snapnum path")
    print("wrote flamingo_halo_files.txt")


if __name__ == "__main__":
    main()

"""
Add the HBT track information needed by the crossing-based lightcone builder
(prep_halocone_crossing.py) to the existing soap_subset files, without re-fetching anything else.

For every local file halo_properties_XXXX.hdf5 it writes
  InputHalos/HBTplus/TrackId   TrackId of each local row (same row order as the rest of the file)
and, for every snapshot s that has a later snapshot s+1 in the set,
  Tracks/TrackId, Tracks/HaloCentre
      position at snapshot s (comoving Mpc, h-free, as in SOAP) of every object whose TrackId
      appears in the local file of snapshot s+1. These are the "outer" positions used to
      interpolate each object between the two snapshots bracketing a map shell; objects that
      only pass the mass cuts at s+1 are included, which is why the local rows of s alone are
      not enough.

Local rows are matched to remote rows through InputHalos/HaloCatalogueIndex.

Usage (same remote path / server as fetch_flamingo_soap.py):
    python -u add_track_info.py --snaps 37-77 --outdir soap_subset
    # or from local SOAP files:  --local-dir /path/to/SOAP-HBT
Resumable: snapshots that already have both datasets are skipped.
"""
import argparse
import os
import time

import numpy as np
import h5py

from fetch_flamingo_soap import read_rows, parse_snaps

TID = "InputHalos/HBTplus/TrackId"
CAT = "InputHalos/HaloCatalogueIndex"
CEN = "InputHalos/HaloCentre"


def remote_rows_of(f_remote, cat_local, cache):
    """Row indices in the remote file of the local rows, via HaloCatalogueIndex."""
    ds = f_remote[CAT]
    n = ds.shape[0]
    head = np.asarray(ds[0:min(n, 100000)])
    if np.array_equal(head, np.arange(len(head))):      # HaloCatalogueIndex == row (cheap path)
        tail = np.asarray(ds[n - 1000:n]) if n > 1000 else head
        if np.array_equal(tail, np.arange(n - len(tail), n)):
            return cat_local.astype(np.int64)
    if "cat" not in cache:
        cache["cat"] = np.asarray(ds[...])
    cat = cache["cat"]
    o = np.argsort(cat)
    k = np.searchsorted(cat[o], cat_local)
    if np.any(k >= n) or np.any(cat[o][np.minimum(k, n - 1)] != cat_local):
        raise ValueError("some local HaloCatalogueIndex values not found in the remote file")
    return o[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", default="37-77")
    ap.add_argument("--remote-dir", default="FLAMINGO/L1_m9/L1_m9_DMO/SOAP-HBT")
    ap.add_argument("--server", default="cosma")
    ap.add_argument("--outdir", default="soap_subset")
    ap.add_argument("--local-dir", default=None)
    args = ap.parse_args()

    if args.local_dir is None:
        import hdfstream
        root = hdfstream.open(args.server, "/")
        opener = lambda p: root[f"{args.remote_dir}/{p}"]
    else:
        opener = lambda p: h5py.File(os.path.join(args.local_dir, p), "r")

    snaps = sorted(parse_snaps(args.snaps))
    name = lambda s: f"halo_properties_{s:04d}.hdf5"
    local = lambda s: os.path.join(args.outdir, name(s))

    # pass 1: TrackId of every local row (needed first, because pass 2 of snapshot s uses s+1)
    for s in snaps:
        t0 = time.time()
        with h5py.File(local(s), "a") as g:
            if TID in g:
                print(f"snap {s:3d}: TrackId present", flush=True)
                continue
            fr = opener(name(s))
            rows = remote_rows_of(fr, g[CAT][:], {})
            srt = np.argsort(rows)
            tid = np.empty(len(rows), dtype=np.int64)
            tid[srt] = read_rows(fr[TID], rows[srt])
            g[TID] = tid
        print(f"snap {s:3d}: wrote TrackId for {len(rows)} rows  {time.time()-t0:.0f}s", flush=True)

    # pass 2: positions at s of everything that is in the local file of s+1
    for s in snaps:
        if s + 1 not in snaps:
            continue
        t0 = time.time()
        with h5py.File(local(s + 1), "r") as g1:
            need = np.unique(g1[TID][:])
        with h5py.File(local(s), "a") as g:
            if "Tracks/HaloCentre" in g:
                print(f"snap {s:3d}: Tracks present", flush=True)
                continue
            fr = opener(name(s))
            tid_all = np.asarray(fr[TID][...])               # one integer column, all rows
            rows = np.flatnonzero(np.isin(tid_all, need))
            cen = read_rows(fr[CEN], rows)
            grp = g.require_group("Tracks")
            grp["TrackId"] = tid_all[rows]
            grp["HaloCentre"] = cen
            grp.attrs["for_snapshot"] = s + 1
            del tid_all
        print(f"snap {s:3d}: {len(rows)} of {len(need)} tracks of snap {s+1} found "
              f"({len(need) - len(rows)} new since s)  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

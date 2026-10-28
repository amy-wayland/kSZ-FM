"""
Halo + subhalo lightcone for the FLAMINGO DMO maps, with every object placed where it crosses
the lightcone (crossing-based placement, adapted from code by J. Bucko's student).

The FLAMINGO HEALPix maps record each particle when it crosses the observer's past lightcone;
map shell i collects the crossings between z_i and z_{i+1} (z_i = 0.05 i). The halo catalogues
exist only at the snapshot redshifts, so each shell is built from the two snapshots at its edges:

    inner snapshot  (z_i,     later,   lightcone radius r_lo)   positions R1
    outer snapshot  (z_{i+1}, earlier, lightcone radius r_hi)   positions R0 (same HBT TrackId)

Between the two snapshots each object is moved on a straight line, x(v) = (1-v) R0 + v R1,
and the lightcone radius shrinks linearly, r_lc(v) = r_hi - v (r_hi - r_lo). Linearising
|x(v)| = r0 + v (r1 - r0), the object crosses the lightcone at

    v = (r_hi - r0) / ((r_hi - r_lo) + (r1 - r0))

and is kept if 0 <= v < 1 and r_lo <= |x(v)| < r_hi, at position x(v). Objects with no match at
the outer snapshot (tracks born between the snapshots) stay at R1.

  - Centrals: M200c >= Mhalo_min, N200c >= Nmin_per_halo (SOAP SO/200_crit quantities).
  - Satellites: bound mass >= Msub_min, Nbound >= Nmin_per_subhalo, and (require_host_above_cut)
    only if their host central passes the central cut. IDhost and the box-replica label link
    each satellite to its host.
  - Outer positions come from the Tracks/ group written by add_track_info.py.
  - Properties (mass, velocity, radius, concentration) are those at the inner snapshot, where
    the selection is made.
  - Shell rows carry the map-shell edges, so lightcone shell i pairs with map shell i.
"""
import os
from types import SimpleNamespace as NS

import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

# IDs in the lightcone are made unique per shell as ID + shell_id * ID_STRIDE
ID_STRIDE = 10**8


def get_shell_cov_dist(z_boundary, cosmo):

    # get the comoving distances
    cov_inner = cosmo.comoving_distance(z_boundary[0]).to(u.Mpc / cosmo.h).value
    cov_outer = cosmo.comoving_distance(z_boundary[1]).to(u.Mpc / cosmo.h).value
    # we take the same shell comoving distance as UFalcon
    shell_cov = cosmo.comoving_distance(np.mean(z_boundary)).to(u.Mpc / cosmo.h).value

    return shell_cov, cov_inner, cov_outer

def check_points(points, min_val, max_val, buffer=1e-9):
    """
    A function that checks if all points are either within min_val or large than max_val
    """
    # get the radi of the points
    r_points = np.linalg.norm(points, axis=1)

    # if all points are larger than the outer shell return false
    if np.all(r_points > max_val + buffer):
        return False
    # if all points are smaller than the outer shell return false
    if np.all(r_points < min_val - buffer):
        return False
    # Otherwise we return True
    return True

def get_offset_indices(boxsize, min_val, max_val, max_repli=7, buffer=1):
    """
    Gets all indices that intersect with a given shell
    """
    box_indices = []

    # cycle through all boxes
    for i in range(-max_repli, max_repli):
        for j in range(-max_repli, max_repli):
            for k in range(-max_repli, max_repli):
                # set the origin
                origin = np.array([i, j, k])*boxsize
                # check all the corners
                corners = np.zeros((8, 3))
                corner_index = 0
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            # set corner offset
                            corner_offset = np.array([ii, jj, kk])*boxsize
                            # set corner
                            corners[corner_index] = origin + corner_offset
                            # update
                            corner_index += 1
                # check
                check_box = check_points(corners, min_val, max_val,buffer=buffer)
                if check_box:
                    box_indices.append(np.array([i,j,k]))
    return box_indices


def encode_replica(offset, max_repli):
    """Unique integer label for a box replica (i,j,k), i,j,k in [-max_repli, max_repli)."""
    n = 2 * max_repli
    i, j, k = (np.asarray(offset, dtype=int) + max_repli)
    return int((i * n + j) * n + k)


def decode_replica(replica, max_repli):
    """Inverse of encode_replica -> array (..., 3) of (i,j,k)."""
    n = 2 * max_repli
    replica = np.asarray(replica, dtype=np.int64)
    k = replica % n
    j = (replica // n) % n
    i = replica // (n * n)
    return np.stack([i, j, k], axis=-1) - max_repli


def _read_first(f, candidates):
    """Read the first dataset in `candidates` that exists (SOAP group names changed
    between versions, e.g. BoundSubhaloProperties -> BoundSubhalo)."""
    for name in candidates:
        if name in f:
            return f[name][:]
    raise KeyError(f"None of {candidates} found in {f.filename}")


def load_snapshot_halos(param, halo_file, preamble=""):
    """
    Read one FLAMINGO SOAP-HBT snapshot catalogue and apply the selection cuts.

    Returns a structured array with positions in comoving kpc/h and masses in Msun/h.
    This is exactly the population the crossing builder places in the lightcone, so it is
    also what the snapshot side of the HMF validation uses.
    """
    if param.files.halofile_format != 'FLAMINGO-SOAP-HBT':
        raise ValueError(preamble + f"unsupported halo format {param.files.halofile_format}")

    # SOAP/HBTplus halo catalogue (SWIFT-style HDF5). All physical quantities are
    # stored h-free (h-scale exponent = 0 on every dataset) -- comoving Mpc for
    # lengths, 1e10 Msun for masses -- so *h converts to the Mpc/h, Msun/h
    # convention used throughout.
    #
    # Centrals (InputHalos/IsCentral == 1) are always kept, with SO/200_crit
    # quantities as before (cut on M200c >= Mhalo_min, N200c >= Nmin_per_halo).
    #
    # Satellites (param.code.include_satellites = True) are added on top. SOAP
    # only computes SO properties for centrals (SO/* is zero for satellites), so
    # satellites are described by their HBT *bound* quantities instead:
    #   Mvir   <- BoundSubhalo/TotalMass                (Msun/h)
    #   Nvir   <- BoundSubhalo/NumberOfDarkMatterParticles
    #   v      <- BoundSubhalo/CentreOfMassVelocity
    #   rvir, cvir <- NaN   (no SO radius/concentration: downstream profile
    #                        code must branch on is_central, NaN makes misuse loud)
    #   IDhost <- HaloCatalogueIndex of the top-level central (SOAP/HostHaloIndex)
    # and are cut on Mbound >= Msub_min, Nbound >= Nmin_per_subhalo.
    # Two extra columns are stored for every object:
    #   Mbound     -- bound mass (centrals too), a single consistent mass definition
    #                 for (sub)halo mass functions
    #   is_central -- 1 for centrals, 0 for satellites
    names = "ID,IDhost,Mvir,Nvir,x,y,z,vx,vy,vz,rvir,b_ov_a,c_ov_a,Eax,Eay,Eaz,Ebx,Eby,Ebz,Ecx,Ecy,Ecz,cvir"
    dtype = [(n, np.int64 if n in ('ID', 'IDhost', 'Nvir') else np.float64) for n in names.split(',')]
    dtype += [("Mbound", np.float64), ("is_central", np.int8)]
    # HBT TrackId (persistent across snapshots), needed by the crossing-based builder
    # (prep_halocone_crossing.py). Off by default so older soap_subset files still work.
    with_trackid = getattr(param.code, 'with_trackid', False)
    if with_trackid:
        dtype += [("TrackId", np.int64)]

    h_hubble = getattr(param.cosmo, 'h0')
    include_sats = getattr(param.code, 'include_satellites', False)
    Msub_min = getattr(param.code, 'Msub_min', param.code.Mhalo_min)            # Msun/h, bound mass
    Nmin_sub = getattr(param.sim, 'Nmin_per_subhalo', param.sim.Nmin_per_halo)  # bound DM particles

    BOUND = ('BoundSubhalo/', 'BoundSubhaloProperties/')

    with h5py.File(halo_file, 'r') as f:
        iscentral = f['InputHalos/IsCentral'][:] == 1
        m200c = f["SO/200_crit/TotalMass"][:] * 1.0e10 * h_hubble  # -> Msun/h
        n200c = f["SO/200_crit/NumberOfDarkMatterParticles"][:]
        mbound = _read_first(f, [g + 'TotalMass' for g in BOUND]) * 1.0e10 * h_hubble
        nbound = _read_first(f, [g + 'NumberOfDarkMatterParticles' for g in BOUND])

        sel_cen = iscentral & (n200c >= param.sim.Nmin_per_halo) & (m200c >= param.code.Mhalo_min)
        if include_sats:
            sel_sat = (~iscentral) & (nbound >= Nmin_sub) & (mbound >= Msub_min)
        else:
            sel_sat = np.zeros_like(iscentral)

        # Keep only satellites whose host central itself passes the central cut
        if include_sats and getattr(param.code, 'require_host_above_cut', False):
            host_all = f["SOAP/HostHaloIndex"][:]
            idx = np.flatnonzero(sel_sat)
            hi = host_all[idx]
            no_host = ~sel_cen[hi]
            sel_sat[idx[no_host]] = False
        n_cen, n_sat = int(sel_cen.sum()), int(sel_sat.sum())

        h = np.zeros(n_cen + n_sat, dtype=dtype)
        C, S = slice(0, n_cen), slice(n_cen, None)

        cat_index = f["InputHalos/HaloCatalogueIndex"][:]
        pos_all = f["InputHalos/HaloCentre"][:] * h_hubble * 1000.0  # Mpc (comoving) -> kpc/h

        # centrals
        h["ID"][C] = cat_index[sel_cen]
        h["IDhost"][C] = 0  # all are host halos by construction
        h["Mvir"][C] = m200c[sel_cen]
        h["Nvir"][C] = n200c[sel_cen]
        vel = f["SO/200_crit/CentreOfMassVelocity"][:][sel_cen]  # peculiar velocity, km/s
        h["vx"][C], h["vy"][C], h["vz"][C] = vel[:, 0], vel[:, 1], vel[:, 2]
        h["rvir"][C] = f["SO/200_crit/SORadius"][:][sel_cen] * h_hubble * 1000.0  # Mpc (comoving) -> kpc/h
        h["cvir"][C] = f["SO/200_crit/Concentration"][:][sel_cen]
        h["Mbound"][C] = mbound[sel_cen]
        h["is_central"][C] = 1

        # satellites
        if n_sat > 0:
            host_idx = f["SOAP/HostHaloIndex"][:][sel_sat]  # row index of top-level central, -1 for centrals
            if np.any(host_idx < 0) or not np.all(iscentral[host_idx]):
                raise ValueError(preamble + "SOAP/HostHaloIndex does not point to centrals for all "
                                 "satellites -- check the field semantics in this SOAP version.")
            h["ID"][S] = cat_index[sel_sat]
            h["IDhost"][S] = cat_index[host_idx]
            h["Mvir"][S] = mbound[sel_sat]
            h["Nvir"][S] = nbound[sel_sat]
            vel = _read_first(f, [g + 'CentreOfMassVelocity' for g in BOUND])[sel_sat]
            h["vx"][S], h["vy"][S], h["vz"][S] = vel[:, 0], vel[:, 1], vel[:, 2]
            h["rvir"][S] = np.nan
            h["cvir"][S] = np.nan
            h["Mbound"][S] = mbound[sel_sat]
            h["is_central"][S] = 0

        sel = np.concatenate([np.flatnonzero(sel_cen), np.flatnonzero(sel_sat)])
        h["x"], h["y"], h["z"] = pos_all[sel, 0], pos_all[sel, 1], pos_all[sel, 2]
        if with_trackid:
            if "InputHalos/HBTplus/TrackId" not in f:
                raise KeyError(preamble + "InputHalos/HBTplus/TrackId missing -- run add_track_info.py "
                               "on the soap_subset files first")
            h["TrackId"] = f["InputHalos/HBTplus/TrackId"][:][sel]

    # the per-snapshot ID offset (ID + i_file*ID_STRIDE) must not collide, otherwise
    # satellite -> host matching through IDhost silently breaks
    if len(h) and h["ID"].max() >= ID_STRIDE:
        raise ValueError(preamble + f"HaloCatalogueIndex up to {h['ID'].max()} >= ID_STRIDE={ID_STRIDE}; "
                         "increase ID_STRIDE in prep_halocone_crossing.py")

    print(f"number of halos: {len(h)} ({n_cen} centrals, {n_sat} satellites)")
    if n_cen:
        print(f"Central M200c range: {np.min(h['Mvir'][C]):.2e} - {np.max(h['Mvir'][C]):.2e} Msun/h")
    if n_sat:
        print(f"Satellite Mbound range: {np.min(h['Mvir'][S]):.2e} - {np.max(h['Mvir'][S]):.2e} Msun/h")

    if getattr(param.code, 'lean_output', True):
        lean_dtype = [("ID", np.int64), ("IDhost", np.int64), ("Mvir", np.float32), ("Nvir", np.int32),
                      ("x", np.float64), ("y", np.float64), ("z", np.float64),
                      ("vx", np.float32), ("vy", np.float32), ("vz", np.float32),
                      ("rvir", np.float32), ("cvir", np.float32),
                      ("Mbound", np.float32), ("is_central", np.int8)]
        if with_trackid:
            lean_dtype += [("TrackId", np.int64)]
        h_lean = np.zeros(len(h), dtype=lean_dtype)
        for name, _ in lean_dtype:
            h_lean[name] = h[name]
        h = h_lean

    return h


def build_default_param():
    """Parameter object with the FLAMINGO L1_m9_DMO defaults; the run script sets every value."""
    return NS(
        cosmo=NS(Om=0.306, Ob=0.0486, h0=0.681, s8=0.807, ns=0.967),
        sim=NS(Lbox=681.0000003288388, Nmin_per_halo=100, Nmin_per_subhalo=100),
        code=NS(halo_lc_buffer=1e-9, Mhalo_min=8e12, Msub_min=1e12, include_satellites=True,
                require_host_above_cut=True, observer_pos=(0.0, 0.0, 0.0), lean_output=True,
                with_trackid=False),
        files=NS(halofile_format="FLAMINGO-SOAP-HBT"),
    )


SHELL_DTYPE = np.dtype([("shell_id", np.int32), ("lower_z", np.float32), ("upper_z", np.float32),
                        ("lower_com", np.float32), ("upper_com", np.float32), ("shell_com", np.float32)])


def read_outer_tracks(halo_file_out, h_hubble):
    """TrackId and position [kpc/h] at the outer snapshot, sorted by TrackId."""
    with h5py.File(halo_file_out, "r") as f:
        if "Tracks/TrackId" in f:
            tid = f["Tracks/TrackId"][:]
            pos = f["Tracks/HaloCentre"][:]
        else:
            print("WARNING: no Tracks/ group in", halo_file_out, "-- using its own rows only "
                  "(objects below the cuts at the outer snapshot will not be interpolated). "
                  "Run add_track_info.py.")
            tid = f["InputHalos/HBTplus/TrackId"][:]
            pos = f["InputHalos/HaloCentre"][:]
    pos = pos.astype(np.float64) * h_hubble * 1000.0
    o = np.argsort(tid)
    return tid[o], pos[o]


def match_displacement(tid_in, pos_in, tid_out_sorted, pos_out_sorted, Lbox_kpc):
    """Periodic minimal-image displacement R0 - R1 [kpc/h]; 0 where there is no match."""
    k = np.searchsorted(tid_out_sorted, tid_in)
    k = np.minimum(k, max(len(tid_out_sorted) - 1, 0))
    ok = (len(tid_out_sorted) > 0) & (tid_out_sorted[k] == tid_in)
    disp = np.zeros_like(pos_in)
    d = pos_out_sorted[k[ok]] - pos_in[ok]
    disp[ok] = d - Lbox_kpc * np.round(d / Lbox_kpc)
    return disp, ok


def get_single_shell_crossing(param, halo_file_in, halo_file_out, z_boundary, shell_id, max_repli=7):
    cosmo = FlatLambdaCDM(H0=100 * param.cosmo.h0, Ob0=param.cosmo.Ob, Om0=param.cosmo.Om)
    shell_com, r_lo, r_hi = get_shell_cov_dist(z_boundary, cosmo)            # Mpc/h
    shell_data = np.array([(shell_id, z_boundary[0], z_boundary[1], r_lo, r_hi, shell_com)], dtype=SHELL_DTYPE)
    preamble = f"Shell {shell_id}: "
    print(f"\n{preamble}{z_boundary[0]:.3f} < z < {z_boundary[1]:.3f}, r = {r_lo:.1f}-{r_hi:.1f} Mpc/h")
    print(f"  inner: {os.path.basename(halo_file_in)}   outer: {os.path.basename(halo_file_out)}")

    param.code.with_trackid = True
    h = load_snapshot_halos(param, halo_file_in, preamble=preamble)
    h_hubble = param.cosmo.h0
    L = param.sim.Lbox * 1000.0                                              # kpc/h

    pos1 = np.c_[h["x"], h["y"], h["z"]].astype(np.float64)                  # box coords, kpc/h
    tid_o, pos_o = read_outer_tracks(halo_file_out, h_hubble)
    disp, matched = match_displacement(h["TrackId"], pos1, tid_o, pos_o, L)
    del tid_o, pos_o
    dnorm = np.linalg.norm(disp, axis=1)
    print(f"  matched {matched.mean()*100:.2f}% of {len(h)}; |R0-R1| median {np.median(dnorm[matched]):.0f}, "
          f"99.9% {np.percentile(dnorm[matched], 99.9):.0f}, max {dnorm.max():.0f} kpc/h")
    if dnorm.max() > 0.05 * L:
        print("  WARNING: very large displacements -- check the TrackId matching")

    # IDs unique per shell; satellites keep the link to their host
    h["ID"] = h["ID"] + shell_id * ID_STRIDE
    sat = h["is_central"] == 0
    h["IDhost"][sat] = h["IDhost"][sat] + shell_id * ID_STRIDE

    # recentre on the observer (periodic), observer at the origin
    obs = np.asarray(getattr(param.code, "observer_pos", (0.0, 0.0, 0.0)), dtype=float) * 1000.0
    R1 = np.mod(pos1 - obs, L) / 1000.0                                      # Mpc/h
    D = disp / 1000.0
    del pos1, disp

    max_disp = dnorm.max() / 1000.0
    offsets = get_offset_indices(boxsize=param.sim.Lbox, min_val=r_lo, max_val=r_hi, max_repli=max_repli,
                                 buffer=max_disp + param.code.halo_lc_buffer)
    print(f"  {len(offsets)} box replicas")

    out_dtype = h.dtype.descr + [("shell_id", np.int32), ("halo_buffer", np.int8), ("replica", np.int32),
                                 ("v_cross", np.float32)]
    pieces, n_kept = [], 0
    for off in offsets:
        X1 = R1 + np.asarray(off) * param.sim.Lbox
        r1 = np.linalg.norm(X1, axis=1)
        near = (r1 >= r_lo - max_disp) & (r1 < r_hi + max_disp)
        if not near.any():
            continue
        idx = np.flatnonzero(near)
        X1 = X1[idx]; r1 = r1[idx]
        X0 = X1 + D[idx]
        r0 = np.linalg.norm(X0, axis=1)
        v = (r_hi - r0) / ((r_hi - r_lo) + (r1 - r0))
        Xc = (1.0 - v)[:, None] * X0 + v[:, None] * X1
        rc = np.linalg.norm(Xc, axis=1)
        keep = (v >= 0) & (v < 1) & (rc >= r_lo) & (rc < r_hi)
        if not keep.any():
            continue
        sel = idx[keep]
        rel = np.zeros(len(sel), dtype=out_dtype)
        hs = h[sel]
        for name in h.dtype.names:
            rel[name] = hs[name]
        rel["x"], rel["y"], rel["z"] = (Xc[keep] * 1000.0).T                 # kpc/h, observer-centred
        rel["shell_id"] = shell_id
        rel["halo_buffer"] = 0
        rel["replica"] = encode_replica(off, max_repli)
        rel["v_cross"] = v[keep]
        pieces.append(rel)
        n_kept += len(sel)

    out = np.concatenate(pieces) if pieces else np.zeros(0, dtype=out_dtype)
    nc = int((out["is_central"] == 1).sum())
    print(f"  kept {n_kept} objects ({nc} centrals, {n_kept - nc} satellites); "
          f"V_shell/Lbox^3 = {4/3*np.pi*(r_hi**3 - r_lo**3)/param.sim.Lbox**3:.3f}")
    return out, shell_data


def prep_lightcone_halos_crossing(param, halo_file_pairs, shells_z, shell_ids=None, max_repli=7):
    """halo_file_pairs[k] = (inner file at shells_z[k][0], outer file at shells_z[k][1])."""
    assert len(halo_file_pairs) == len(shells_z)
    shell_ids = list(range(len(shells_z))) if shell_ids is None else list(shell_ids)
    halos, shells = [], []
    for sid, (fin, fout), zb in zip(shell_ids, halo_file_pairs, shells_z):
        hh, ss = get_single_shell_crossing(param, fin, fout, zb, sid, max_repli=max_repli)
        halos.append(hh); shells.append(ss)
    n = sum(len(x) for x in halos)
    halo_data = np.empty(n, dtype=halos[0].dtype)
    p = 0
    halos.reverse()
    while halos:
        x = halos.pop()
        halo_data[p:p + len(x)] = x
        p += len(x)
        del x
    return halo_data, np.concatenate(shells)

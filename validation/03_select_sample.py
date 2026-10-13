import numpy as np
import argparse
import os

from ksz_utils import lightcone_pos_to_sky, radial_velocity

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BOX = "L1_m9"               # box/resolution grouping directory
SIM = "L1_m9"               # run inside it; set by --sim
LIGHTCONE = "lightcone0"
BASE = f"FLAMINGO/{BOX}/{SIM}"

M_STAR = "ExclusiveSphere/50kpc/StellarMass"
VELOCITY = "ExclusiveSphere/50kpc/StellarCentreOfMassVelocity"
SO_MASS = "SO/500_crit/TotalMass"
IS_CENTRAL = "InputHalos/IsCentral"
HOST_FOF = "InputHalos/HBTplus/HostFOFId"

TARGETS = {
    "cmass": dict(snapshots=(66,), z_min=0.525, z_max=0.575, log_m500_target=13.34),
    "lowz":  dict(snapshots=(71,), z_min=0.275, z_max=0.325, log_m500_target=13.53),
}

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def open_remote(path):
    import hdfstream
    return hdfstream.open("cosma", "/")[path]


def open_local(path):
    import h5py
    return h5py.File(path, "r")


def assign_host_mass(host_fof, is_central, m500):
    """
    Give every subhalo the M500c of the central of its own FoF group.
    """
    # Step 1: Build the lookup table from centrals only.
    # Each FoF group has exactly one central, so the ids are unique.
    ids = host_fof[is_central]
    vals = m500[is_central]
    order = np.argsort(ids, kind="stable")
    ids_s, vals_s = ids[order], vals[order]

    # Step 2: For every sunhalo, binary search its HostFOFId into the
    # sorted central ids.
    idx = np.searchsorted(ids_s, host_fof)
    idx = np.clip(idx, 0, max(ids_s.size - 1, 0))

    # Step 3: verify that the values match.
    matched = (ids_s.size > 0) & (ids_s[idx] == host_fof)

    out = np.zeros_like(m500)
    out[matched] = vals_s[idx[matched]]
    return out, matched


def load_soap(snap, remote, soap_dir):
    """
    Read the SOAP columns needed for one snapshot.
    """
    name = f"halo_properties_{snap:04d}.hdf5"
    if remote:
        f = open_remote(f"{BASE}/SOAP-HBT/{name}")
    else:
        f = open_local(os.path.join(soap_dir, name))

    def node_for(key):
        node = f
        for part in key.split("/"):
            try:
                node = node[part]
            except Exception as exc:
                raise KeyError(
                    f"{key!r} not found in {name}. At {part!r}; "
                    f"available here: {sorted(list(node))[:30]}"
                ) from exc
        return node

    def col(key):
        return np.asarray(node_for(key)[...])

    def show_units(key):
        """
        Print the unit metadata.
        """
        try:
            attrs = dict(node_for(key).attrs)
        except Exception:
            return
        keep = {k: np.atleast_1d(v)[0] for k, v in attrs.items()
                if any(s in k for s in ("a-scale", "h-scale", "Conversion",
                                        "Expression", "U_L", "U_t"))}
        if keep:
            print(f"    units of {key}:")
            for k, v in sorted(keep.items()):
                print(f"      {k} = {v}")

    print(f"  reading SOAP columns from {name} ...", flush=True)
    mstar = col(M_STAR).astype(np.float64) * 1e10  # -> Msun
    m500 = col(SO_MASS).astype(np.float64) * 1e10  # -> Msun
    central = col(IS_CENTRAL).astype(bool)
    host = col(HOST_FOF)
    vel = col(VELOCITY).astype(np.float64)  # km/s

    # SOAP velocities carry a-scale exponent 1: physical peculiar velocity =
    # a * stored. The ratio of the two conversion factors is that a-factor.
    vattrs = dict(node_for(VELOCITY).attrs)
    k_phys = "Conversion factor to physical CGS (including cosmological corrections)"
    k_com = "Conversion factor to CGS (not including cosmological corrections)"
    if k_phys in vattrs and k_com in vattrs:
        afac = float(np.atleast_1d(vattrs[k_phys])[0]) / float(np.atleast_1d(vattrs[k_com])[0])
        vel *= afac
        print(f"    velocities scaled by a = {afac:.4f} to physical km/s")
    else:
        print("    WARNING: no conversion factors on the velocity - left as stored")

    show_units(VELOCITY)
    show_units(M_STAR)

    host_m500, matched = assign_host_mass(host, central, m500)
    host_vel, _ = assign_host_mass(host, central, vel)  # the central's velocity
    print(f"    {mstar.size} subhaloes, {central.sum()} centrals, "
          f"{(~matched).sum()} with no host match")
    return dict(mstar=mstar, m500_host=host_m500, central=central,
                vel=vel, host_vel=host_vel, matched=matched)


def scan_lightcone(snap, soap, keep, halo_dir, remote, chunk):
    """
    Row-scan one halo lightcone file, keeping only pre-selected subhaloes.
    """
    name = f"lightcone_halos_{snap:04d}.hdf5"
    if remote:
        f = open_remote(f"{BASE}/halo_lightcone/{LIGHTCONE}/{name}")
    else:
        f = open_local(os.path.join(halo_dir, name))

    n = f["Lightcone/Redshift"].shape[0]
    n_soap = soap["mstar"].size
    print(f"  scanning {name}: {n} rows in chunks of {chunk}", flush=True)

    pos_l, z_l, si_l = [], [], []
    kept = 0
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        si = np.asarray(f["InputHalos/SOAPIndex"][a:b])
        good = (si >= 0) & (si < n_soap)
        m = np.zeros(si.shape, dtype=bool)
        m[good] = keep[si[good]]
        if not m.any():
            continue
        pos_l.append(np.asarray(f["Lightcone/HaloCentre"][a:b])[m])
        z_l.append(np.asarray(f["Lightcone/Redshift"][a:b])[m])
        si_l.append(si[m])
        kept += int(m.sum())
    print(f"    kept {kept} of {n} rows after the stellar-mass pre-cut")

    if kept == 0:
        return None
    return (np.concatenate(pos_l), np.concatenate(z_l), np.concatenate(si_l))


# --------------------------------------------------------------------------
# Construct sample
# --------------------------------------------------------------------------

def main():
    global SIM, BASE, VELOCITY
    p = argparse.ArgumentParser()
    p.add_argument("--sample", choices=list(TARGETS), default="cmass")
    p.add_argument("--sim", default=SIM,
                   help="run inside FLAMINGO/L1_m9/: L1_m9 (fiducial), "
                        "fgas-4sigma, fgas-8sigma, Jet, ... Note the stellar "
                        "mass cut is re-solved per run, as in McCarthy+24")
    p.add_argument("--snapshots", type=int, nargs="+", default=None,
                   help="override the snapshot list for this sample")
    p.add_argument("--halo-dir", default=f"/mnt/extraspace/{os.environ.get('USER','')}/flamingo",
                   help="directory holding the downloaded lightcone_halos_*.hdf5")
    p.add_argument("--remote-halos", action="store_true",
                   help="stream the halo lightcones instead of reading local copies")
    p.add_argument("--soap-dir", default=None,
                   help="local SOAP directory; omit to stream SOAP (recommended)")
    p.add_argument("--precut", type=float, default=10.8,
                   help="log10 M*/Msun pre-cut applied before the bisection")
    p.add_argument("--statistic", choices=["log-of-mean", "mean-of-log"],
                   default="log-of-mean",
                   help="which reading of 'mean halo mass' to solve the cut on")
    p.add_argument("--halo-mass-bin", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="select haloes with LO <= log10(M500c/Msun) < HI "
                        "instead of applying a stellar-mass cut. No galaxy "
                        "definition enters.")
    p.add_argument("--include-satellites", action="store_true",
                   help="halo-mass mode: also keep satellites, each carrying "
                        "its host's M500c. Default is centrals only, so each "
                        "halo enters the stack once.")
    p.add_argument("--velocity-field", default=VELOCITY,
                   help="SOAP velocity to project onto the line of sight. "
                        "Default is the 50 kpc stellar centre-of-mass velocity, "
                        "matching the aperture the stellar mass uses. For "
                        "halo-mass stacks BoundSubhalo/CentreOfMassVelocity is "
                        "the more natural choice - confirm the name with "
                        "01_explore_flamingo.py before switching.")
    p.add_argument("--mstar-cut", type=float, default=None,
                   help="apply this log10(M*/Msun) minimum directly instead of "
                        "solving for the target halo mass (McCarthy+24 fit "
                        "theirs to lensing: L1_m9 CMASS ~11.2). A low value "
                        "such as 10.9 lets 04 --min-mstar choose cuts later")
    p.add_argument("--chunk", type=int, default=5_000_000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    SIM = args.sim
    BASE = f"FLAMINGO/{BOX}/{SIM}"
    VELOCITY = args.velocity_field
    halo_mode = args.halo_mass_bin is not None
    print(f"simulation: {BASE}")
    print(f"velocity:   {VELOCITY}")

    cfg = TARGETS[args.sample]
    snaps = args.snapshots if args.snapshots else list(cfg["snapshots"])
    if halo_mode:
        lo_m, hi_m = args.halo_mass_bin
        print(f"selection:  haloes with {lo_m} <= log10(M500c/Msun) < {hi_m}, "
              f"{'centrals + satellites' if args.include_satellites else 'centrals only'}")
        print(f"            snapshots {snaps}, "
              f"{cfg['z_min']} < z < {cfg['z_max']}")
    else:
        print(f"selection:  {args.sample}-like, snapshots {snaps}, "
              f"{cfg['z_min']} < z < {cfg['z_max']}, "
              f"target mean log10 M500c = {cfg['log_m500_target']}")

    pos, z, mstar, m500, central, vel, hvel = [], [], [], [], [], [], []

    for snap in snaps:
        print(f"\n--- snapshot {snap:04d} ---")
        soap = load_soap(snap, args.soap_dir is None, args.soap_dir)
        if halo_mode:
            # Pre-cut on host mass is exact here, not approximate: nothing
            # inside the bin can have a host mass below its lower edge.
            keep = (soap["m500_host"] >= 10 ** lo_m) & soap["matched"]
            if not args.include_satellites:
                keep &= soap["central"]
            print(f"    {keep.sum()} pass the pre-cut log10 M500c >= {lo_m}"
                  f"{'' if args.include_satellites else ' (centrals only)'}")
        else:
            keep = (soap["mstar"] > 10 ** args.precut) & soap["matched"]
            print(f"    {keep.sum()} subhaloes pass the pre-cut "
                  f"log10 M* > {args.precut}")

        got = scan_lightcone(snap, soap, keep, args.halo_dir,
                             args.remote_halos, args.chunk)
        if got is None:
            continue
        p_, z_, si_ = got
        pos.append(p_)
        z.append(z_)
        mstar.append(soap["mstar"][si_])
        m500.append(soap["m500_host"][si_])
        central.append(soap["central"][si_])
        vel.append(soap["vel"][si_])
        hvel.append(soap["host_vel"][si_])
        del soap

    if not pos:
        raise SystemExit("no haloes selected - lower --precut or check the paths")

    pos = np.concatenate(pos)
    z = np.concatenate(z)
    mstar = np.concatenate(mstar)
    m500 = np.concatenate(m500)
    central = np.concatenate(central)
    vel = np.concatenate(vel)
    hvel = np.concatenate(hvel)
    print(f"\ncombined: {z.size} haloes, z = {z.min():.4f} - {z.max():.4f}")

    in_z = (z >= cfg["z_min"]) & (z < cfg["z_max"])
    print(f"{in_z.sum()} in the redshift bin")

    # ------------------------------------------------------------------
    # Halo mass mode. No stellar masses are required.
    # ------------------------------------------------------------------
    if halo_mode:
        sel = in_z & (m500 >= 10 ** lo_m) & (m500 < 10 ** hi_m)
        if not args.include_satellites:
            sel &= central
        cut = float("nan")

        print(f"\nhalo-mass selection  {lo_m} <= log10(M500c/Msun) < {hi_m}")
        print(f"  N                  = {sel.sum()}")
        if sel.sum() == 0:
            raise SystemExit("no haloes in that mass bin - check the edges")
        print(f"  log10<M500c>       = {np.log10(m500[sel].mean()):.3f}")
        print(f"  <log10 M500c>      = {np.log10(m500[sel]).mean():.3f}")
        print(f"  satellite fraction = {1 - central[sel].mean():.3f}")
        print(f"  fraction with M*>0 = {(mstar[sel] > 0).mean():.3f}")

    # ------------------------------------------------------------------
    # Stellar mass mode. Solve for the stellar-mass cut reproducing the mean halo mass
    # ------------------------------------------------------------------
    else:
        in_shell = in_z & (mstar > 0) & (m500 > 0)
        print(f"{in_shell.sum()} of those have M* > 0 and a host mass")
        if args.mstar_cut is not None:
            cut = args.mstar_cut
            if cut < args.precut:
                print(f"Warning: --mstar-cut {cut} is below --precut "
                      f"{args.precut}; objects in between were never read")
            sel = in_shell & (mstar > 10 ** cut)
            report_selection(sel, cut, m500, central)
        else:
            sel, cut = solve_stellar_cut(in_shell, mstar, m500, central, cfg, args)

    # Geometry (common to both modes)
    unit, r_comov = lightcone_pos_to_sky(pos[sel])
    v_r = radial_velocity(vel[sel], unit)
    v_r_host = radial_velocity(hvel[sel], unit)
    print(f"  v_r rms            = {np.sqrt((v_r**2).mean()):.1f} km/s")
    print(f"  v_r mean           = {v_r.mean():.2f} km/s  (should be ~0)")

    tag = "-".join(f"{s:04d}" for s in snaps)
    if halo_mode:
        label = (f"halo{lo_m:g}-{hi_m:g}"
                 + ("sat" if args.include_satellites else ""))
    else:
        label = args.sample
    out = args.out or f"sample_{label}_{SIM}_snap{tag}.npz"

    with np.errstate(divide="ignore"):
        log_mstar = np.log10(np.maximum(mstar[sel], 1.0))
    np.savez(out,
             unit_vec=unit, r_comov=r_comov, v_r_kms=v_r,
             v_r_host_kms=v_r_host, z=z[sel],
             log_mstar=log_mstar, log_m500=np.log10(m500[sel]),
             is_central=central[sel], mstar_cut=cut,
             z_min=cfg["z_min"], z_max=cfg["z_max"],
             selection=label, sim=SIM, velocity_field=VELOCITY,
             halo_mass_bin=(np.array(args.halo_mass_bin) if halo_mode
                            else np.array([np.nan, np.nan])))
    print(f"\nwrote {out}")


def report_selection(sel, cut, m500, central):
    """
    Print both readings of 'mean halo mass' for a fixed stellar-mass cut.
    """
    print(f"\nstellar-mass cut     log10(M*/Msun) = {cut:.3f}")
    print(f"  N                  = {sel.sum()}")
    print(f"  log10<M500c>       = {np.log10(m500[sel].mean()):.3f}")
    print(f"  <log10 M500c>      = {np.log10(m500[sel]).mean():.3f}")
    print(f"  satellite fraction = {1 - central[sel].mean():.3f}")


def solve_stellar_cut(in_shell, mstar, m500, central, cfg, args):
    """
    Bisect on the stellar-mass cut to hit the target halo mass.
    """
    target = cfg["log_m500_target"]

    def stats(cut):
        sel = in_shell & (mstar > 10 ** cut)
        if sel.sum() < 100:
            return np.nan, np.nan, sel
        return (np.log10(m500[sel].mean()),
                np.log10(m500[sel]).mean(),
                sel)

    pick = (lambda a, b: a) if args.statistic == "log-of-mean" else (lambda a, b: b)

    lo_v = pick(*stats(args.precut)[:2])
    hi_v = pick(*stats(12.0)[:2])
    print(f"\nreachable mean log10 M500c over cuts [{args.precut}, 12.0]: "
          f"{lo_v:.3f} -> {hi_v:.3f}   (target {target}, "
          f"statistic '{args.statistic}')")

    if not (np.nanmin([lo_v, hi_v]) <= target <= np.nanmax([lo_v, hi_v])):
        print("Warning: the target lies outside that range, so no cut can hit "
              "it. The bisection below will stick at an endpoint. Check the "
              "statistic, the redshift bin, and that host masses are populated.")

    lo, hi = args.precut, 12.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        a, b, _ = stats(mid)
        val = pick(a, b)
        if np.isnan(val):
            hi = mid
        elif val < target:
            lo = mid
        else:
            hi = mid
    cut = 0.5 * (lo + hi)
    a, b, sel = stats(cut)

    print(f"\nstellar-mass cut     log10(M*/Msun) = {cut:.3f}")
    print(f"  N                  = {sel.sum()}")
    print(f"  log10<M500c>       = {a:.3f}")
    print(f"  <log10 M500c>      = {b:.3f}      (target {target})")
    print(f"  satellite fraction = {1 - central[sel].mean():.3f}")
    if abs(pick(a, b) - target) > 0.05:
        print("Warning: did not converge onto the target - see the range above.")
    return sel, cut


if __name__ == "__main__":
    main()

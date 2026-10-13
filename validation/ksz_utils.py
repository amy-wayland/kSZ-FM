import re
import numpy as np
import healpy as hp

T_CMB_K = 2.7255
SR_TO_ARCMIN2 = (180.0 * 60.0 / np.pi) ** 2
C_KMS = 299792.458

# --------------------------------------------------------------------------
# Locating shell files
# --------------------------------------------------------------------------

def _pick_shell_entry(entries, shell_nr):
    """
    Selects the relevant list of names inside the shells directory;
    `shell_nr' is the integer shell wanted.
    """
    exact = f"shell_{shell_nr}"  # name used by the archive, e.g. shell_10
    if exact in entries:
        return exact
    pat = re.compile(rf"(^|[._])shell_{shell_nr}([._]|$)")
    hits = sorted(e for e in entries if pat.search(e))
    if not hits:
        raise KeyError(f"no entry for shell {shell_nr}. Saw: {sorted(entries)[:12]} ...")
    return hits[0]


def find_shell_files(ls, base, shell_nr, lightcone="lightcone0",
                     nside_dir="nside_4096"):
    """
    Return full paths of the HDF5 file holding one shell.

    `ls` is a callable taking a path and returning the names inside it, so the
    same code works for a local directory (os.listdir) and for a remote
    hdfstream directory (list(root[path])).
    """
    shells_dir = f"{base}/healpix_maps/{nside_dir}/{lightcone}_shells"
    entry = _pick_shell_entry(list(ls(shells_dir)), shell_nr)
    path = f"{shells_dir}/{entry}"

    if entry.endswith(".hdf5"):
        return [path]

    inner = list(ls(path))
    files = [f"{path}/{n}" for n in sorted(inner) if n.endswith(".hdf5")]
    if not files:
        raise KeyError(f"no .hdf5 files under {path}. Contents: {sorted(inner)[:12]}")
    return files


# --------------------------------------------------------------------------
# CAP filter
# --------------------------------------------------------------------------

def cap_profile(sky_map, nside, vec, theta_d_rad, pix_area=None, area_norm=False):
    """
    Compensated aperture photometry profile at one position.

    W = +1 for theta < theta_d, -1 for theta_d <= theta < sqrt(2) theta_d, 0 else.
    The annulus has the same solid angle as the disc, so a constant background
    integrates to zero.

    Parameters
    ----------
    sky_map : (npix,) array, RING ordering, units of temperature (e.g. uK)
    vec     : (3,) unit vector towards the object
    theta_d_rad : (n_ap,) array of disc radii in radians

    Returns
    -------
    (n_ap,) array in <map units> * steradian.  Multiply by SR_TO_ARCMIN2 for
    uK arcmin^2.
    """
    theta_d_rad = np.atleast_1d(theta_d_rad)
    if pix_area is None:
        pix_area = hp.nside2pixarea(nside)

    # One disc query at the largest radius, then bin by angular separation.
    r_max = np.sqrt(2.0) * theta_d_rad.max()
    pix = hp.query_disc(nside, vec, r_max, inclusive=True)
    if pix.size == 0:
        return np.zeros(theta_d_rad.size)

    px, py, pz = hp.pix2vec(nside, pix)  # unit vector of each pixel centre
    cosang = np.clip(vec[0] * px + vec[1] * py + vec[2] * pz, -1.0, 1.0)
    ang = np.arccos(cosang)  # angular separation of each pixel from the galaxy
    vals = np.asarray(sky_map[pix], dtype=np.float64)

    out = np.empty(theta_d_rad.size)
    for i, td in enumerate(theta_d_rad):
        inner = ang < td
        outer = (ang >= td) & (ang < np.sqrt(2.0) * td)
        if area_norm:
            a = np.pi*td**2
            v_in = vals[inner].mean() if inner.any() else vals[np.argmin(ang)]
            v_out = vals[outer].mean() if outer.any() else v_in
            out[i] = (v_in - v_out) * a
        else:
            out[i] = (vals[inner].sum() - vals[outer].sum()) * pix_area
    return out


def cap_profiles_catalogue(sky_map, nside, vecs, theta_d_rad, progress_every=20000, area_norm=False):
    """
    CAP profile for every object in a catalogue. 
    Returns (n_obj, n_ap).
    """
    pix_area = hp.nside2pixarea(nside)
    n = len(vecs)
    out = np.empty((n, len(theta_d_rad)))
    for i in range(n):
        out[i] = cap_profile(sky_map, nside, vecs[i], theta_d_rad, pix_area, area_norm)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    CAP {i + 1}/{n}", flush=True)
    return out

# --------------------------------------------------------------------------
# Velocity-weighted stacking estimator
# --------------------------------------------------------------------------

def velocity_weighted_stack(T_cap, v_r_kms, sigma=None, r_v=1.0, v_rms_kms=None):
    """
    McCarthy+24 equation for the stacked kSZ signal.

        T_hat(theta_d) = -(1/r_v) (v_rms/c) * sum_i T_i (v_i/c) / sigma_i^2
                                            / sum_i (v_i/c)^2 / sigma_i^2

    For simulations use true radial velocities: r_v = 1, sigma_i = 1.
    Sign convention: v_r > 0 means receding, which gives a kSZ decrement, so
    a physical signal comes out positive from this estimator.
    """
    T_cap = np.atleast_2d(T_cap)  # accept a single profile or a stack of them
    v = np.asarray(v_r_kms, dtype=np.float64) / C_KMS  # units of v/c
    w = np.ones_like(v) if sigma is None else 1.0 / np.asarray(sigma) ** 2  # inverse-variance weights
    if v_rms_kms is None:
        v_rms_kms = np.sqrt(np.mean(np.asarray(v_r_kms, dtype=np.float64) ** 2))  # rms radial velocity

    num = np.sum(T_cap * (v * w)[:, None], axis=0)
    den = np.sum(v ** 2 * w)
    return -(1.0 / r_v) * (v_rms_kms / C_KMS) * num / den


def bootstrap_errors(T_cap, v_r_kms, n_boot=500, seed=0, **kwargs):
    """
    Bootstrap over objects: resample the catalogue with replacement,
    restack, and take the scatter across the restacks.

    Returns (mean, std) of the stacked profile.
    """
    rng = np.random.default_rng(seed)
    n = T_cap.shape[0]  # number of objects
    boots = np.empty((n_boot, T_cap.shape[1]))  # one row per bootstrap realisation, one column per aperture
    for b in range(n_boot):
        idx = rng.integers(0, n, n) # draw with replacement
        boots[b] = velocity_weighted_stack(T_cap[idx], np.asarray(v_r_kms)[idx], **kwargs)  # restack
    return boots.mean(axis=0), boots.std(axis=0)

# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def lightcone_pos_to_sky(halo_centre):
    """
    Convert (N,3) comoving position relative to observer to unit vectors in comoving distance.

    The lightcone stores the (x, y, z) coordinates of each halo relative to the obeserver.
    The length of each vector equals the comoving distance to thaat halo.
    """
    pos = np.asarray(halo_centre, dtype=np.float64)
    r = np.linalg.norm(pos, axis=1)
    return pos / r[:, None], r


def radial_velocity(vel_kms, unit_vecs):
    """
    Compute the peculiar radial velocity by projecting the 3D velocity of each halo
    onto its own line of sight. Positive = moving away from the observer.
    """
    return np.sum(np.asarray(vel_kms, dtype=np.float64) * unit_vecs, axis=1)


def gaussian_beam_smooth(sky_map, fwhm_arcmin, lmax=None):
    """
    Convolve a full-sky map with a Gaussian beam (ACT f150 ~ 1.3', f090 ~ 2.1').
    """
    return hp.smoothing(sky_map, fwhm=np.radians(fwhm_arcmin / 60.0), lmax=lmax)

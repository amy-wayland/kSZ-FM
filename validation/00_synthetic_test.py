import numpy as np
import healpy as hp

from ksz_utils import (cap_profile, cap_profiles_catalogue,
                       velocity_weighted_stack, bootstrap_errors,
                       SR_TO_ARCMIN2, C_KMS)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

NSIDE = 2048  # 1.72 arcmin pixels
SIGMA_ARCMIN = 30.0  # width of the injected Gaussian source
AMP_UK = 50.0  # peak dT for a source moving at v_rms
N_OBJ = 600  # number of mock objects
V_RMS = 300.0  # km/s
NOISE_UK = 20.0  # white noise per pixel
THETA = np.geomspace(26.0, 90.0, 6)  # aperture radii in arcmin

rng = np.random.default_rng(42)

# --------------------------------------------------------------------------
# Create artificial source
# --------------------------------------------------------------------------

def analytic_cap(A, sigma_rad, theta_d_rad):
    """
    Continuum CAP of a circular Gaussian.
    """
    x = theta_d_rad ** 2 / (2 * sigma_rad ** 2)  # disc radius in units of the Gaussian variance
    return 2 * np.pi * A * sigma_rad ** 2 * (1 - 2 * np.exp(-x) + np.exp(-2 * x))


def brute_force_cap(sky, nside, vec, theta_d_rad):
    """
    The same filter evaluated over every pixel in the map.

    cap_profile() in ksz_utils.py does the same sum but only over the pixels
    query_disc returns, which is fast but relies on boundary behaviour. 
    Here we test this.
    """
    px, py, pz = hp.pix2vec(nside, np.arange(sky.size))  # unit vectors of every pixel in the map
    ang = np.arccos(np.clip(vec[0] * px + vec[1] * py + vec[2] * pz, -1, 1))  # angular separation
    area = hp.nside2pixarea(nside)
    out = []
    for td in np.atleast_1d(theta_d_rad):
        w = np.zeros(sky.size)
        w[ang < td] = 1.0
        w[(ang >= td) & (ang < np.sqrt(2) * td)] = -1.0
        out.append(float(np.sum(sky * w) * area))
    return np.array(out)


def gaussian_source(sky, nside, vec, amp, sigma_rad):
    """
    Paint a circular Gaussian of peak amplitude `amp' at direction `vec'
    """
    pix = hp.query_disc(nside, vec, 6 * sigma_rad)  # only considers pixels within 6 sigma of the centre
    px, py, pz = hp.pix2vec(nside, pix)  # unit vectors for just these pixels
    cos = np.clip(vec[0] * px + vec[1] * py + vec[2] * pz, -1, 1)
    ang = np.arccos(cos)
    sky[pix] += amp * np.exp(-0.5 * (ang / sigma_rad) ** 2)  # superposition of sources

# --------------------------------------------------------------------------
# Perform sanity checks
# --------------------------------------------------------------------------

def main():
    # Convert to radians and compute pixel size
    sigma_rad = np.radians(SIGMA_ARCMIN / 60.0)
    theta_rad = np.radians(THETA / 60.0)
    pix_arcmin = hp.nside2resol(NSIDE, arcmin=True)
    print(f"nside {NSIDE}, pixel {pix_arcmin:.2f}', "
          f"theta_d/pixel = {THETA[0]/pix_arcmin:.1f} - {THETA[-1]/pix_arcmin:.1f}")
    ok = True

    # Create synthetic source
    sky = np.zeros(hp.nside2npix(NSIDE))
    vec = np.array(hp.ang2vec(np.pi / 2, 0.3))
    gaussian_source(sky, NSIDE, vec, AMP_UK, sigma_rad)

    # ----------------------------------------------------------------
    # Check 1: CAP profile implementation
    # ----------------------------------------------------------------
    print("\n" + "=" * 66)
    print("Check 1: cap_profile() vs brute-force filter over the whole map")
    print("=" * 66)
    fast = cap_profile(sky, NSIDE, vec, theta_rad)  # from ksz_utils.py
    slow = brute_force_cap(sky, NSIDE, vec, theta_rad)
    rel = np.abs(fast / slow - 1)
    for t, f, s in zip(THETA, fast * SR_TO_ARCMIN2, slow * SR_TO_ARCMIN2):
        print(f"{t:8.2f}' {f:14.4f} {s:14.4f}")
    print(f"\nmax relative difference {rel.max():.2e}")
    if rel.max() > 1e-10:
        ok = False
        print("FAIL: disc query / binning logic is wrong")
    else:
        print("PASS")

    # ----------------------------------------------------------------
    # Check 2: Normalisation of CAP filter
    # ----------------------------------------------------------------
    print("\n" + "=" * 66)
    print("Check 2: CAP filter vs continuum analytic Gaussian")
    print("=" * 66)
    num = fast * SR_TO_ARCMIN2
    ana = analytic_cap(AMP_UK, sigma_rad, theta_rad) * SR_TO_ARCMIN2
    print(f"{'theta':>9} {'numeric':>13} {'analytic':>13} {'ratio':>8}")
    for t, n, a in zip(THETA, num, ana):
        print(f"{t:8.2f}' {n:13.2f} {a:13.2f} {n/a:8.4f}")
    worst = np.max(np.abs(num / ana - 1))
    print(f"\nworst deviation {100*worst:.2f}% (pixelisation)")
    if worst > 0.05:
        ok = False
        print("FAIL")
    else:
        print("PASS")

    # ----------------------------------------------------------------
    # Check 3: Spatially flat background integrates to zero
    # ----------------------------------------------------------------
    print("\n" + "=" * 66)
    print("Check 3: constant background integrates to zero")
    print("=" * 66)
    flat = np.full(hp.nside2npix(NSIDE), 123.4)
    res = cap_profile(flat, NSIDE, vec, theta_rad)
    disc = 123.4 * np.pi * theta_rad ** 2
    frac = np.abs(res) / disc
    print("residual / disc integral:", np.array2string(frac, precision=5))
    if frac.max() > 0.03:
        ok = False
        print("FAIL: filter is not compensated at this pixelisation")
    else:
        print("PASS")

    # ----------------------------------------------------------------
    # Check 4: Recovery of input amplitude
    # ----------------------------------------------------------------
    print("\n" + "=" * 66)
    print("Check 4: velocity-weighted stack recovers the input amplitude")
    print("=" * 66)
    sky = rng.normal(0.0, NOISE_UK, hp.nside2npix(NSIDE))  # single-source map
    v_r = rng.normal(0.0, V_RMS, N_OBJ)  # line-of-sight peculiar velocities
    vecs = np.array(hp.ang2vec(np.arccos(rng.uniform(-1, 1, N_OBJ)),
                               rng.uniform(0, 2 * np.pi, N_OBJ)))  # uniformly distributed directions

    # kSZ: dT = -(v_r/c) * profile, normalised so |amp| = AMP_UK at v_r = V_RMS
    for v, u in zip(v_r, vecs):
        gaussian_source(sky, NSIDE, u, -(v / V_RMS) * AMP_UK, sigma_rad)

    T_cap = cap_profiles_catalogue(sky, NSIDE, vecs, theta_rad,
                                   progress_every=0) * SR_TO_ARCMIN2
    stack = velocity_weighted_stack(T_cap, v_r)
    _, err = bootstrap_errors(T_cap, v_r, n_boot=400)

    v_rms_meas = np.sqrt((v_r ** 2).mean())
    expect = (v_rms_meas / V_RMS) * ana  # expected amplitude

    print(f"{'theta':>9} {'stacked':>12} {'+/-':>9} {'expected':>12} {'pull':>7}")
    pulls = []  # (measured - expected) / uncertainty
    for t, s, e, x in zip(THETA, stack, err, expect):
        pulls.append((s - x) / e)
        print(f"{t:8.2f}' {s:12.2f} {e:9.2f} {x:12.2f} {pulls[-1]:7.2f}")
    pulls = np.array(pulls)
    print(f"\nmax |pull| = {np.abs(pulls).max():.2f} sigma")
    if not np.all(stack > 0):
        ok = False
        print("FAIL: sign convention wrong - a real kSZ signal must stack positive")
    elif np.abs(pulls).max() > 3.5:
        ok = False
        print("FAIL: recovered amplitude is biased")
    else:
        print("PASS")

    # Null test: randomly flip the sign of each velocity. This destroys the correlation
    # between velocity and temperature that carries the signal. Should be consistent with zero.
    null = velocity_weighted_stack(T_cap, v_r * rng.choice([-1, 1], N_OBJ))
    print(f"\nvelocity-shuffled null: max |null/sigma| = {np.abs(null/err).max():.2f} "
          "(single realisation, apertures are correlated)")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

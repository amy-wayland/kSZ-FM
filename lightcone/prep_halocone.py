import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
from astropy import units as u
import os

try:
    from .constants import *
    from .params import *
    from .params import par as params_par
except ImportError:
    from constants import *
    from params import *
    from params import par as params_par

# IDs in the lightcone are made unique per snapshot as ID + i_file * ID_STRIDE
ID_STRIDE = 10**8


def get_shell_cov_dist(z_boundary, cosmo):

    # get the comoving distances
    from astropy.units import Mpc, Msun
    cov_inner = cosmo.comoving_distance(z_boundary[0]).to(Mpc/cosmo.h).value
    cov_outer = cosmo.comoving_distance(z_boundary[1]).to(Mpc/cosmo.h).value
    # we take the same shell comoving distance as UFalcon
    shell_cov = cosmo.comoving_distance(np.mean(z_boundary)).to(Mpc/cosmo.h).value

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
    Read one snapshot halo catalogue and apply the base selection cuts.

    Returns a structured array with positions in comoving kpc/h and masses in Msun/h.
    This is exactly the population that gets copied into the lightcone, so it is
    also what the snapshot side of the HMF validation should use.
    """
    # get the param mask
    if param.files.halofile_format in ('AHF-TEXT', 'AHF-ASCII'):
        names = "ID,IDhost,Mvir,Nvir,x,y,z,vx,vy,vz,rvir,b_ov_a,c_ov_a,Eax,Eay,Eaz,Ebx,Eby,Ebz,Ecx,Ecy,Ecz,cvir"
        h = np.genfromtxt(halo_file,usecols=(0,1,3,4,5,6,7,8,9,10,11,24,25,26,27,28,29,30,31,32,33,34,42),
                            comments='#',
                            dtype=None,
                            names=names)

        # use only halos that have sensible values
        base_mask = (h['Nvir'] >= param.sim.Nmin_per_halo) & (h['cvir'] > 0) & (h['Mvir'] >= param.code.Mhalo_min)
        h = h[base_mask]
        print(f"number of halos: {len(h['Mvir'])}")
        print(f"Mass range: {np.min(h['Mvir']):.2e} - {np.max(h['Mvir']):.2e} Msun/h")

    elif param.files.halofile_format == 'VELOCIraptor-HDF5':
        # mass field to use as "Mvir" below, and its matching radius/concentration
        # fields (keep these paired -- don't mix e.g. Mass_200crit with Rvir)
        mass_field = getattr(param.files, 'vr_mass_field', 'Mass_200crit')
        radius_field = {'Mass_200crit': 'R_200crit', 'Mass_200mean': 'R_200mean',
                         'Mass_BN98': 'R_BN98', 'Mvir': 'Rvir'}.get(mass_field, 'R_200crit')
        conc_field = {'Mass_200crit': 'cNFW_200crit', 'Mass_200mean': 'cNFW_200mean',
                      'Mass_BN98': 'cNFW_BN98'}.get(mass_field, 'cNFW_200crit')

        names = "ID,IDhost,Mvir,Nvir,x,y,z,vx,vy,vz,rvir,b_ov_a,c_ov_a,Eax,Eay,Eaz,Ebx,Eby,Ebz,Ecx,Ecy,Ecz,cvir"
        dtype = [(n, np.int64 if n in ('ID', 'IDhost', 'Nvir') else np.float64) for n in names.split(',')]

        with h5py.File(halo_file, 'r') as f:
            hostHaloID = f['hostHaloID'][:]
            isfield = hostHaloID == -1

            h = np.zeros(int(isfield.sum()), dtype=dtype)
            h["ID"] = f["ID"][:][isfield]
            h["IDhost"] = 0  # all are field halos by construction, matches AHF's "no host" convention
            h["Mvir"] = f[mass_field][:][isfield] * 1.0e10  # -> Msun/h (comoving, Comoving_units=1)
            h["Nvir"] = f["npart"][:][isfield]
            h["x"] = f["Xc"][:][isfield] * 1000.0
            h["y"] = f["Yc"][:][isfield] * 1000.0
            h["z"] = f["Zc"][:][isfield] * 1000.0
            h["vx"] = f["VXc"][:][isfield]
            h["vy"] = f["VYc"][:][isfield]
            h["vz"] = f["VZc"][:][isfield]
            h["rvir"] = f[radius_field][:][isfield] * 1000.0
            h["b_ov_a"] = f["q"][:][isfield]
            h["c_ov_a"] = f["s"][:][isfield]
            h["Eax"], h["Eay"], h["Eaz"] = f["eig_xx"][:][isfield], f["eig_yx"][:][isfield], f["eig_zx"][:][isfield]
            h["Ebx"], h["Eby"], h["Ebz"] = f["eig_xy"][:][isfield], f["eig_yy"][:][isfield], f["eig_zy"][:][isfield]
            h["Ecx"], h["Ecy"], h["Ecz"] = f["eig_xz"][:][isfield], f["eig_yz"][:][isfield], f["eig_zz"][:][isfield]
            h["cvir"] = f[conc_field][:][isfield]

        base_mask = (h['Nvir'] >= param.sim.Nmin_per_halo) & (h['Mvir'] >= param.code.Mhalo_min)
        h = h[base_mask]
        print(f"number of halos: {len(h['Mvir'])}")
        print(f"Mass range: {np.min(h['Mvir']):.2e} - {np.max(h['Mvir']):.2e} Msun/h")

    elif param.files.halofile_format == 'FLAMINGO-SOAP-HBT':
        # SOAP/HBTplus halo catalogue (SWIFT-style HDF5). All physical quantities are
        # stored h-free (h-scale exponent = 0 on every dataset) -- comoving Mpc for
        # lengths, 1e10 Msun for masses -- so *h converts to the Mpc/h, Msun/h
        # convention the rest of this function (and AHF/VR branches) use.
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

        # the per-snapshot ID offset (ID + i_file*ID_STRIDE) must not collide, otherwise
        # satellite -> host matching through IDhost silently breaks
        if len(h) and h["ID"].max() >= ID_STRIDE:
            raise ValueError(preamble + f"HaloCatalogueIndex up to {h['ID'].max()} >= ID_STRIDE={ID_STRIDE}; "
                             "increase ID_STRIDE in prep_halocone.py")

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
            h_lean = np.zeros(len(h), dtype=lean_dtype)
            for name, _ in lean_dtype:
                h_lean[name] = h[name]
            h = h_lean

    else:
        raise ValueError(preamble + "This halo format is currently not supported {}...".format(param.files.halofile_format))

    return h


def get_single_halo_shell(param, halo_file, z_boundary, i_file, max_repli=7, test=False):
    """
    Creates a halo file for the lightcone, given the halo files of the snapshots and their cut out shells
    """
    # set the logging preamble
    preamble = "File ID {}: ".format(i_file)

    # get a cosmology object
    cosmo = cosmo = FlatLambdaCDM(H0=100*param.cosmo.h0, Ob0=param.cosmo.Ob, Om0=param.cosmo.Om)

    # save the relevant lines
    relevant_lines = []

    # a list for the shell file
    shell_dtype = np.dtype([("shell_id", np.int32),
                            ("lower_z", np.float32),
                            ("upper_z", np.float32),
                            ("lower_com", np.float32),
                            ("upper_com", np.float32),
                            ("shell_com", np.float32)])

    shell_cov, cov_inner, cov_outer = get_shell_cov_dist(z_boundary, cosmo)

    print(f"\nProcessing snapshot {i_file}: {os.path.basename(halo_file)}")
    print(f"Redshift shell: {z_boundary[0]:.2f} < z < {z_boundary[1]:.2f}")
    print(f"Comoving distances: {cov_inner:.1f} - {cov_outer:.1f} Mpc/h")

    # create the shell data (shell_id, lower z, upper z, lower com, upper com, shell com)
    shell_data = np.array([(i_file, z_boundary[0], z_boundary[1], cov_inner, cov_outer, shell_cov)],
                            dtype=shell_dtype)

    # read the snapshot catalogue + base cuts
    h = load_snapshot_halos(param, halo_file, preamble=preamble)

    # update the ids
    h["ID"] = h["ID"] + i_file*ID_STRIDE
    if "is_central" in h.dtype.names:
        sat = h["is_central"] == 0
        h["IDhost"][sat] = h["IDhost"][sat] + i_file*ID_STRIDE

    # recentre on the lightcone observer: subtract its position and re-wrap
    # periodically so the observer sits at the coordinate origin (the shell
    # selection below measures radii from the origin). observer_pos is in the
    # same comoving Mpc/h units as param.sim.Lbox; default (0,0,0) keeps the
    # observer at the box corner (previous behaviour).
    obs_pos = np.asarray(getattr(param.code, "observer_pos", (0.0, 0.0, 0.0)), dtype=float)
    if np.any(obs_pos != 0.0):
        Lbox_kpc = param.sim.Lbox * 1000.0
        for _coord, _off in zip(("x", "y", "z"), obs_pos):
            h[_coord] = np.mod(h[_coord] - _off * 1000.0, Lbox_kpc)
        print(f"Recentred on observer at {tuple(obs_pos)} Mpc/h (periodic wrap)")

    # N tot for counting
    N_tot = 0

    # get the positions and transform
    halos = np.array([h["x"], h["y"], h["z"]]).T/1000.0  # to Mpc

    # copy the halo lines
    halo_lines = h.copy()

    assert len(halo_lines) == len(halos), preamble + "There seems to be something wrong with the halo file, " \
                                                     "more than one comment on top?"

    # get the offset indices
    offset_indices = get_offset_indices(boxsize=param.sim.Lbox, min_val=cov_inner, max_val=cov_outer,
                                        max_repli=max_repli,buffer=param.code.halo_lc_buffer)

    print(f"\nBox replication:")
    print(f"Total replicated boxes: {len(offset_indices)}")
    print(f"First 5 box indices: {offset_indices[:5]}")

    # cycle and project
    old_offset = np.zeros(3)

    # select all relevant halos
    for offset in offset_indices:

        # get the new offset and add to positions
        mpc_offset = (offset - old_offset) * param.sim.Lbox
        halos += mpc_offset

        # update halos
        halo_lines['x'] += mpc_offset[0] * 1000
        halo_lines['y'] += mpc_offset[1] * 1000
        halo_lines['z'] += mpc_offset[2] * 1000

        # update
        old_offset = offset

        # convert to radii
        radii = np.linalg.norm(halos, axis=1)

        # get the masks
        # thickness of the padding zones
        buffer = param.code.halo_lc_buffer

        mask_inner = np.logical_and(radii >= (cov_inner - buffer), radii <  cov_inner)
        mask_middle = np.logical_and(radii >= cov_inner, radii <  cov_outer)
        mask_outer = np.logical_and(radii >= cov_outer, radii <  (cov_outer + buffer))
        n_pass = np.sum(mask_inner) + np.sum(mask_middle) + np.sum(mask_outer)
        N_tot += n_pass

        # if there are no halos in the shel, do nothing
        if n_pass == 0:
            continue

        # replica label: (ID, replica) is unique in the lightcone, ID alone is not
        # (the same snapshot halo appears once per replicated box that hits the shell).
        # Satellites link to their host via (IDhost, replica).
        replica = encode_replica(offset, max_repli)

        for buffer_flag, mask in zip([-1, 0, 1],[mask_inner, mask_middle, mask_outer]):
            local_pass = np.sum(mask)
            if local_pass == 0:
                continue
            extended_halo_dtype = halo_lines.dtype.descr + [("shell_id", np.int32), ("halo_buffer", np.int8),
                                                            ("replica", np.int32)]
            relevant = np.zeros(local_pass, dtype=extended_halo_dtype)
            relevant['shell_id']    = i_file
            relevant['halo_buffer'] = buffer_flag
            relevant['replica']     = replica
            masked = halo_lines[mask]
            for name in halo_lines.dtype.names:
                relevant[name] = masked[name]
            relevant_lines.append(relevant)


    print(f"\nTotal halos selected: {N_tot} "
          f"({N_tot/max(len(halo_lines), 1)*100:.1f}% of input)")

    return relevant_lines, shell_data

def prep_lightcone_halos(param, halo_files, shells_z, max_repli=7, test=False):
    """
    Creates a halo file for the lightcone, given the halo files of the snapshots and their cut out shells
    """
    print(param)
    assert len(halo_files) == len(shells_z), "Halo file and boundaries need to match in first dimension"

    relevant_lines = []
    shell_data = []

    for i_file, (halo_file, z) in enumerate(zip(halo_files, shells_z)):

        halos_local, shell_local = get_single_halo_shell(param, halo_file, z, i_file, max_repli=max_repli, test=test)
        shell_data.append(shell_local)
        relevant_lines.extend(halos_local)

    # concat shell data
    shell_data = np.concatenate(shell_data)

    # concat the haloes
    n_total = sum(len(r) for r in relevant_lines)
    halo_data = np.empty(n_total, dtype=relevant_lines[0].dtype)
    pos = 0
    relevant_lines.reverse()
    while relevant_lines:
        piece = relevant_lines.pop()
        halo_data[pos:pos + len(piece)] = piece
        pos += len(piece)
        del piece

    # pieces are appended in shell order, so this is normally already sorted
    if param.code.halo_lc_buffer > 0 and np.any(np.diff(halo_data["shell_id"]) < 0):
        sorted_args = np.argsort(halo_data["shell_id"], kind="stable")
        halo_data = halo_data[sorted_args]
        del sorted_args

    # trim impossible shell ids
    mask = np.logical_and(halo_data["shell_id"] != -1, halo_data["shell_id"] != len(shell_data))
    if not np.all(mask):
        halo_data = halo_data[mask]
    del mask

    has_sats = "is_central" in halo_data.dtype.names

    print("\n=== Final Lightcone Summary ===")
    print(f"Total shells processed: {len(shell_data)}")
    print(f"Total halos in lightcone: {len(halo_data)}")
    if has_sats:
        n_c = int(np.sum(halo_data["is_central"] == 1))
        print(f"  centrals: {n_c}, satellites: {len(halo_data) - n_c}")

    print("\nShell statistics:")
    for shell in shell_data:
        shell_mask = halo_data['shell_id'] == shell['shell_id']
        line = (f"Shell {shell['shell_id']}: "
                f"z={shell['lower_z']:.2f}-{shell['upper_z']:.2f} | "
                f"Halos: {np.sum(shell_mask)}")
        if has_sats:
            line += f" (sat: {np.sum(shell_mask & (halo_data['is_central'] == 0))})"
        print(line)

    return halo_data, shell_data


def build_default_param():
    """Return the default nested parameter object from params.py."""
    return params_par()


if __name__ == "__main__":
    param = build_default_param()

    # Cosmology overrides
    param.cosmo.Om = 0.306
    param.cosmo.Ob = 0.0486
    param.cosmo.h0 = 0.681
    param.cosmo.s8 = 0.807
    param.cosmo.ns = 0.967

    # Lightcone / halo-selection controls
    param.sim.Lbox = 900.0
    param.sim.Nmin_per_halo = 100
    param.code.halo_lc_buffer = 5.0
    param.files.halofile_format = 'AHF-TEXT'
    param.code.nside_out = 2048
    param.code.interp_part = False

    print("Parameter object initialized. Call prep_lightcone_halos(...) with your inputs.")
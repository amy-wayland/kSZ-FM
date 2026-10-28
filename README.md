# kSZ Forward Model

## Part 1: FLAMINGO validation data

The code in the validation folder reproduces the stacked kSZ profile from the public FLAMINGO data. This will form the validation data vector against which to build the forward model. The measurement follows McCarthy et al. 2024 (arXiv:2410.19905).

### Requirements

```bash
python -m pip install -r requirements.txt
```

### Quick start

```bash
python 00_synthetic_test.py  # verify the machinery, no data needed
python 01_explore_flamingo.py  # find out what is in the files
python 02_build_ksz_map.py --shells 10 11 12 --remote
python 03_select_sample.py --sample cmass --mstar-cut 10.9 --remote-halos --velocity-field BoundSubhalo/CentreOfMassVelocity --out sample_cmass_bsv.npz
python 04_stack_cap.py --map dT_ksz_*.npy --catalogue sample_cmass_*.npz --min-mstar 11.2 --beam 1.3
python 05_plot_datavector.py --cap cap_profile.npz --label "FLAMINGO L1_m9"
```

### Pipeline

#### ksz_utils.py:
- Contains the physics and geometry that the other scripts import.

#### 00_synethetic_test.py:
- Constructs an artificial sky with a signal of known amplitude and scape dependence, and checks that the framework recovers it.
- The sanity checks performed are:
  - Indexing logic of the CAP profile;
  - Normalisation of the CAP profile;
  - Integration to zero of a constant sky;
  - Recovery of the injected signal from the velocity weighted stack.

#### 01_explore_flamingo.py:
- Reads the FLAMINGO metadata from the Durham server (Cosma) and prints:
  - Which simulations exist and what their directories are called;
  - The layout of the shell files and which maps are inside each one;
  - The redshift and comoving bounds of each shell, and the pixel size;
  - The halo-lightcone file list and redshift coverage;
  - The SOAP dataset names.

#### 02_build_ksz_map.py:
- Builds the kSZ temperature map from the Doppler B parameter, $b$, using $\Delta T = −T_{\rm CMB} \times \Sigma_{i \in {\rm shells}} b_i$.

#### 03_select_sample.py:
- Build the sample to stack on by joining the halo lightcone to SOAP.
- There are two selection modes:
  - Galaxy-like. This follows McCarthy et al. 2024 in which a minimum stellar mass cut is applied.
    ```bash
    python 03_select_sample.py --sample cmass --mstar-cut 11.2
    ```
  - Halo-mass bin. This selects haloes directly in a bin of $M_{500{\rm c}}$ and stacks them.
    ```bash
    python 03_select_sample.py --halo-mass-bin 13.0 13.5
    ```

#### 04_stack_cap.py:
- Computes the final stacked kSZ measurement via CAP on each galaxy, with velocity weighting, and obtains bootstrap errors.

#### 05_plot_datavector.py:
- Plots the stacked kSZ profile as a function of aperture radius.


## Part 2: FLAMINGO DMO halo and subhalo lightcone

Builds a full-sky halo and subhalo lightcone $(0 < z < 2)$ from the public FLAMINGO L1_m9_DMO SOAP-HBT catalogues, with every object placed where it crosses the lightcone. The pipeline is validated against the snapshot mass functions and the FLAMINGO DMO HEALPix maps.

### Pipeline

#### fetch_flamingo_soap.py
- Download the halo catalogues:
  - Streams only the needed columns and rows of the SOAP-HBT catalogues from the public FLAMINGO server into small local files (`soap_subset/').
  - Writes 'flamingo_shells_z.txt' (shell redshift edges) and 'flamingo_halo_files.txt' (catalogue paths), which the run script reads.
  - The mass cuts only decide what is downloaded. Fetch at or below the cuts used in step 3.

    ```bash
    python fetch_flamingo_soap.py --snaps 37-77 --outdir soap_subset --mhalo-min 8e12 --msub-min 1e12 --nmin 100
    ```
    
#### add_track_info.py
- Adds the HBT track information needed to place haloes at their lightcone crossing:
  - Writes the `TrackId` of every row into each `soap_subset` file.
  - For each snapshot, stores the position of every object in the next, later snapshot (`Tracks/`), including objects below the mass cuts at the earlier time.
  - Resumable; run once after `fetch_flamingo_soap.py`.

    ```bash
    python add_track_info.py --snaps 37-77 --outdir soap_subset
    ```
    
#### prep_halocone_crossing.py
- Lightcone builder with crossing-based placement:
  - Reads each snapshot catalogue and applies the cuts: centrals in M200c and satellite in bound mass.
  - The FLAMINGO maps record each particle when it crosses the lightcone, but haloes exist only at snapshot times. Each map shell is therefore built from the two snapshots at its edges.
  - Every halo is matched between them by its HBT `TrackId`, moved on a straight line, and placed where it crosses the lightcone. Output shells coincide with the map shells.

#### run_prep_halocone_crossing.py
- Builds the crossing-based lightcone for all 40 shells ($0 < z < 2$):
  - Sets the cosmology, mass cuts, and observer position at (750, 750, 750) cMpc, which matches FLAMINGO's `lightcone0` maps.
 
    ```bash
    python run_prep_halocone_crossing.py --out halo_lightcone_flamingo_crossing_obs750_hostcut.h5
    ```

#### validate_lightcone_hmf.py
- Validation test 1 (halo mass function):
  - For each chosen shell, compares the lightcone number density with that of the snapshot at the shell's lower edge, in which its objects were selected.
  - Centrals are compared in M200c and satellites in bound mass.
  - Makes sky-coverage maps and reports the fraction of satellites whose host is in  the ligthcone.
 
    ```bash
    python validate_lightcone_hmf.py halo_lightcone_flamingo_subhalos_obs750.h5 --shells 0 5 10 20 30 40 --out plots/hmf
    ```

#### fetch_healpix_shell.py
- Download a DMO map shell:
  - Downloads one FLAMINGO HEALPix shell, by default `DarkMatterMass' at $N_{\rm side} = 4096$.
  - A given shell $y$ covers $0.05 y < z < 0.05 (1+y)$.
  - Prints the comoving radii of the shell in Mpc/h.
 
    ```bash
    python fetch_healpix_shell.py --observer 0 --shell 2
    ```

#### overlay_halos_on_map.py
- Validation test 2 (positions):
  - Overlays lightcone haloes and subhaloes in the radial range of the map on the DMO map.

    ```bash
    python overlay_halos_on_map.py halo_lightcone_flamingo_crossing_obs750_hostcut.h5 \
    --map maps/L1_m9_DMO_lightcone0_shell2_DarkMatterMass_nside4096.hdf5 \
    --dataset DarkMatterMass --zmin 0.10 --zmax 0.15 --out plots/overlay_shell2
    ```

#### halo_map_offsets.py
- Validation test 3 (halo–map offsets):
  - For centrals in bins of mass, finds the mass centre in the DMO map with a shrinking-circle centroid and reports the median offset from the catalogue position.
  - Reports the component of the offset along the timing shift a snapshot-based placement would introduce ($\approx 0$ for the crossing-based lightcone).

    ```bash
    python halo_map_offsets.py halo_lightcone_flamingo_crossing_obs750_hostcut.h5 \
        --map maps/L1_m9_DMO_lightcone0_shell2_DarkMatterMass_nside4096.hdf5 --out plots/offsets_shell2
    ```

#### split_for_bfc.py
- Writes one halo file per map shell in the format read by the baryonification code (centrals only, with M200c, R200c and c200c).

    ```bash
    python split_for_bfc.py halo_lightcone_flamingo_crossing_obs750_hostcut.h5 --outdir bfc_halos --shells 0 39
    ```


## Part 3: Shell baryonification

Baryonifies the FLAMINGO DMO HEALPix shells with the shell implementation of Schneider et al. (2025), using the `euclid_fs2` branch of https://github.com/aurelschneider/baryonification.

### Pipeline

#### run_bfc_flamingo.py
- Sets up and runs the `ShellDisplacer`: FLAMINGO cosmology, FLAMINGO-calibrated baryonic parameters (`--model fiducial` or `fgas_8sigma`), multi-component output (dark matter, gas, stars), haloes above `--mhalo-min`, input maps and per-shell halo files.

#### check_bfc_output.py
- Checks a baryonified shell against the DMO input:
  - Combines the components as $\rho_{\rm tot} = (1-f_{\rm b})\,\rho_{\rm dm} + f_{\rm b}\,(\rho_{\rm gas} + \rho_{\rm star})$.
  - Mass conservation and component fractions, the ratio $C_\ell^{\rm bfc}/C_\ell^{\rm DMO}$, and a zoom on the most massive halo.

    ```bash
    python check_bfc_output.py bfc_fiducial_shell2_nside4096.h5 --shell 2 \
        --map maps/L1_m9_DMO_lightcone0_shell2_DarkMatterMass_nside4096.hdf5 \
        --halos bfc_halos/halo_lightcone_shell_2.h5 --out plots/bfc_check_shell2
    ```

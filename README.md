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

Builds a full-sky halo and subhalo lightcone $(0 < z < 2)$ from the public FLAMINGO L1_m9_DMO SOAP-HBT catalogues, and validates it against the snapshot mass functions and the FLAMINGO DMO HEALPix maps.

### Pipeline

#### fetch_flamingo_soap.py
- Downloads the halo catalogues:
  - Streams only the needed columns and rows of the SOAP-HBT catalogues from the public FLAMINGO server into small local files (`soap_subset/').
  - Writes 'flamingo_shells_z.txt' (shell redshift edges) and 'flamingo_halo_files.txt' (catalogue paths), which the run script reads.
  - The mass cuts only decide what is downloaded. Fetch at or below the cuts used in step 3.

    ```bash
    python fetch_flamingo_soap.py --snaps 37-77 --outdir soap_subset --mhalo-min 8e12 --msub-min 1e12 --nmin 100
    ```
    
#### prep_halocone.py
- Lightcone builder:
  - Places the haloes of each snapshot into its redshift shell using periodic box replication and recentring on the observer.
  - Includes centrals (selected on M200c) and satellite subhaloes (selected on bound mass).

#### run_prep_halocone_flamingo_subhalos.py
- Construct the lightcone using prep_halo.py:
  - Sets the cosmology, mass cuts, and observer position at (750, 750, 750) cMpc, which matches FLAMINGO's `lightcone0' maps.
  - Writes 'halo_ligthcone_flamingo_subhalos_obs750.h5'.
 
    ```bash
    python run_prep_halocone_flamingo_subhalos.py
    ```

#### validate_lightcone_hmf.py
- Validation test 1 (halo mass function):
  - For each chosen shell, compares the lightcone number density with that of the snapshot that filled it. Centrals are compared in M200c and satellites in bound mass.
  - Makes sky-coverage maps and reports the fraction of satellites whose host is in  the ligthcone.
 
    ```bash
    python validate_lightcone_hmf.py halo_lightcone_flamingo_subhalos_obs750.h5 --shells 0 5 10 20 30 40 --out plots/hmf
    ```

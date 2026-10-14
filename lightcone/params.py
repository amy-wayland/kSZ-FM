   from types import SimpleNamespace as NS

   def par():
       """Minimal stand-in for the project's params.par(); every value is set in the run script."""
       return NS(
           cosmo=NS(Om=0.306, Ob=0.0486, h0=0.681, s8=0.807, ns=0.967),
           sim=NS(Lbox=681.0, Nmin_per_halo=100),
           code=NS(halo_lc_buffer=1e-9, Mhalo_min=8e12, observer_pos=(0.0, 0.0, 0.0)),
           files=NS(halofile_format="FLAMINGO-SOAP-HBT"),
       )

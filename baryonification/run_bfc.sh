#!/bin/bash
# Launch run_bfc_flamingo.py under MPI (1 master + NP-1 workers) inside one job.
#
# addqueue -n 1xNP starts this script NP times through srun (one copy per reserved core).
# Only the first copy (SLURM_PROCID=0) launches the MPI run; the others exit at once.
# The MPI run uses the bfc environment's own mpiexec, so the Slurm/PMIx variables that srun
# sets are removed first (otherwise MPICH sees both PMI_SIZE and PMIX_NAMESPACE and aborts).
#
# Usage:  addqueue -c bfc -n 1x17 -m 4 ./run_bfc.sh --first 6 --last 6 --transfct ...
# conda env with the baryonification package, mpi4py and psutil, e.g.
#   ENV=$HOME/miniconda3/envs/bfc addqueue ... ./run_bfc.sh ...
ENV=${ENV:-$HOME/miniconda3/envs/bfc}

if [ "${SLURM_PROCID:-0}" != "0" ]; then
    exit 0
fi

# number of MPI processes: the cores reserved for the job (or NP if given)
NP=${NP:-${SLURM_NTASKS:-17}}

# keep the job ID for the temp-file names, then drop the launcher variables srun set
JOBID=${SLURM_JOB_ID:-bfc$$}
for v in $(env | grep -oE '^(PMIX_[A-Za-z0-9_]*|PMI_[A-Za-z0-9_]*|OMPI_[A-Za-z0-9_]*)'); do unset "$v"; done
export SLURM_JOB_ID=$JOBID

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHON_LOGGER_LEVEL=${PYTHON_LOGGER_LEVEL:-info}
export BFC_NCPU=$((NP - 1))

# srun may have pinned this copy to a single core; widen the affinity to every core the job
# was given (the kernel limits it to the job's own cores), so the MPI processes can spread out
if command -v taskset >/dev/null 2>&1; then
    taskset -p -c 0-$(( $(getconf _NPROCESSORS_CONF) - 1 )) $$ >/dev/null 2>&1
fi
echo "usable cores: $(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)  (should be >= $NP)"

cd "$(dirname "$0")"
echo "launching $NP MPI processes on $(hostname) (job $SLURM_JOB_ID)"
# -launcher fork: start all processes on this node directly, not through srun again
exec "$ENV/bin/mpiexec" -launcher fork -n "$NP" "$ENV/bin/python" -u run_bfc_flamingo.py "$@"

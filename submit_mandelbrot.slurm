#!/bin/bash
# ============================================================================
# submit_mandelbrot.slurm
# ----------------------------------------------------------------------------
# SLURM batch script that submits mandelbrot_benchmark.py as a job, using
# mpi4py for the parallel path instead of Python's multiprocessing.
#
# Written for a single c7i-flex.large node: 2 vCPUs, 4 GiB RAM
# ----------------------------------------------------------------------------
# An MPI job's rank count is fixed for its whole lifetime (set by `mpirun
# -np N` / `srun -n N`), unlike a multiprocessing.Pool that one running
# process can resize on the fly. So this job doesn't sweep worker counts in
# a single srun/mpirun call -- it runs the serial baseline once, then one
# MPI launch per rank count (1, 2, ... up to the allocation), each appending
# to the same CSV, and finishes with a --summarize pass that prints the
# combined table and redraws the speedup chart. This mirrors
# run_mpi_benchmark.sh, just expressed as one SLURM job instead of a
# standalone shell loop.
#
# --nodes=1 here isn't a limitation of the instance -- mandelbrot_benchmark.py
# only parallelizes within a single node (no cross-node halo exchange or
# similar), so there's nothing for a second node to do even if one were
# available.
#
# Submit with:
#   sbatch submit_mandelbrot.slurm
#
# Override any benchmark setting at submit time by appending extra args --
# they're forwarded to the Python script and take priority over the defaults
# baked in below (argparse: later --flag wins), e.g.:
#   sbatch submit_mandelbrot.slurm --width 2000 --height 2000 --max-iter 1000
#
# Check status / output:
#   squeue -u $USER
#   tail -f logs/mandelbrot-<jobid>.out
# ============================================================================

#SBATCH --job-name=mandelbrot-mpi
#SBATCH --output=logs/mandelbrot-%j.out
#SBATCH --error=logs/mandelbrot-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=2                 # c7i-flex.large has 2 vCPUs -- max real MPI ranks
#SBATCH --cpus-per-task=1          # each MPI rank uses one core
#SBATCH --mem=3500M                # leave headroom under the instance's 4 GiB total
#SBATCH --time=00:30:00
#SBATCH --partition=c7i-flex       # set to whatever partition targets your c7i-flex.large node(s)
# #SBATCH --exclusive              # uncomment for clean, un-contended timing (no other jobs on the node)

set -euo pipefail

mkdir -p logs

# --- environment -------------------------------------------------------------
# Adjust to however Python + MPI are provided on your cluster. Uncomment
# whichever applies; leave both commented to use whatever's already on PATH.
# module load openmpi/4.1
# module load python/3.11
# source /path/to/venv/bin/activate   # must have mpi4py + numpy (+ matplotlib) installed
#
# Sanity-check once, cheaply, that mpi4py is actually importable in this
# environment before burning the job's walltime on it:
python3 -c "import mpi4py" || {
    echo "mpi4py is not importable in this environment -- fix the module/venv lines above." >&2
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/mandelbrot_benchmark.py"
CSV="logs/results_${SLURM_JOB_ID}.csv"
OUT_PREFIX="logs/${SLURM_JOB_ID}_"

echo "Job ${SLURM_JOB_ID} running on $(hostname)"
echo "Allocated tasks (MPI ranks) for this job: ${SLURM_NTASKS}"

rm -f "${CSV}"

# --- serial baseline: one plain process, no MPI launch ----------------------
echo "=== serial baseline ==="
python3 "${PYTHON_SCRIPT}" --mode serial \
    --width 1200 --height 1200 --max-iter 500 --repeats 3 \
    --csv "${CSV}" --save-image --out-prefix "${OUT_PREFIX}" \
    "$@"

# --- MPI sweep: 1, 2, ... up to what this job was allocated ------------------
ranks=1
while [ "${ranks}" -le "${SLURM_NTASKS}" ]; do
    echo
    echo "=== MPI -n ${ranks} ==="
    # srun launches the ranks directly using SLURM's own PMI/PMIx support.
    # If your cluster's OpenMPI/MPICH wasn't built against Slurm's PMI (common
    # on a manually-set-up single node), swap this line for:
    #   mpirun -np "${ranks}" python3 "${PYTHON_SCRIPT}" --mode mpi ...
    srun --ntasks="${ranks}" --cpus-per-task=1 \
        python3 "${PYTHON_SCRIPT}" --mode mpi \
            --width 1200 --height 1200 --max-iter 500 --repeats 3 \
            --csv "${CSV}" \
            "$@"
    ranks=$((ranks * 2))
done

# To also see the falloff from oversubscribing past the 2 real vCPUs (as the
# multiprocessing version's --workers 1 2 4 did), bump the directives above to
# `#SBATCH --ntasks=4` and add `--overcommit` to the srun call (OpenMPI) or
# `mpirun --oversubscribe` if you switch to the mpirun line instead.

# --- combined summary + speedup chart ----------------------------------------
echo
echo "=== summary ==="
python3 "${PYTHON_SCRIPT}" --summarize "${CSV}" --out-prefix "${OUT_PREFIX}"

echo "Done. Results: ${CSV}"
echo "Images: ${OUT_PREFIX}mandelbrot.png, ${OUT_PREFIX}mandelbrot_speedup.png"

#!/bin/bash
# ============================================================================
# run_mpi_benchmark.sh
# ----------------------------------------------------------------------------
# Drives the full serial + MPI sweep for mandelbrot_benchmark.py and produces
# the combined summary table and plots.
#
# Because an MPI job's process count is fixed at launch (mpirun -np N), a
# single invocation of the Python script can't sweep multiple rank counts the
# way the old multiprocessing version swept --workers. This script does that
# sweep at the shell level instead: one plain run for the serial baseline,
# then one `mpirun -np N` call per rank count, all appending to the same CSV,
# finished off with a --summarize pass that prints the table and redraws the
# speedup chart.
#
# Usage:
#   ./run_mpi_benchmark.sh [max_ranks]
#   max_ranks defaults to `nproc` (2 on a c7i-flex.large).
#
# Requires: mpi4py installed, and an MPI runtime (OpenMPI or MPICH) providing
# `mpirun` on PATH.
# ============================================================================

set -euo pipefail

WIDTH="${WIDTH:-1200}"
HEIGHT="${HEIGHT:-1200}"
MAX_ITER="${MAX_ITER:-500}"
REPEATS="${REPEATS:-3}"
CSV="${CSV:-results.csv}"
MAX_RANKS="${1:-$(nproc)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${SCRIPT_DIR}/mandelbrot_benchmark.py"

rm -f "${CSV}"

echo "=== serial baseline ==="
python3 "${PY}" --mode serial \
    --width "${WIDTH}" --height "${HEIGHT}" --max-iter "${MAX_ITER}" \
    --repeats "${REPEATS}" --csv "${CSV}" --save-image --no-plot

ranks=1
while [ "${ranks}" -le "${MAX_RANKS}" ]; do
    echo
    echo "=== mpi -np ${ranks} ==="
    # --oversubscribe (OpenMPI) lets `ranks` exceed the physical/logical CPU
    # count so you can see the falloff past 2 vCPUs on a c7i-flex.large, not
    # just the well-behaved range. Drop it if your MPI is MPICH (which
    # oversubscribes by default) or if you only want in-bounds rank counts.
    mpirun -np "${ranks}" --oversubscribe python3 "${PY}" --mode mpi \
        --width "${WIDTH}" --height "${HEIGHT}" --max-iter "${MAX_ITER}" \
        --repeats "${REPEATS}" --csv "${CSV}" --no-plot
    ranks=$((ranks * 2))
done

echo
echo "=== summary ==="
python3 "${PY}" --summarize "${CSV}"

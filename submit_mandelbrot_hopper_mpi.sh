#!/bin/bash
#SBATCH -N 4 -n 160 -p normal
#SBATCH --job-name=mandelbrot_scaling_hopper_mpi
#SBATCH --output=logs/mandelbrot_scaling_hopper_mpi.out
#SBATCH --error=logs/mandelbrot_scaling_hopper_mpi.err
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=40
#SBATCH --ntasks-per-core=1
#SBATCH --exclusive
#SBATCH --mail-user=lbl59@cornell.edu
#SBATCH --mail-type=ALL

module load gnu9/9.3.0
module load openmpi4/4.0.5
source ~/py_env/bin/activate

rm -f figures/mandelbrot_scaling_hopper_mpi.csv figures/mandelbrot_memory_usage.csv figures/mandelbrot_overhead.csv

# reuse this one 4-node allocation for the 1, 2, and 4 node runs by handing
# mpirun a shrinking --host subset of the allocated nodes each time, each
# capped at TASKS_PER_NODE slots so every node contributes the same count
TASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))

for n in 1 2 4; do
    HOSTS=$(IFS=,; echo "${NODES[*]:0:$n}" | sed "s/,/:${TASKS_PER_NODE},/g")
    HOSTS="${HOSTS}:${TASKS_PER_NODE}"
    mpirun --host "${HOSTS}" -np $((n * TASKS_PER_NODE)) python3 mandelbrot_scaling_hopper_mpi.py
done

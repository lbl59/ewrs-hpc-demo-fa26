#!/bin/bash
#SBATCH -n 16 -N 1 -p normal
#SBATCH --job-name=mandelbrot_scaling_hopper_mpi
#SBATCH --output=logs/mandelbrot_scaling_hopper_mpi.out
#SBATCH --error=logs/mandelbrot_scaling_hopper_mpi.err
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-core=1
#SBATCH --exclusive
#SBATCH --mail-user=lbl59@cornell.edu
#SBATCH --mail-type=ALL

module load gnu9/9.3.0
module load openmpi4/4.0.5
source ~/mandelbrot-env/bin/activate

rm -f figures/mandelbrot_scaling_hopper_mpi.csv

for n in $(seq 1 16); do
    mpirun -n ${n} python3 mandelbrot_scaling_hopper_mpi.py
done

python3 mandelbrot_scaling_hopper_mpi.py --plot

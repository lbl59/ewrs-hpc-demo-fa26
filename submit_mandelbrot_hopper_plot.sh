#!/bin/bash
#SBATCH -N 1 -n 1 -p normal
#SBATCH --job-name=mandelbrot_scaling_hopper_plot
#SBATCH --output=logs/mandelbrot_scaling_hopper_plot.out
#SBATCH --error=logs/mandelbrot_scaling_hopper_plot.err
#SBATCH --time=00:05:00
#SBATCH --mail-user=lbl59@cornell.edu
#SBATCH --mail-type=ALL

module load gnu9/9.3.0
module load openmpi4/4.0.5

python3 mandelbrot_scaling_hopper_mpi.py --plot

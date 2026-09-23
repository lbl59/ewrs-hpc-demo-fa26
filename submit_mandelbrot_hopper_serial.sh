#!/bin/bash
#SBATCH -N 1 -n 1 -p normal
#SBATCH --job-name=mandelbrot_scaling_hopper_serial
#SBATCH --output=logs/mandelbrot_scaling_hopper_serial.out
#SBATCH --error=logs/mandelbrot_scaling_hopper_serial.err
#SBATCH --time=01:00:00
#SBATCH --mail-user=lbl59@cornell.edu
#SBATCH --mail-type=ALL

OMP_NUM_THREADS=40

source ~/mandelbrot-env/bin/activate

python3 mandelbrot_scaling_hopper_serial.py

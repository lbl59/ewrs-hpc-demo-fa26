# ewrs-hpc-demo-fa26

A simple parallelization exercise built around the Mandelbrot set. This exercise is split into two types of parallel implementations: 

### Workflow
1. `sbatch submit_mandelbrot_hopper_mpi.sh` allocates 4 Hopper nodes and runs
   the MPI Mandelbrot calculation at 1, 2, and 4 nodes, appending each run's
   time to `figures/mandelbrot_scaling_hopper_mpi.csv`.
2. `sbatch submit_mandelbrot_hopper_plot.sh` reads that CSV and (re)draws the
   runtime, speedup, and efficiency figures in `figures/`.

## :two: Amazon AWS shared-memory parallel implementation 
This exercises uses the Amazon (insert type of CPU here) with 16 nodes
compute-bound pixel-by-pixel calculation is implemented serially, then
parallelized with Python's `multiprocessing` across the cores of a single
machine

### Workflow 
(describr the Amazon AWS workflow here)

## :one: Cornell's Hopper Cluster hybrid parallel implementation 
This exercise uses the Python `mpi4py` library across multiple nodes of Cornell's Hopper
HPC cluster. The MPI version is scaled up (6144x6144 pixels, 2500 iterations
per pixel) so that runtime, speedup, and parallel efficiency can be compared
meaningfully across 1, 2, and 4 nodes rather than being swamped by
job/communication overhead.


## Files

| File | Description |
| --- | --- |
| `mandelbrot_serial.py` | Single-threaded baseline: computes the 1024x1024 Mandelbrot grid and saves an image of the set. |
| `mandelbrot_scaling.py` | Multiprocessing version that times the same grid across 1-16 CPU cores on one machine and plots runtime vs. CPU count. |
| `mandelbrot_scaling_hopper_mpi.py` | MPI version sized for Hopper (6144x6144, 2500 iterations). Each launch computes the grid at whatever node/rank count it was started with, records the runtime to CSV, and (with `--plot`) draws the runtime, speedup, and efficiency figures using `parallel_scaling_functions.py`. |
| `parallel_scaling_functions.py` | Reusable functions for parallel scaling analysis: `compute_speedup`, `compute_efficiency`, and `plot_speedup_efficiency`, which plots both metrics on one figure with two y-axes. |
| `submit_mandelbrot_hopper_mpi.sh` | Slurm batch script that allocates 4 Hopper nodes and runs the MPI scaling sweep at 1, 2, and 4 nodes. |
| `submit_mandelbrot_hopper_plot.sh` | Slurm batch script that regenerates the scaling figures from the CSV produced by the run above. |
| `requirements.txt` | Python dependencies (`numpy`, `matplotlib`, `mpi4py`). |
| `figures/` | Generated images and the `mandelbrot_scaling_hopper_mpi.csv` timing data. |

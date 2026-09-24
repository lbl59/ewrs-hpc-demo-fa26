# ewrs-hpc-demo-fa26

This repo contains two presentations and a simple parallelization exercise built around the Mandelbrot set. The exercise is split into two types of parallel implementations detailed below. It is highly recommended to review the two presentations for background information and detailed step-by-step instructions prior to completing the exercise: 

- `EWRS_Seminar_HPC_intro.pptx` provides an introduction to setting up HPC workflows, with examples relevant to the Cornell Hopper Cluster. 
- `EWRS_Seminar_HPC_demo.pptx` provides detailed step-by-step instructions on how to perform and measure parallel performance on both the cloud (Amazon AWS) and on the Hopper Cluster. 

Once you have reviewed these presentations, proceed with the exercise below.

## :one: Cornell's Hopper Cluster hybrid parallel implementation 
This exercise uses the Python `mpi4py` library across multiple nodes of Cornell's Hopper HPC cluster. Runtime, speedup, and parallel efficiency are compared across 1, 2, and 4 nodes. 

### Workflow
1. `sbatch submit_mandelbrot_hopper_single.sh` allocates 1 Hopper node and core to run the serial implementation of the Mandelbrot calculation. This forms the serial baseline to compare the parallel implementation agains.
2. `sbatch submit_mandelbrot_hopper_mpi.sh` allocates 4 Hopper nodes and runs the MPI Mandelbrot calculation at 1, 2, and 4 nodes, appending each run's time to `figures/mandelbrot_scaling_hopper_mpi.csv`.
3. `sbatch submit_mandelbrot_hopper_plot.sh` reads that CSV and (re)draws the runtime, speedup, and efficiency figures in `figures/`.

## :two: Amazon AWS shared-memory parallel implementation 
This  uses the Amazon c3.4xlarge with 16 nodes compute-bound pixel-by-pixel calculation is implemented serially, then parallelized with Python's `multiprocessing` across the cores of a single machine.

### Workflow 
1. `mandelbrot_serial.py` runs a smaller, serial version of the Mandelbrot problem on your selected AWS instance. 
2. `mandelbrot_scaling.py` runs the parallel multiprocessor version of the Mandelbrot problem across up to 16 cores on one AWS machine. It plots and save a figure showing runtime vs. core count in the `figures/` folder.

## Files

| File | Description |
| --- | --- |
| `mandelbrot_serial.py` | Single-threaded baseline: computes the 1024x1024 Mandelbrot grid and saves an image of the set. |
| `mandelbrot_scaling.py` | Multiprocessing version that times the same grid across 1-16 CPU cores on one AWS machine and plots runtime vs. CPU count. |
| `mandelbrot_scaling_hopper_mpi.py` | MPI version sized for Hopper (6144x6144, 2500 iterations). Each launch computes the grid at whatever node/rank count it was started with, records the runtime to CSV, and (with `--plot`) draws the runtime, speedup, and efficiency figures using `parallel_scaling_functions.py`. |
| `parallel_scaling_functions.py` | Reusable functions for parallel scaling analysis: `compute_speedup`, `compute_efficiency`, and `plot_speedup_efficiency`, which plots both metrics on one figure with two y-axes. |
| `submit_mandelbrot_hopper_mpi.sh` | Slurm batch script that allocates 4 Hopper nodes and runs the MPI scaling sweep at 1, 2, and 4 nodes. |
| `submit_mandelbrot_hopper_plot.sh` | Slurm batch script that regenerates the scaling figures from the CSV produced by the run above. |
| `requirements.txt` | Python dependencies (`numpy`, `matplotlib`, `mpi4py`). |
| `figures/` | Generated images and the `mandelbrot_scaling_hopper_mpi.csv` timing data. |

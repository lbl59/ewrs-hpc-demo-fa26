#!/usr/bin/env python3
"""Mandelbrot runtime vs. number of MPI ranks (1-16), for Cornell's Hopper cluster.

MPI rank count is fixed per launch, so this script does one thing per
invocation: compute the grid at whatever COMM_WORLD size it was launched
with (mpiexec -n N ...) and append the timing to a CSV. Run --plot
afterwards to read the CSV and draw ranks-vs-time.
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 1024
HEIGHT = 1024
MAX_ITER = 500
XMIN, XMAX = -2.0, 0.5
YMIN, YMAX = -1.25, 1.25
CSV_PATH = "figures/mandelbrot_scaling_hopper_mpi.csv"
OUT_PATH = "figures/mandelbrot_scaling_hopper_mpi.png"


def mandelbrot_point(cx, cy, max_iter):
    x, y = 0.0, 0.0
    for i in range(max_iter):
        x2 = x * x
        y2 = y * y
        if x2 + y2 > 4.0:
            return i
        y = 2.0 * x * y + cy
        x = x2 - y2 + cx
    return max_iter


def compute_row(row, width, height, xmin, xmax, ymin, ymax, max_iter):
    cy = ymin + (ymax - ymin) * row / (height - 1)
    values = [0] * width
    for col in range(width):
        cx = xmin + (xmax - xmin) * col / (width - 1)
        values[col] = mandelbrot_point(cx, cy, max_iter)
    return values


def row_partition(height, size):
    base, extra = divmod(height, size)
    counts = [base + 1 if r < extra else base for r in range(size)]
    starts = [sum(counts[:r]) for r in range(size)]
    return starts, counts


def run_mpi():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()   # get the process number (0 to size-1)
    size = comm.Get_size()   # get the total number of processes

    # chunk the rows of the grid among the ranks
    # so each rank computes a contiguous block of rows
    starts, counts = row_partition(HEIGHT, size)
    # each rank computes its own block of rows
    my_start, my_count = starts[rank], counts[rank]

    # synchronize to ensure all ranks start timing at the same moment
    comm.Barrier()
    t0 = MPI.Wtime()

    local_rows = np.empty((my_count, WIDTH), dtype=np.int32)
    for i in range(my_count):
        row = my_start + i
        local_rows[i, :] = compute_row(row, WIDTH, HEIGHT, XMIN, XMAX, YMIN, YMAX, MAX_ITER)

    elapsed = MPI.Wtime() - t0
    elapsed = comm.allreduce(elapsed, op=MPI.MAX)

    if rank == 0:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["ranks", "seconds"])
            writer.writerow([size, f"{elapsed:.6f}"])
        print(f"ranks={size}: {elapsed:.3f}s (appended to {CSV_PATH})")


def plot_scaling():
    ranks, times = [], []
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranks.append(int(row["ranks"]))
            times.append(float(row["seconds"]))
    order = sorted(range(len(ranks)), key=lambda i: ranks[i])
    ranks = [ranks[i] for i in order]
    times = [times[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(ranks, times, marker="o", color="#5B7FA6", linewidth=2)
    ax.set_xlabel("Number of MPI ranks (CPUs)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Mandelbrot ({WIDTH}x{HEIGHT}) runtime vs. CPU count -- Hopper")
    ax.set_xticks(ranks)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Figure written to {OUT_PATH}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true",
                    help="skip computing; read the CSV and (re)draw the scaling plot")
    args = p.parse_args()

    if args.plot:
        plot_scaling()
    else:
        run_mpi()


if __name__ == "__main__":
    main()

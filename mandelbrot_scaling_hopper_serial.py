#!/usr/bin/env python3
"""Serial Mandelbrot baseline at the same problem size used on Hopper
(6144x6144, 2500 iterations), for comparison against the MPI scaling runs."""

import csv
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mandelbrot_scaling_hopper_mpi import (
    compute_row, WIDTH, HEIGHT, MAX_ITER, XMIN, XMAX, YMIN, YMAX,
)
from parallel_scaling_functions import get_peak_memory_mb, record_memory_usage

CSV_PATH = "figures/mandelbrot_scaling_hopper_serial.csv"
MEMORY_CSV_PATH = "figures/mandelbrot_memory_usage_serial.csv"
OUT_PATH = "figures/mandelbrot_hopper_serial.png"


def main():
    grid = np.empty((HEIGHT, WIDTH), dtype=np.int32)

    t0 = time.perf_counter()
    for row in range(HEIGHT):
        grid[row, :] = compute_row(row, WIDTH, HEIGHT, XMIN, XMAX, YMIN, YMAX, MAX_ITER)
    elapsed = time.perf_counter() - t0
    peak_mb = get_peak_memory_mb()

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["nodes", "ranks", "seconds"])
        writer.writerow([1, 1, f"{elapsed:.6f}"])
    record_memory_usage(MEMORY_CSV_PATH, 1, 1, [peak_mb])

    print(f"serial: {elapsed:.3f}s peak_mem={peak_mb:.1f}MB (appended to {CSV_PATH})")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap="magma", extent=(XMIN, XMAX, YMIN, YMAX), origin="lower")
    ax.set_title(f"Mandelbrot set ({WIDTH}x{HEIGHT}) -- serial, Hopper")
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Image written to {OUT_PATH}")


if __name__ == "__main__":
    main()

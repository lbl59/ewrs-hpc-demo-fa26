#!/usr/bin/env python3
"""Mandelbrot runtime vs. number of CPUs (1-16, multiprocessing), plotted."""

import time
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 1024
HEIGHT = 1024
MAX_ITER = 500
XMIN, XMAX = -2.0, 0.5
YMIN, YMAX = -1.25, 1.25
MAX_CPUS = 16
OUT_PATH = "figures/mandelbrot_scaling.png"


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


def compute_row(args):
    row, width, height, xmin, xmax, ymin, ymax, max_iter = args
    cy = ymin + (ymax - ymin) * row / (height - 1)
    values = [0] * width
    for col in range(width):
        cx = xmin + (xmax - xmin) * col / (width - 1)
        values[col] = mandelbrot_point(cx, cy, max_iter)
    return row, values


def row_args():
    return [
        (row, WIDTH, HEIGHT, XMIN, XMAX, YMIN, YMAX, MAX_ITER)
        for row in range(HEIGHT)
    ]


def compute_grid(n_cpus):
    grid = np.empty((HEIGHT, WIDTH), dtype=np.int32)
    start = time.perf_counter()
    if n_cpus == 1:
        for args in row_args():
            row, values = compute_row(args)
            grid[row, :] = values
    else:
        # Pool of CPUs to compute rows in parallel
        with mp.Pool(processes=n_cpus) as pool:
            for row, values in pool.imap_unordered(compute_row, row_args()):
                grid[row, :] = values
    elapsed = time.perf_counter() - start
    return grid, elapsed


def plot_scaling(cpu_counts, times, out_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(cpu_counts, times, marker="o", color="#5B7FA6", linewidth=2)
    ax.set_xlabel("Number of CPUs")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Mandelbrot ({WIDTH}x{HEIGHT}) runtime vs. CPU count")
    ax.set_xticks(cpu_counts)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    cpu_counts = list(range(1, MAX_CPUS + 1))
    times = []
    for n in cpu_counts:
        _, elapsed = compute_grid(n)
        print(f"CPUs={n:2d}: {elapsed:.3f}s")
        times.append(elapsed)

    plot_scaling(cpu_counts, times, OUT_PATH)
    print(f"Figure written to {OUT_PATH}")


if __name__ == "__main__":
    main()

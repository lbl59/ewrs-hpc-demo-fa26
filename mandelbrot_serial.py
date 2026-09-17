#!/usr/bin/env python3
"""Serial computation and plot of the Mandelbrot set at 1024x1024 pixels."""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 1024
HEIGHT = 1024
MAX_ITER = 500   # max iterations per pixel
XMIN, XMAX = -2.0, 0.5
YMIN, YMAX = -1.25, 1.25
OUT_PATH = "figures/mandelbrot_serial.png"

def mandelbrot_point(cx, cy, max_iter):
    x, y = 0.0, 0.0
    for i in range(max_iter):
        x2 = x * x
        y2 = y * y
        if x2 + y2 > 4.0:  
            # Point has escaped the Mandelbrot set, return the number of iterations
            return i
        y = 2.0 * x * y + cy
        x = x2 - y2 + cx
    return max_iter


def compute_grid(width, height, xmin, xmax, ymin, ymax, max_iter):
    grid = np.empty((height, width), dtype=np.int32)
    for row in range(height):
        cy = ymin + (ymax - ymin) * row / (height - 1)
        for col in range(width):
            cx = xmin + (xmax - xmin) * col / (width - 1)
            grid[row, col] = mandelbrot_point(cx, cy, max_iter)
    return grid


def plot_grid(grid, xmin, xmax, ymin, ymax, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap="magma", extent=(xmin, xmax, ymin, ymax), origin="lower")
    ax.set_title(f"Mandelbrot set ({grid.shape[1]}x{grid.shape[0]})")
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    start = time.perf_counter()
    grid = compute_grid(WIDTH, HEIGHT, XMIN, XMAX, YMIN, YMAX, MAX_ITER)
    elapsed = time.perf_counter() - start
    print(f"Computed {WIDTH}x{HEIGHT} grid in {elapsed:.3f}s")

    plot_grid(grid, XMIN, XMAX, YMIN, YMAX, OUT_PATH)
    print(f"Image written to {OUT_PATH}")


if __name__ == "__main__":
    main()

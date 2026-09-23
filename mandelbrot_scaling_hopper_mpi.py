#!/usr/bin/env python3
"""Mandelbrot runtime vs. number of Hopper compute nodes (1, 2, 4).

Node count is fixed per launch, so this script does one thing per
invocation: compute the grid at whatever COMM_WORLD size it was launched
with (srun -N <nodes> ...), and append the timing, peak memory use, and
job/communication overhead to CSVs. Run --plot afterwards to read those
CSVs and draw the runtime/speedup/efficiency, memory, and overhead figures.
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parallel_scaling_functions import (
    compute_speedup,
    compute_efficiency,
    plot_speedup_efficiency,
    get_peak_memory_mb,
    record_memory_usage,
    compute_memory_efficiency,
    plot_memory_usage,
    record_overhead,
    plot_overhead,
)

WIDTH = 6144
HEIGHT = 6144
MAX_ITER = 2500
XMIN, XMAX = -2.0, 0.5
YMIN, YMAX = -1.25, 1.25

CSV_PATH = "figures/mandelbrot_scaling_hopper_mpi.csv"
MEMORY_CSV_PATH = "figures/mandelbrot_memory_usage.csv"
OVERHEAD_CSV_PATH = "figures/mandelbrot_overhead.csv"

OUT_PATH = "figures/mandelbrot_scaling_hopper_mpi.png"
SPEEDUP_OUT_PATH = "figures/mandelbrot_speedup_efficiency.png"
MEMORY_OUT_PATH = "figures/mandelbrot_memory_usage.png"
OVERHEAD_OUT_PATH = "figures/mandelbrot_overhead.png"


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
    rank = comm.Get_rank()
    size = comm.Get_size()

    t_start = MPI.Wtime()

    # number of distinct nodes actually used, measured rather than
    # trusted to a Slurm env var, since ranks may span srun sub-steps
    hosts = comm.gather(MPI.Get_processor_name(), root=0)
    n_nodes = len(set(hosts)) if rank == 0 else None
    n_nodes = comm.bcast(n_nodes, root=0)

    starts, counts = row_partition(HEIGHT, size)
    my_start, my_count = starts[rank], counts[rank]

    comm.Barrier()
    t_compute_start = MPI.Wtime()

    local_rows = np.empty((my_count, WIDTH), dtype=np.int32)
    for i in range(my_count):
        row = my_start + i
        local_rows[i, :] = compute_row(row, WIDTH, HEIGHT, XMIN, XMAX, YMIN, YMAX, MAX_ITER)

    t_compute_end = MPI.Wtime()
    my_peak_mb = get_peak_memory_mb()

    compute_time = comm.allreduce(t_compute_end - t_compute_start, op=MPI.MAX)
    setup_time = comm.allreduce(t_compute_start - t_start, op=MPI.MAX)
    peak_mb_values = comm.gather(my_peak_mb, root=0)

    t_end = MPI.Wtime()
    commpost_time = comm.allreduce(t_end - t_compute_end, op=MPI.MAX)
    overhead_time = setup_time + commpost_time

    if rank == 0:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        write_header = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["nodes", "ranks", "seconds"])
            writer.writerow([n_nodes, size, f"{compute_time:.6f}"])

        record_memory_usage(MEMORY_CSV_PATH, n_nodes, size, peak_mb_values)
        record_overhead(OVERHEAD_CSV_PATH, n_nodes, size, compute_time, overhead_time)

        print(f"nodes={n_nodes} ranks={size}: compute={compute_time:.3f}s "
              f"overhead={overhead_time:.3f}s peak_mem_max={max(peak_mb_values):.1f}MB")


def _read_csv_columns(path, columns):
    rows = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row["nodes"])
            rows.setdefault(n, []).append([float(row[c]) for c in columns])
    nodes = sorted(rows)
    averaged = [
        [sum(v[i] for v in rows[n]) / len(rows[n]) for i in range(len(columns))]
        for n in nodes
    ]
    return nodes, list(zip(*averaged)) if averaged else [[] for _ in columns]


def plot_scaling():
    nodes, (times,) = _read_csv_columns(CSV_PATH, ["seconds"])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(nodes, times, marker="o", color="#5B7FA6", linewidth=2)
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Mandelbrot ({WIDTH}x{HEIGHT}, {MAX_ITER} iter) runtime vs. node count -- Hopper")
    ax.set_xticks(nodes)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Figure written to {OUT_PATH}")

    speedup = compute_speedup(nodes, times)
    efficiency = compute_efficiency(nodes, speedup)
    plot_speedup_efficiency(
        nodes, speedup, efficiency, SPEEDUP_OUT_PATH,
        title=f"Mandelbrot ({WIDTH}x{HEIGHT}, {MAX_ITER} iter) speedup & efficiency -- Hopper",
    )
    print(f"Figure written to {SPEEDUP_OUT_PATH}")

    mem_nodes, (mean_mb, total_mb) = _read_csv_columns(MEMORY_CSV_PATH, ["mean_mb", "total_mb"])
    memory_efficiency = compute_memory_efficiency(list(total_mb))
    plot_memory_usage(
        mem_nodes, list(mean_mb), memory_efficiency, MEMORY_OUT_PATH,
        title=f"Mandelbrot ({WIDTH}x{HEIGHT}) memory use vs. node count -- Hopper",
    )
    print(f"Figure written to {MEMORY_OUT_PATH}")

    oh_nodes, (compute_s, overhead_s) = _read_csv_columns(
        OVERHEAD_CSV_PATH, ["compute_seconds", "overhead_seconds"]
    )
    plot_overhead(
        oh_nodes, list(compute_s), list(overhead_s), OVERHEAD_OUT_PATH,
        title=f"Mandelbrot ({WIDTH}x{HEIGHT}) job/communication overhead vs. node count -- Hopper",
    )
    print(f"Figure written to {OVERHEAD_OUT_PATH}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true",
                    help="skip computing; read the CSVs and (re)draw the scaling plots")
    args = p.parse_args()

    if args.plot:
        plot_scaling()
    else:
        run_mpi()


if __name__ == "__main__":
    main()

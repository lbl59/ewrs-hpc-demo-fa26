#!/usr/bin/env python3
"""
mandelbrot_benchmark.py
========================

Serial vs. MPI-parallel Mandelbrot-set benchmark.

Computes an escape-time Mandelbrot grid of fixed size once with a plain
single-process loop, and again distributed across MPI ranks via mpi4py
(row-range decomposition + gather), reporting wall-clock time for each.

Why a plain Python loop and not NumPy vectorization for the per-pixel math:
a vectorized version is fast enough on a single core that parallelization
overhead (process/rank startup, communication) can swamp the real effect
you're trying to measure. The scalar loop below gives every pixel genuine,
non-trivial CPU work, which is what makes a serial-vs-parallel comparison
meaningful in the first place.

A note on MPI process counts (read this before you look for --workers)
------------------------------------------------------------------------
mpi4py processes are separate OS processes launched all at once by
`mpirun`/`mpiexec`/`srun -n N`, and that count N is fixed for the life of
the job -- there's no in-process pool you can resize on the fly the way
multiprocessing.Pool(processes=N) allows. So this script no longer sweeps
multiple worker counts in a single invocation. Instead:

  --mode serial   a single, non-MPI process. Run as a plain `python3`
                  command -- do NOT launch it with mpirun.
  --mode mpi      an MPI run at whatever size `mpirun -np N` (or
                  `srun -n N`) launched. Rank 0 does the printing/saving;
                  every rank does its share of the computation.

To reproduce a full serial-vs-N-ranks sweep, invoke this script once per
process count (serial once, then mpirun -np 1, -np 2, ...), each time
appending to the same --csv file, then run:

    python3 mandelbrot_benchmark.py --summarize results.csv

to print the combined table and redraw the speedup chart. run_mpi_benchmark.sh
in this same folder automates exactly that sequence.

Tuned for an AWS c7i-flex.large instance
-----------------------------------------
c7i-flex.large has 2 vCPUs (SMT siblings on one physical core, Sapphire-
Rapids-based Xeon Scalable) and 4 GiB RAM. That means:

  * Expect close-to-linear speedup from 1 -> 2 MPI ranks, then diminishing
    or negative returns beyond that -- 2 vCPUs is all there is. Going to
    4 ranks (oversubscribing) shows that falloff directly instead of just
    asserting it.
  * The two vCPUs share one physical core, so even the 1->2 speedup will
    usually land below a clean 2.0x -- that gap is a real, useful data
    point about this instance, not a bug in the script.
  * Keep --width/--height and --max-iter modest for quick iteration
    (e.g. 800x800, max-iter 300) and scale up once you've confirmed the
    setup works; a 2000x2000 grid at max-iter 1000 can take minutes serially.
  * 4 GiB RAM is comfortable for any grid size reasonable to compute here
    (a 4000x4000 int32 grid is ~64 MB), so memory won't be the constraint.

Usage
-----
    python3 mandelbrot_benchmark.py --mode serial --csv results.csv --save-image
    mpirun -np 2 python3 mandelbrot_benchmark.py --mode mpi --csv results.csv
    python3 mandelbrot_benchmark.py --summarize results.csv

    # or just:
    ./run_mpi_benchmark.sh

Outputs
-------
    Printed timing (mean +/- stdev over --repeats) for whichever mode ran
    mandelbrot.png          -- rendered fractal (only from --save-image runs)
    mandelbrot_speedup.png  -- bar chart of time + line of speedup vs. rank count
                                (produced by --summarize, from the combined CSV)
    results.csv (optional)  -- raw timing data, appended to across invocations
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# matplotlib is optional at import time so the benchmark itself still runs
# (e.g. over SSH with no display) even if plotting isn't needed.
try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe backend; works on a bare EC2 instance
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# --------------------------------------------------------------------------
# Core Mandelbrot kernel (deliberately scalar, not vectorized -- see module
# docstring for why)
# --------------------------------------------------------------------------

def mandelbrot_point(cx: float, cy: float, max_iter: int) -> int:
    """Escape-time iteration count for one complex point c = cx + i*cy."""
    x, y = 0.0, 0.0
    for i in range(max_iter):
        x2 = x * x
        y2 = y * y
        if x2 + y2 > 4.0:
            return i
        y = 2.0 * x * y + cy
        x = x2 - y2 + cx
    return max_iter


def compute_row(args: Tuple[int, int, int, float, float, float, float, int]) -> Tuple[int, List[int]]:
    """Compute one full row (fixed pixel y) of the Mandelbrot grid."""
    row_idx, width, height, xmin, xmax, ymin, ymax, max_iter = args
    cy = ymin + (ymax - ymin) * row_idx / (height - 1)
    row = [0] * width
    for col in range(width):
        cx = xmin + (xmax - xmin) * col / (width - 1)
        row[col] = mandelbrot_point(cx, cy, max_iter)
    return row_idx, row


@dataclass
class GridSpec:
    width: int = 1000
    height: int = 1000
    xmin: float = -2.0
    xmax: float = 0.5
    ymin: float = -1.25
    ymax: float = 1.25
    max_iter: int = 500

    def row_args(self) -> List[Tuple[int, int, int, float, float, float, float, int]]:
        return [
            (y, self.width, self.height, self.xmin, self.xmax, self.ymin, self.ymax, self.max_iter)
            for y in range(self.height)
        ]


def _row_partition(height: int, size: int) -> Tuple[List[int], List[int]]:
    """Split `height` rows into `size` contiguous, near-equal chunks.

    Returns (starts, counts) -- the first `height % size` ranks get one
    extra row, so the partition never differs by more than one row between
    ranks even when height doesn't divide evenly.
    """
    base, extra = divmod(height, size)
    counts = [base + 1 if r < extra else base for r in range(size)]
    starts = [sum(counts[:r]) for r in range(size)]
    return starts, counts


# --------------------------------------------------------------------------
# Serial and MPI compute paths
# --------------------------------------------------------------------------

def compute_serial(spec: GridSpec) -> Tuple[np.ndarray, float]:
    grid = np.empty((spec.height, spec.width), dtype=np.int32)
    start = time.perf_counter()
    for args in spec.row_args():
        row_idx, row = compute_row(args)
        grid[row_idx, :] = row
    elapsed = time.perf_counter() - start
    return grid, elapsed


def compute_mpi(spec: GridSpec) -> Tuple[Optional[np.ndarray], float, int, int]:
    """Compute the grid across all MPI ranks in COMM_WORLD.

    Each rank computes a contiguous band of rows, then rank 0 gathers them
    into the full grid. Returns (grid, elapsed, rank, size): `grid` is the
    assembled (height, width) array on rank 0 and None on every other rank;
    `elapsed` is the wall-clock time (max across ranks, via Allreduce) and
    is identical on every rank, so any rank can log it consistently.

    Imports mpi4py lazily so `--mode serial` keeps working on machines that
    don't have MPI installed at all.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    starts, counts = _row_partition(spec.height, size)
    my_start, my_count = starts[rank], counts[rank]

    comm.Barrier()  # align the start line across ranks before timing
    t0 = MPI.Wtime()

    local_rows = np.empty((my_count, spec.width), dtype=np.int32)
    for i in range(my_count):
        y = my_start + i
        _, row = compute_row(
            (y, spec.width, spec.height, spec.xmin, spec.xmax, spec.ymin, spec.ymax, spec.max_iter)
        )
        local_rows[i, :] = row

    gathered = comm.gather(local_rows, root=0)
    elapsed_local = MPI.Wtime() - t0
    # The job isn't "done" until the slowest rank finishes -- take the max,
    # not rank 0's own time, so an unevenly-loaded run is reported honestly.
    elapsed = comm.allreduce(elapsed_local, op=MPI.MAX)

    grid = None
    if rank == 0:
        grid = np.empty((spec.height, spec.width), dtype=np.int32)
        for r in range(size):
            grid[starts[r]:starts[r] + counts[r], :] = gathered[r]

    return grid, elapsed, rank, size


# --------------------------------------------------------------------------
# Timing bookkeeping, CSV, and summary
# --------------------------------------------------------------------------

@dataclass
class TimingResult:
    label: str
    workers: int
    times: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.times)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0


def append_csv(path: str, label: str, workers: int, times: List[float]) -> None:
    """Append one configuration's repeats to a shared CSV, writing the
    header only if the file doesn't exist yet. Safe to call once per
    process/mpirun invocation -- each call is a separate open/close.
    """
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["configuration", "workers", "repeat", "seconds"])
        for i, t in enumerate(times):
            writer.writerow([label, workers, i + 1, f"{t:.6f}"])
    print(f"Appended {len(times)} row(s) for '{label}' to {path}")


def read_csv_grouped(path: str) -> List[TimingResult]:
    groups: "dict[str, TimingResult]" = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["configuration"]
            workers = int(row["workers"])
            seconds = float(row["seconds"])
            if label not in groups:
                groups[label] = TimingResult(label=label, workers=workers)
            groups[label].times.append(seconds)
    return sorted(groups.values(), key=lambda r: (r.workers, r.label != "serial"))


def _baseline(results: List[TimingResult]) -> TimingResult:
    """Prefer an explicit 'serial' row; otherwise fall back to whichever
    configuration used the fewest workers (so --summarize still works on a
    CSV built from MPI-only runs with no serial baseline in it)."""
    for r in results:
        if r.label == "serial":
            return r
    return min(results, key=lambda r: r.workers)


def print_summary(results: List[TimingResult]) -> None:
    baseline = _baseline(results).mean
    print("\n=== Summary ===")
    header = f"{'Configuration':<14}{'Mean (s)':>10}{'Stdev (s)':>11}{'Speedup':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        speedup = baseline / r.mean if r.mean > 0 else float("nan")
        print(f"{r.label:<14}{r.mean:>10.3f}{r.stdev:>11.3f}{speedup:>9.2f}x")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

# A small, fixed categorical palette: bars stay one muted neutral hue (time is
# the metric being measured, not a categorical identity), the speedup line
# uses a single distinct accent so it reads clearly against the bars.
BAR_COLOR = "#5B7FA6"     # muted blue -- timing bars
LINE_COLOR = "#D97757"    # warm accent -- speedup line
IDEAL_COLOR = "#9CA3AF"   # neutral gray -- ideal-speedup reference


def plot_fractal(grid: np.ndarray, spec: GridSpec, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7 * spec.height / spec.width))
    # Sequential colormap: iteration count is a magnitude, not a category.
    ax.imshow(
        grid,
        cmap="magma",
        extent=(spec.xmin, spec.xmax, spec.ymin, spec.ymax),
        origin="lower",
    )
    ax.set_title(f"Mandelbrot set ({spec.width}x{spec.height}, max_iter={spec.max_iter})")
    ax.set_xlabel("Re(c)")
    ax.set_ylabel("Im(c)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Fractal image written to {out_path}")


def plot_speedup(results: List[TimingResult], out_path: str, cpu_count: int) -> None:
    baseline = _baseline(results).mean
    labels = [r.label for r in results]
    means = [r.mean for r in results]
    stdevs = [r.stdev for r in results]
    speedups = [baseline / m if m > 0 else float("nan") for m in means]
    x = np.arange(len(results))

    fig, ax1 = plt.subplots(figsize=(8, 5.5))

    bars = ax1.bar(x, means, yerr=stdevs, capsize=4, color=BAR_COLOR, width=0.55, zorder=3)
    ax1.set_ylabel("Wall-clock time (s)", color=BAR_COLOR)
    ax1.tick_params(axis="y", labelcolor=BAR_COLOR)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.spines["top"].set_visible(False)
    ax1.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    # Headroom above the tallest bar so its direct label never collides
    # with the title or the legend above the plot.
    ax1.set_ylim(0, max(means) * 1.22)

    # Direct labels on bars instead of a legend for the single series.
    for rect, m in zip(bars, means):
        ax1.annotate(
            f"{m:.2f}s",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#374151",
        )

    ax2 = ax1.twinx()
    line_measured, = ax2.plot(x, speedups, color=LINE_COLOR, marker="o", markersize=6,
                               linewidth=2, zorder=4, label="Measured speedup")
    workers_by_idx = [r.workers for r in results]
    line_ideal, = ax2.plot(x, workers_by_idx, color=IDEAL_COLOR, linestyle="--", linewidth=1.5,
                            zorder=2, label="Ideal (linear) speedup")
    ax2.set_ylabel("Speedup vs. baseline (x)", color=LINE_COLOR)
    ax2.tick_params(axis="y", labelcolor=LINE_COLOR)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, max(max(speedups), max(workers_by_idx)) * 1.15)

    # Legend placed above the plot area (outside the data region) so it
    # never overlaps bars, labels, or the speedup line.
    fig.legend(
        handles=[line_measured, line_ideal],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    ax1.set_title(f"Serial vs. MPI Mandelbrot benchmark ({cpu_count} vCPUs detected)", pad=28)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Speedup chart written to {out_path}")


# --------------------------------------------------------------------------
# Mode drivers
# --------------------------------------------------------------------------

def run_serial(spec: GridSpec, args: argparse.Namespace) -> None:
    print(f"Grid: {spec.width}x{spec.height}, max_iter={spec.max_iter}")
    print(f"Platform: {platform.platform()}, Python {platform.python_version()}")
    print()

    times: List[float] = []
    grid = None
    for r in range(args.repeats):
        grid, elapsed = compute_serial(spec)
        times.append(elapsed)
        print(f"[serial] repeat {r + 1}/{args.repeats}: {elapsed:.3f}s")

    result = TimingResult(label="serial", workers=1, times=times)
    print(f"\nserial: mean={result.mean:.3f}s stdev={result.stdev:.3f}s")

    if args.csv:
        append_csv(args.csv, "serial", 1, times)
    if args.save_image and HAVE_MPL and grid is not None:
        plot_fractal(grid, spec, f"{args.out_prefix}mandelbrot.png")


def run_mpi(spec: GridSpec, args: argparse.Namespace) -> None:
    try:
        from mpi4py import MPI
    except ImportError:
        print(
            "mpi4py is not installed. Install it (pip install mpi4py, after an MPI "
            "implementation such as OpenMPI or MPICH is available) and launch this "
            "mode with mpirun/mpiexec/srun, e.g.: mpirun -np 2 python3 "
            f"{os.path.basename(__file__)} --mode mpi",
            file=sys.stderr,
        )
        sys.exit(1)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Grid: {spec.width}x{spec.height}, max_iter={spec.max_iter}")
        print(f"MPI world size: {size} rank(s)")
        print(f"Platform: {platform.platform()}, Python {platform.python_version()}")
        print()

    times: List[float] = []
    grid = None
    for r in range(args.repeats):
        grid, elapsed, rank, size = compute_mpi(spec)
        times.append(elapsed)
        if rank == 0:
            print(f"[mpi size={size}] repeat {r + 1}/{args.repeats}: {elapsed:.3f}s")

    if rank == 0:
        label = f"mpi-{size}"
        result = TimingResult(label=label, workers=size, times=times)
        print(f"\n{label}: mean={result.mean:.3f}s stdev={result.stdev:.3f}s")

        if args.csv:
            append_csv(args.csv, label, size, times)
        if args.save_image and HAVE_MPL and grid is not None:
            plot_fractal(grid, spec, f"{args.out_prefix}mandelbrot.png")


def summarize(csv_path: str, out_prefix: str, no_plot: bool) -> None:
    results = read_csv_grouped(csv_path)
    if not results:
        print(f"No rows found in {csv_path}", file=sys.stderr)
        sys.exit(1)
    print_summary(results)
    if not no_plot and HAVE_MPL:
        plot_speedup(results, f"{out_prefix}mandelbrot_speedup.png", os.cpu_count() or 1)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark serial vs. MPI-parallel (mpi4py) Mandelbrot-set computation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--width", type=int, default=800, help="grid width in pixels")
    p.add_argument("--height", type=int, default=800, help="grid height in pixels")
    p.add_argument("--max-iter", type=int, default=300, help="max escape-time iterations per point")
    p.add_argument("--xmin", type=float, default=-2.0)
    p.add_argument("--xmax", type=float, default=0.5)
    p.add_argument("--ymin", type=float, default=-1.25)
    p.add_argument("--ymax", type=float, default=1.25)
    p.add_argument(
        "--mode",
        choices=["serial", "mpi"],
        default="serial",
        help="'serial': plain single process, run with `python3 ...` directly. "
             "'mpi': MPI run, must be launched with mpirun/mpiexec/srun -n N.",
    )
    p.add_argument("--repeats", type=int, default=3, help="repeats for this run, for mean/stdev")
    p.add_argument("--save-image", action="store_true",
                    help="save the computed grid as mandelbrot.png (rank 0 only in --mode mpi)")
    p.add_argument("--csv", type=str, default=None,
                    help="CSV file to append this run's timing to (created with a header if new)")
    p.add_argument("--out-prefix", type=str, default="", help="prefix for output image filenames")
    p.add_argument("--no-plot", action="store_true",
                    help="in --summarize mode, skip drawing the speedup chart (table is still printed); "
                         "has no effect on --save-image, which is an independent opt-in")
    p.add_argument("--summarize", type=str, default=None, metavar="CSV",
                    help="skip computing anything; read a CSV built from prior serial/mpi runs, "
                         "print the combined summary table, and redraw the speedup chart")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.summarize:
        summarize(args.summarize, args.out_prefix, args.no_plot)
        return 0

    if not HAVE_MPL and not args.no_plot:
        print("matplotlib not available -- skipping plots (timing results still printed). "
              "Install with: pip install matplotlib", file=sys.stderr)

    spec = GridSpec(
        width=args.width,
        height=args.height,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        max_iter=args.max_iter,
    )

    if args.mode == "serial":
        run_serial(spec, args)
    else:
        run_mpi(spec, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())

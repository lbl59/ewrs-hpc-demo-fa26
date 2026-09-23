#!/usr/bin/env python3
"""Parallel scaling analysis: runtime speedup/efficiency, memory use
efficiency, and job/communication overhead -- computation and plots."""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_speedup(nodes, times):
    baseline_time = times[0]
    return [baseline_time / t for t in times]


def compute_efficiency(nodes, speedup):
    baseline_nodes = nodes[0]
    return [s * baseline_nodes / n for s, n in zip(speedup, nodes)]


def plot_speedup_efficiency(nodes, speedup, efficiency, out_path, title=None):
    fig, ax1 = plt.subplots(figsize=(7, 5.5))
    ax2 = ax1.twinx()

    ideal = [n / nodes[0] for n in nodes]
    ax1.plot(nodes, ideal, linestyle="--", color="#A6ADBB", label="Ideal speedup")
    line_speedup, = ax1.plot(nodes, speedup, marker="o", color="#5B7FA6",
                              linewidth=2, label="Speedup")
    line_efficiency, = ax2.plot(nodes, efficiency, marker="s", color="#C97064",
                                 linewidth=2, label="Efficiency")

    ax1.set_xlabel("Number of nodes")
    ax1.set_ylabel("Speedup", color="#5B7FA6")
    ax1.tick_params(axis="y", colors="#5B7FA6")
    ax2.set_ylabel("Efficiency", color="#C97064")
    ax2.tick_params(axis="y", colors="#C97064")
    ax2.set_ylim(0, 1.1)
    ax1.set_xticks(nodes)
    ax1.grid(True, color="#E5E7EB", linewidth=0.8)
    if title:
        ax1.set_title(title)

    lines = [line_speedup, line_efficiency]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def get_peak_memory_mb():
    """Peak resident set size of the calling process so far, in MB."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def record_memory_usage(csv_path, nodes, ranks, peak_mb_values):
    """Append one row of per-rank peak memory stats for a run."""
    max_mb = max(peak_mb_values)
    mean_mb = sum(peak_mb_values) / len(peak_mb_values)
    total_mb = sum(peak_mb_values)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["nodes", "ranks", "max_mb", "mean_mb", "total_mb"])
        writer.writerow([nodes, ranks, f"{max_mb:.2f}", f"{mean_mb:.2f}", f"{total_mb:.2f}"])
    return max_mb, mean_mb, total_mb


def compute_memory_efficiency(total_memory_mb):
    """How close the total memory footprint stays to the baseline run's as
    ranks are added -- 1.0 means parallelizing added no extra memory."""
    baseline_total = total_memory_mb[0]
    return [baseline_total / m for m in total_memory_mb]


def plot_memory_usage(nodes, mean_mb_per_rank, memory_efficiency, out_path, title=None):
    fig, ax1 = plt.subplots(figsize=(7, 5.5))
    ax2 = ax1.twinx()

    line_mem, = ax1.plot(nodes, mean_mb_per_rank, marker="o", color="#5B7FA6",
                          linewidth=2, label="Mean peak memory / rank (MB)")
    line_eff, = ax2.plot(nodes, memory_efficiency, marker="s", color="#C97064",
                          linewidth=2, label="Memory efficiency")

    ax1.set_xlabel("Number of nodes")
    ax1.set_ylabel("Mean peak memory per rank (MB)", color="#5B7FA6")
    ax1.tick_params(axis="y", colors="#5B7FA6")
    ax2.set_ylabel("Memory efficiency", color="#C97064")
    ax2.tick_params(axis="y", colors="#C97064")
    ax2.set_ylim(0, 1.1)
    ax1.set_xticks(nodes)
    ax1.grid(True, color="#E5E7EB", linewidth=0.8)
    if title:
        ax1.set_title(title)

    lines = [line_mem, line_eff]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def record_overhead(csv_path, nodes, ranks, compute_seconds, overhead_seconds):
    """Append one row splitting a run's time into compute vs.
    job/communication overhead (startup, barriers, reductions)."""
    total_seconds = compute_seconds + overhead_seconds

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["nodes", "ranks", "compute_seconds", "overhead_seconds", "total_seconds"])
        writer.writerow([nodes, ranks, f"{compute_seconds:.6f}", f"{overhead_seconds:.6f}", f"{total_seconds:.6f}"])
    return total_seconds


def compute_overhead_fraction(compute_seconds, overhead_seconds):
    return [oh / (c + oh) for c, oh in zip(compute_seconds, overhead_seconds)]


def plot_overhead(nodes, compute_seconds, overhead_seconds, out_path, title=None):
    fig, ax1 = plt.subplots(figsize=(7, 5.5))
    ax2 = ax1.twinx()

    ax1.bar(nodes, compute_seconds, color="#5B7FA6", label="Compute time")
    ax1.bar(nodes, overhead_seconds, bottom=compute_seconds, color="#C97064",
            label="Job/communication overhead")

    fraction_pct = [f * 100 for f in compute_overhead_fraction(compute_seconds, overhead_seconds)]
    line, = ax2.plot(nodes, fraction_pct, marker="o", color="#2E2E2E",
                      linewidth=2, label="Overhead (% of total)")

    ax1.set_xlabel("Number of nodes")
    ax1.set_ylabel("Time (s)")
    ax2.set_ylabel("Overhead fraction (%)")
    ax2.set_ylim(0, max(100.0, max(fraction_pct) * 1.2))
    ax1.set_xticks(nodes)
    ax1.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
    if title:
        ax1.set_title(title)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles + [line], labels + [line.get_label()], loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

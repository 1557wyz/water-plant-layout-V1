"""Compare the layout optimizer with and without RL-guided mutation.

The main algorithm remains unchanged. This script instantiates AdvancedGA twice
for each seed:
1. RL-enabled: the original optimizer behavior.
2. RL-disabled: Q-learning action selection/update is disabled and the RL-guided
   mutation branch falls back to the optimizer's ordinary random mutation.

Outputs include run-level metrics, convergence histories, and publication-style
comparison figures.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import random
import time
from datetime import datetime
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def import_main_symbols():
    with contextlib.redirect_stdout(io.StringIO()):
        from water_plant_layout_v3 import AdvancedGA, create_facilities, _layout_metrics
    return AdvancedGA, create_facilities, _layout_metrics


AdvancedGA, create_facilities, layout_metrics = import_main_symbols()


METHOD_LABELS = {
    "rl_enabled": "With RL guidance",
    "rl_disabled": "Without RL guidance",
}

METHOD_COLORS = {
    "rl_enabled": "#1565C0",
    "rl_disabled": "#8C6D62",
}


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Microsoft YaHei", "SimHei"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
        "font.size": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def disable_rl(ga):
    def select_action(self, state):
        return "baseline_mutation"

    def guided_mutate(self, chrom, action):
        return self.mutate(chrom)

    def update(self, state, action, reward, next_state):
        return None

    ga.rl_epsilon = 0.0
    ga.rl_q_table = {}
    ga._rl_select_action = MethodType(select_action, ga)
    ga._rl_guided_mutate = MethodType(guided_mutate, ga)
    ga._rl_update = MethodType(update, ga)
    return ga


def run_once(method: str, seed: int, population: int, generations: int, mutation_rate: float, log_dir: Path):
    random.seed(seed)
    np.random.seed(seed)

    facilities = create_facilities()
    ga = AdvancedGA(
        facilities,
        population_size=population,
        generations=generations,
        mutation_rate=mutation_rate,
        random_seed=seed,
        enable_cache=True,
    )
    if method == "rl_disabled":
        disable_rl(ga)

    log_buffer = io.StringIO()
    start_time = time.perf_counter()
    with contextlib.redirect_stdout(log_buffer):
        layout, bw, bh = ga.optimize()
    runtime_s = time.perf_counter() - start_time

    log_path = log_dir / f"{method}_seed{seed}.log"
    log_path.write_text(log_buffer.getvalue(), encoding="utf-8")

    metrics = layout_metrics(layout, bw, bh, ga.fitness_history)
    pipeline_summary = metrics["pipeline_summary"]
    adjacency_rows = metrics["adjacency_rows"]
    safety_rows = metrics["safety_rows"]

    adjacency_rate = (
        sum(1 for row in adjacency_rows if row["satisfied"]) / len(adjacency_rows)
        if adjacency_rows else np.nan
    )
    safety_violation_count = sum(1 for row in safety_rows if not row["satisfied"])

    result = {
        "method": method,
        "method_label": METHOD_LABELS[method],
        "seed": seed,
        "population": population,
        "generations": generations,
        "runtime_s": runtime_s,
        "best_fitness": min(ga.fitness_history) if ga.fitness_history else np.nan,
        "final_fitness": ga.fitness_history[-1] if ga.fitness_history else np.nan,
        "layout_width_m": bw,
        "layout_height_m": bh,
        "layout_area_m2": bw * bh,
        "utilization_pct": metrics["utilization"],
        "pipeline_length_m": pipeline_summary["routed_length_m"],
        "weighted_pipeline_length": pipeline_summary["weighted_length_m"],
        "adjacency_rate_pct": adjacency_rate * 100,
        "safety_violation_count": safety_violation_count,
        "rl_state_count": len(ga.rl_q_table),
        "log_path": str(log_path),
    }

    history_rows = [
        {"method": method, "method_label": METHOD_LABELS[method], "seed": seed, "generation": i, "fitness": v}
        for i, v in enumerate(ga.fitness_history)
    ]
    return result, history_rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_history(df_history: pd.DataFrame) -> pd.DataFrame:
    return (
        df_history.groupby(["method", "method_label", "generation"], as_index=False)
        .agg(
            median_fitness=("fitness", "median"),
            q25=("fitness", lambda x: np.percentile(x, 25)),
            q75=("fitness", lambda x: np.percentile(x, 75)),
            mean_fitness=("fitness", "mean"),
        )
    )


def plot_comparison(df_results: pd.DataFrame, df_history: pd.DataFrame, output_base: Path):
    apply_style()
    hist = aggregate_history(df_history)

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 125 / 25.4), dpi=180)
    ax_conv, ax_fit, ax_area, ax_pipe = axes.flat

    for method, group in hist.groupby("method"):
        color = METHOD_COLORS[method]
        x = group["generation"].to_numpy()
        median = group["median_fitness"].to_numpy()
        q25 = group["q25"].to_numpy()
        q75 = group["q75"].to_numpy()
        ax_conv.plot(x, median, color=color, lw=1.3, label=METHOD_LABELS[method])
        ax_conv.fill_between(x, q25, q75, color=color, alpha=0.15, lw=0)
    ax_conv.set_title("Convergence profile", loc="left", fontweight="bold", pad=4)
    ax_conv.set_xlabel("Generation")
    ax_conv.set_ylabel("Fitness (lower is better)")
    ax_conv.grid(axis="y", color="#EAEAEA", lw=0.35)
    ax_conv.legend(loc="upper right")

    metric_specs = [
        (ax_fit, "best_fitness", "Best fitness", ""),
        (ax_area, "layout_area_m2", "Layout area", "m2"),
        (ax_pipe, "weighted_pipeline_length", "Weighted pipeline length", ""),
    ]
    for ax, col, title, unit in metric_specs:
        summary = df_results.groupby("method", as_index=False)[col].agg(["mean", "std"]).reset_index()
        order = ["rl_enabled", "rl_disabled"]
        means = [summary.loc[summary["method"] == m, "mean"].iloc[0] for m in order]
        stds = [summary.loc[summary["method"] == m, "std"].fillna(0).iloc[0] for m in order]
        colors = [METHOD_COLORS[m] for m in order]
        labels = [METHOD_LABELS[m].replace(" guidance", "") for m in order]
        ax.bar(np.arange(len(order)), means, yerr=stds, color=colors, edgecolor="#303030", lw=0.35, capsize=2)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.set_title(title, loc="left", fontweight="bold", pad=4)
        ax.set_ylabel(unit)
        ax.grid(axis="y", color="#EAEAEA", lw=0.35)

    fig.suptitle("Effect of reinforcement-learning guidance on layout optimization", fontweight="bold", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{output_base}.svg", bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_table(df_results: pd.DataFrame, output_base: Path):
    apply_style()
    quality_metrics = [
        ("best_fitness", "Best fitness", "lower"),
        ("layout_area_m2", "Area", "lower"),
        ("pipeline_length_m", "Pipe length", "lower"),
        ("weighted_pipeline_length", "Weighted pipe", "lower"),
        ("adjacency_rate_pct", "Adjacency", "higher"),
    ]
    grouped = df_results.groupby("method")

    def improvement(col: str, direction: str) -> float:
        rl = grouped[col].mean()["rl_enabled"]
        base = grouped[col].mean()["rl_disabled"]
        if direction == "lower":
            return (base - rl) / max(abs(base), 1e-9) * 100
        return (rl - base) / max(abs(base), 1e-9) * 100

    quality_rows = [
        {"metric": label, "improvement_pct": improvement(col, direction)}
        for col, label, direction in quality_metrics
    ]
    runtime_improvement = improvement("runtime_s", "lower")

    df = pd.DataFrame(quality_rows)
    fig, (ax_quality, ax_runtime) = plt.subplots(
        1,
        2,
        figsize=(145 / 25.4, 74 / 25.4),
        dpi=180,
        gridspec_kw={"width_ratios": [4.4, 1.6], "wspace": 0.52},
    )

    colors = ["#1565C0" if v >= 0 else "#B64342" for v in df["improvement_pct"]]
    ax_quality.barh(df["metric"], df["improvement_pct"], color=colors, edgecolor="#303030", lw=0.35)
    ax_quality.axvline(0, color="#303030", lw=0.6)
    ax_quality.set_xlabel("Relative improvement (%)")
    ax_quality.set_title("Optimization-quality indicators", loc="left", fontweight="bold", pad=4)
    ax_quality.grid(axis="x", color="#EAEAEA", lw=0.35)
    min_x = min(-5, df["improvement_pct"].min() - 3)
    max_x = max(5, df["improvement_pct"].max() + 3)
    ax_quality.set_xlim(min_x, max_x)
    for i, value in enumerate(df["improvement_pct"]):
        ha = "left" if value >= 0 else "right"
        offset = 0.35 if value >= 0 else -0.35
        ax_quality.text(value + offset, i, f"{value:+.1f}%", va="center", ha=ha, fontsize=6)

    runtime_color = "#1565C0" if runtime_improvement >= 0 else "#B64342"
    ax_runtime.barh(["Runtime"], [runtime_improvement], color=runtime_color, edgecolor="#303030", lw=0.35)
    ax_runtime.axvline(0, color="#303030", lw=0.6)
    ax_runtime.set_yticks([0])
    ax_runtime.set_yticklabels([""])
    ax_runtime.set_xlabel("Relative improvement (%)")
    ax_runtime.set_title("Computation", loc="left", fontweight="bold", pad=4)
    ax_runtime.grid(axis="x", color="#EAEAEA", lw=0.35)
    span = max(8, abs(runtime_improvement) + 8)
    ax_runtime.set_xlim(-span, span)
    ax_runtime.text(runtime_improvement / 2, 0, f"Runtime\n{runtime_improvement:+.1f}%", va="center", ha="center", fontsize=6, color="white")
    fig.suptitle("RL-enabled advantage summary", fontweight="bold", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{output_base}.svg", bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def paired_improvements(df_results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("best_fitness", "best_fitness_improvement_pct", "lower"),
        ("layout_area_m2", "area_improvement_pct", "lower"),
        ("pipeline_length_m", "pipeline_length_improvement_pct", "lower"),
        ("weighted_pipeline_length", "weighted_pipeline_improvement_pct", "lower"),
        ("adjacency_rate_pct", "adjacency_improvement_pct", "higher"),
        ("runtime_s", "runtime_improvement_pct", "lower"),
    ]
    pivot = df_results.pivot(index="seed", columns="method")
    rows = []
    for seed in pivot.index:
        row = {"seed": seed}
        for source_col, out_col, direction in metrics:
            rl = pivot.loc[seed, (source_col, "rl_enabled")]
            base = pivot.loc[seed, (source_col, "rl_disabled")]
            if direction == "lower":
                value = (base - rl) / max(abs(base), 1e-9) * 100
            else:
                value = (rl - base) / max(abs(base), 1e-9) * 100
            row[out_col] = value
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare AdvancedGA with and without RL guidance.")
    parser.add_argument("--runs", type=int, default=3, help="Number of paired seeds.")
    parser.add_argument("--population", type=int, default=48, help="Population size passed to AdvancedGA.")
    parser.add_argument("--generations", type=int, default=35, help="Optimization generations.")
    parser.add_argument("--mutation-rate", type=float, default=0.35, help="Initial mutation rate.")
    parser.add_argument("--seed-start", type=int, default=20260527, help="First paired random seed.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or Path("outputs") / f"rl_comparison_{timestamp}")
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed_start + i for i in range(args.runs)]
    results = []
    histories = []
    for seed in seeds:
        for method in ["rl_enabled", "rl_disabled"]:
            print(f"Running {METHOD_LABELS[method]} seed={seed} ...")
            result, history_rows = run_once(
                method,
                seed,
                population=args.population,
                generations=args.generations,
                mutation_rate=args.mutation_rate,
                log_dir=log_dir,
            )
            results.append(result)
            histories.extend(history_rows)

    df_results = pd.DataFrame(results)
    df_history = pd.DataFrame(histories)
    write_csv(output_dir / "rl_comparison_results.csv", results)
    write_csv(output_dir / "rl_comparison_history.csv", histories)

    plot_comparison(df_results, df_history, output_dir / "rl_comparison_overview")
    plot_metric_table(df_results, output_dir / "rl_comparison_advantage")
    df_paired = paired_improvements(df_results)
    df_paired.to_csv(output_dir / "rl_paired_improvements.csv", index=False, encoding="utf-8-sig")

    summary = (
        df_results.groupby(["method", "method_label"], as_index=False)
        .agg(
            runs=("seed", "count"),
            mean_best_fitness=("best_fitness", "mean"),
            std_best_fitness=("best_fitness", "std"),
            mean_area_m2=("layout_area_m2", "mean"),
            mean_pipeline_m=("pipeline_length_m", "mean"),
            mean_weighted_pipeline=("weighted_pipeline_length", "mean"),
            mean_adjacency_pct=("adjacency_rate_pct", "mean"),
            mean_runtime_s=("runtime_s", "mean"),
            mean_rl_states=("rl_state_count", "mean"),
        )
    )
    summary.to_csv(output_dir / "rl_comparison_summary.csv", index=False, encoding="utf-8-sig")

    print("\nComparison completed.")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Overview figure: {(output_dir / 'rl_comparison_overview.png').resolve()}")
    print(f"Advantage figure: {(output_dir / 'rl_comparison_advantage.png').resolve()}")


if __name__ == "__main__":
    main()

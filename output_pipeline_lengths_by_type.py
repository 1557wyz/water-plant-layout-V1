"""Calculate pipeline lengths by system for an optimized layout."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random

import numpy as np
import pandas as pd

PIPELINE_SYSTEM = None


def load_layout_module():
    """Load the main layout module while suppressing import-time device probes."""
    with contextlib.redirect_stdout(io.StringIO()):
        from water_plant_layout_v3 import (
            AdvancedGA,
            FACILITIES_DATA,
            Facility,
            PIPELINE_SYSTEM,
            calculate_pipeline_network,
        )
    return AdvancedGA, FACILITIES_DATA, Facility, PIPELINE_SYSTEM, calculate_pipeline_network


def create_facilities(facility_cls, facilities_data):
    return [facility_cls(*args) for args in facilities_data]


def pipeline_lengths_by_system(placed, calculate_pipeline_network_fn) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    segments, summary = calculate_pipeline_network_fn(placed)
    if segments.empty:
        grouped = pd.DataFrame(columns=["管线系统", "管线类型", "线段数量", "路由长度(m)", "加权长度"])
    else:
        grouped = (
            segments.groupby(["system", "label"], as_index=False)
            .agg(
                线段数量=("from", "count"),
                **{"路由长度(m)": ("routed_length_m", "sum"), "加权长度": ("weighted_length_m", "sum")},
            )
            .rename(columns={"system": "管线系统", "label": "管线类型"})
        )
        grouped["路由长度(m)"] = grouped["路由长度(m)"].round(3)
        grouped["加权长度"] = grouped["加权长度"].round(3)

    segment_cols = {
        "system": "管线系统",
        "label": "管线类型",
        "from": "起点",
        "to": "终点",
        "center_manhattan_m": "中心曼哈顿距离(m)",
        "edge_clearance_m": "边界净距(m)",
        "routed_length_m": "路由长度(m)",
        "weighted_length_m": "加权长度",
        "detour_factor": "绕行系数",
        "terminal_allowance_m": "接入余量(m)",
    }
    segment_df = segments.rename(columns=segment_cols) if not segments.empty else pd.DataFrame(columns=segment_cols.values())
    return grouped, segment_df, summary


def quick_sample_layout(ga: AdvancedGA, samples: int) -> tuple[list[Facility], float, float, float]:
    best_fitness = float("inf")
    best_chrom = None
    for _ in range(samples):
        chrom = ga.create_chromosome()
        fitness = ga.calculate_fitness(chrom)
        if fitness < best_fitness:
            best_fitness = fitness
            best_chrom = chrom

    base_size = np.sqrt(ga.total_area) * 1.8
    placed, width, height = ga.decode_chromosome(best_chrom, base_size)
    return placed, width, height, best_fitness


def run_pipeline_analysis(args) -> dict:
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    global PIPELINE_SYSTEM
    AdvancedGA, facilities_data, facility_cls, PIPELINE_SYSTEM, calculate_pipeline_network_fn = load_layout_module()
    facilities = create_facilities(facility_cls, facilities_data)
    ga = AdvancedGA(
        facilities,
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        random_seed=args.seed,
    )

    if args.fast:
        placed, width, height, fitness = quick_sample_layout(ga, args.samples)
    else:
        placed, width, height = ga.optimize()
        fitness = ga.best_fitness

    df, segment_df, pipeline_summary = pipeline_lengths_by_system(placed, calculate_pipeline_network_fn)
    summary = {
        "layout_width_m": round(width, 3),
        "layout_height_m": round(height, 3),
        "layout_area_m2": round(width * height, 3),
        "facility_area_m2": round(sum(f.area for f in placed), 3),
        "space_utilization_pct": round(sum(f.area for f in placed) / (width * height) * 100, 3),
        "fitness": fitness,
        "total_pipeline_length_m": round(pipeline_summary["routed_length_m"], 3),
        "weighted_pipeline_length": round(pipeline_summary["weighted_length_m"], 3),
        "center_manhattan_length_m": round(pipeline_summary["center_manhattan_m"], 3),
    }
    return {"pipeline_lengths": df, "pipeline_segments": segment_df, "summary": summary}


def export_results(result: dict, output_dir: str, basename: str):
    os.makedirs(output_dir, exist_ok=True)
    df = result["pipeline_lengths"]
    segment_df = result["pipeline_segments"]
    summary = result["summary"]

    xlsx_path = os.path.join(output_dir, f"{basename}.xlsx")
    csv_path = os.path.join(output_dir, f"{basename}.csv")
    json_path = os.path.join(output_dir, f"{basename}_summary.json")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="管线长度")
        segment_df.to_excel(writer, index=False, sheet_name="管段明细")
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="布局摘要")
        for ws in writer.sheets.values():
            ws.freeze_panes = "A2"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    segment_csv_path = os.path.join(output_dir, f"{basename}_segments.csv")
    segment_df.to_csv(segment_csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return [xlsx_path, csv_path, segment_csv_path, json_path]


def main():
    parser = argparse.ArgumentParser(description="按管线系统统计优化布局中的管线长度")
    parser.add_argument("--output-dir", default="exports", help="输出目录")
    parser.add_argument("--basename", default="pipeline_lengths", help="输出文件名前缀")
    parser.add_argument("--population", type=int, default=80, help="GA种群规模")
    parser.add_argument("--generations", type=int, default=50, help="GA优化代数")
    parser.add_argument("--mutation-rate", type=float, default=0.35, help="初始变异率")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--fast", action="store_true", help="快速采样模式，不运行完整GA优化")
    parser.add_argument("--samples", type=int, default=80, help="快速采样个体数")
    args = parser.parse_args()

    result = run_pipeline_analysis(args)
    written = export_results(result, args.output_dir, args.basename)

    print("管线长度统计完成:")
    for path in written:
        print(f"  {path}")
    print(f"总管线长度: {result['summary']['total_pipeline_length_m']:.3f} m")


if __name__ == "__main__":
    main()

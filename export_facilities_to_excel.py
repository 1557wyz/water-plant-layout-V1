"""Export facility metadata from the main layout model.

The script intentionally imports facility definitions from water_plant_layout_v3.py
instead of keeping a duplicated FACILITIES_DATA table.
"""

import argparse
import contextlib
import io
import os

import pandas as pd


def load_facility_source():
    """Load canonical facility data while suppressing import-time device probes."""
    with contextlib.redirect_stdout(io.StringIO()):
        from water_plant_layout_v3 import CATEGORY_NAMES, FACILITIES_DATA
    return CATEGORY_NAMES, FACILITIES_DATA


def build_facility_dataframe() -> pd.DataFrame:
    category_names, facilities_data = load_facility_source()
    rows = []
    safety_labels = {1: "普通", 2: "中危", 3: "高危"}
    for code, name_cn, length, width, category, safety_level in facilities_data:
        rows.append({
            "代码": code,
            "中文名": name_cn,
            "长度(m)": length,
            "宽度(m)": width,
            "面积(m²)": length * width,
            "功能分区": category,
            "功能分区名称": category_names.get(category, category),
            "安全等级": safety_level,
            "安全等级名称": safety_labels.get(safety_level, str(safety_level)),
        })
    return pd.DataFrame(rows)


def export_facilities(output_dir: str, basename: str, formats: list[str]) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    df = build_facility_dataframe()
    written = []

    if "xlsx" in formats:
        path = os.path.join(output_dir, f"{basename}.xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="设施参数")
            ws = writer.sheets["设施参数"]
            widths = [8, 14, 10, 10, 10, 14, 14, 10, 12]
            for idx, width in enumerate(widths, 1):
                ws.column_dimensions[chr(64 + idx)].width = width
            ws.freeze_panes = "A2"
        written.append(path)

    if "csv" in formats:
        path = os.path.join(output_dir, f"{basename}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)

    return written


def main():
    parser = argparse.ArgumentParser(description="导出供水厂设施参数表")
    parser.add_argument("--output-dir", default="exports", help="输出目录")
    parser.add_argument("--basename", default="facilities", help="输出文件名前缀")
    parser.add_argument("--formats", default="xlsx,csv", help="逗号分隔格式: xlsx,csv")
    args = parser.parse_args()

    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    unsupported = sorted(set(formats) - {"xlsx", "csv"})
    if unsupported:
        raise ValueError(f"不支持的导出格式: {', '.join(unsupported)}")

    written = export_facilities(args.output_dir, args.basename, formats)
    print("设施参数已导出:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()

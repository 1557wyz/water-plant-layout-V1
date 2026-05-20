# Water Plant Facility Layout Optimization

供水厂设施布局优化项目。项目以 `water_plant_layout_v3.py` 为核心数据与算法入口，融合剩余矩形/Bottom-Left 排样思想与遗传算法，并叠加工艺邻接、安全距离、功能分区、管线路由和竖向高程约束，用于生成可审查、可视化、可导出的供水厂总平面布置方案。

## 核心特性

- 保留“剩余矩形排样 + 遗传算法”核心框架。
- 多策略遗传优化：岛屿模型、差分进化、模拟退火、强化学习引导、Pareto 前沿。
- 功能分区联合布置：取水、预处理、常规处理、深度处理、加药、送配水、动力、污泥、行政、辅助、生活区。
- 约束检查：设施不重叠、安全距离、工艺流程邻接、宿舍联合布置、门卫入口位置。
- 管线计算：基于设施边界净距、接入余量、转弯余量、管线系统绕行系数和权重的路由长度估算。
- 高程计算：基于主文件高程数据和分段水力参数，可读取实际布局坐标计算水头损失。
- 论文级输出：SVG/PDF/TIFF/PNG 图件、CSV/Excel 表格、图注与 QA 说明。

## 文件结构

```text
.
├── water_plant_layout_v3.py          # 主算法、主数据、布局优化与论文级图件输出
├── elevation_analysis.py             # 竖向高程约束分析，依赖主文件高程数据
├── export_facilities_to_excel.py     # 设施参数导出，依赖主文件设施数据
├── output_pipeline_lengths_by_type.py# 管线长度统计，依赖主文件管线模型
├── requirements.txt                  # Python 依赖
└── README.md
```

所有分析脚本均以 `water_plant_layout_v3.py` 为唯一核心数据源，避免设施、高程、管线定义在多个脚本中漂移。

## 环境准备

建议使用 Python 3.10+。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU 加速为可选能力。若环境中安装了 CuPy 和可用 CUDA，主程序会自动尝试启用；否则自动回退到 CPU/Numba 路径。

## 运行主优化

完整优化示例：

```powershell
python .\water_plant_layout_v3.py --population 200 --generations 150 --seed 42 --output-prefix outputs\full_run\layout\academic_analysis
```

常用参数：

- `--population`：种群规模，默认 `200`
- `--generations`：优化代数，默认 `150`
- `--mutation-rate`：初始变异率，默认 `0.35`
- `--seed`：随机种子，便于复现实验
- `--output-prefix`：论文图件和结果表输出前缀
- `--skip-plots`：只运行优化，不导出图件

主程序会生成：

- 布局图：`svg/pdf/tiff/png`
- 复合论文图
- 设施坐标表
- 分区统计表
- 邻接报告
- 安全距离报告
- 管线系统汇总和管段明细
- 图注与 QA 说明

## 导出设施参数

```powershell
python .\export_facilities_to_excel.py --output-dir exports\data --basename facilities --formats xlsx,csv
```

输出：

- `facilities.xlsx`
- `facilities.csv`

## 管线长度统计

快速采样模式：

```powershell
python .\output_pipeline_lengths_by_type.py --fast --samples 80 --output-dir exports\pipeline --basename pipeline_lengths
```

完整 GA 优化模式：

```powershell
python .\output_pipeline_lengths_by_type.py --population 80 --generations 50 --seed 42 --output-dir exports\pipeline --basename pipeline_lengths
```

输出：

- 管线系统汇总
- 管段明细
- 布局摘要 JSON
- Excel 工作簿

## 高程分析

使用典型距离假设：

```powershell
python .\elevation_analysis.py --output-dir exports\elevation --prefix elevation
```

使用主程序生成的实际布局坐标计算管线长度和水头损失：

```powershell
python .\elevation_analysis.py --layout-csv outputs\full_run\layout\academic_analysis_publication\facility_layout.csv --output-dir exports\elevation --prefix elevation
```

输出：

- `elevation_report.xlsx`
- 重力流可行性 CSV
- 构筑物水深/高程 CSV
- 主水处理、污泥处理、可行性、参数表和合并总图的 `svg/pdf/tiff/png`

## 推荐端到端流程

```powershell
$runDir = "outputs\full_run"
python .\water_plant_layout_v3.py --population 200 --generations 150 --seed 42 --output-prefix "$runDir\layout\academic_analysis"
python .\export_facilities_to_excel.py --output-dir "$runDir\data" --basename facilities --formats xlsx,csv
python .\elevation_analysis.py --layout-csv "$runDir\layout\academic_analysis_publication\facility_layout.csv" --output-dir "$runDir\elevation" --prefix elevation
```

管线汇总和管段明细已由主程序输出在 `layout\academic_analysis_publication` 中；如需单独重新计算，可运行 `output_pipeline_lengths_by_type.py`。

## 输出与版本控制

`outputs/`、`exports/`、图件和报表属于生成结果，默认通过 `.gitignore` 排除。仓库建议只提交源码、配置和文档；需要归档某次运行结果时，可单独打包输出目录。

## 注意事项

- 当前模型面向方案优化和工程估算，不替代施工图阶段的管线综合、地形复核和专业水力计算。
- 管线长度采用正交管廊/管沟估算模型，包含边界净距、接入余量、转弯余量和系统绕行系数。
- 高程分析可基于实际布局坐标运行，建议在最终方案生成后使用 `--layout-csv` 复核。

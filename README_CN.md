# 供水厂设施布局优化项目说明

本项目面向供水厂总平面设施布局优化问题，围绕有限建设用地条件下构筑物布置、工艺联系、安全间距、管线组织、高程衔接和结果可视化等内容，建立了一套可复现的智能优化与分析流程。

项目主文件为 `water_plant_layout_v3.py`。其他分析脚本均依赖主文件中的设施数据、工艺连接、管线参数和高程参数，避免多个脚本之间出现数据定义不一致的问题。

## 研究目标

本项目主要解决以下问题：

- 在有限厂区范围内生成可行的构筑物布局；
- 提高土地利用率并降低布局面积；
- 优化工艺构筑物之间的邻接关系；
- 校核安全间距和功能分区合理性；
- 估算不同系统管线长度及加权代价；
- 分析主要处理流程的高程与水头损失；
- 对比强化学习机制启用与不启用时的优化效果。

本项目适用于算法研究、方案早期比选和学术可视化展示，不替代施工图阶段的详细管线综合、地形复核和专业水力计算。

## 核心算法

项目采用 RRP-Enhanced GA 思路，即“剩余矩形布局算法 + 增强遗传算法”的融合框架：

1. 通过剩余矩形布局思想生成较高质量的可行初始排布；
2. 通过遗传算法进行全局搜索，优化设施顺序、朝向和布局结构；
3. 引入强化学习机制，在演化过程中自适应选择变异策略；
4. 适应度函数综合考虑布局面积、管线代价、邻接关系、安全距离、分区紧凑性、入口逻辑和高程可行性；
5. 后处理脚本输出布局图、管线图、高程图、统计表和对比实验结果。

其中，“剩余矩形算法与遗传算法融合”仍是核心算法，强化学习机制是二次增强模块，不替代核心求解框架。

## 文件结构

```text
.
├── water_plant_layout_v3.py           # 主数据、主算法、评价指标和布局图输出
├── elevation_analysis.py              # 高程与水头损失分析
├── draw_pipeline_layout.py            # 管路绘制脚本
├── compare_rl_mechanism.py            # 强化学习机制对比实验脚本
├── output_pipeline_lengths_by_type.py # 按管线系统统计长度
├── export_facilities_to_excel.py      # 设施参数导出
├── requirements.txt                   # Python 依赖
├── README.md                          # 英文说明
└── README_CN.md                       # 中文说明
```

`outputs/` 和 `exports/` 为生成结果目录，默认不纳入 Git 版本控制，因为其中通常包含较大的 TIFF、PDF、PNG 和 CSV 结果文件。

## 环境配置

建议使用 Python 3.10 或以上版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

依赖包包括：

- `numpy`
- `pandas`
- `matplotlib`
- `openpyxl`
- `numba`

如果本地安装了 CuPy 和可用 CUDA，主程序会尝试使用 GPU 加速；否则自动回退到 CPU/Numba 路径。

## 运行主优化程序

```powershell
python .\water_plant_layout_v3.py `
  --population 200 `
  --generations 150 `
  --mutation-rate 0.35 `
  --seed 42 `
  --output-prefix outputs\full_run\layout\academic_analysis
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--population` | 种群规模，默认 `200` |
| `--generations` | 优化代数，默认 `150` |
| `--mutation-rate` | 初始变异率，默认 `0.35` |
| `--seed` | 随机种子，用于复现实验 |
| `--no-cache` | 关闭适应度缓存 |
| `--skip-plots` | 只运行优化，不导出图件 |
| `--output-prefix` | 图件和结果表输出前缀 |

主程序会输出：

- 优化布局图：`SVG/PDF/TIFF/PNG`；
- 设施坐标表；
- 邻接关系报告；
- 安全间距报告；
- 管线分系统汇总和管段明细；
- 功能分区统计；
- 图注和 QA 说明。

## 管线长度计算

管线长度基于优化后的构筑物坐标计算。模型没有简单采用中心点欧氏距离，而是采用更接近厂区正交管廊布置的边界净距估算方法。

单段管线计算逻辑为：

```text
center_manhattan = |cx2 - cx1| + |cy2 - cy1|
edge_clearance = gap_x + gap_y
如果 x 和 y 两个方向都存在净距，则 elbow_allowance = 1.5 m，否则为 0
base_external = edge_clearance + terminal_allowance + elbow_allowance
routed_length = max(terminal_allowance, base_external) × detour_factor
weighted_length = routed_length × system_weight
```

各系统参数如下：

| 系统 | 接管余量 | 绕行系数 | 权重 |
|---|---:|---:|---:|
| 主水处理管线 | 4.0 m | 1.08 | 3.0 |
| 深度处理管线 | 4.0 m | 1.10 | 2.4 |
| 污泥管线 | 5.0 m | 1.18 | 2.2 |
| 反冲洗管线 | 3.0 m | 1.15 | 1.5 |
| 加药管线 | 2.5 m | 1.25 | 2.0 |
| 电力线路 | 2.0 m | 1.12 | 1.0 |

单独运行管线统计：

```powershell
python .\output_pipeline_lengths_by_type.py `
  --population 80 `
  --generations 50 `
  --seed 42 `
  --output-dir exports\pipeline `
  --basename pipeline_lengths
```

快速采样模式：

```powershell
python .\output_pipeline_lengths_by_type.py `
  --fast `
  --samples 80 `
  --output-dir exports\pipeline `
  --basename pipeline_lengths
```

## 管路图绘制

使用主优化程序导出的 `facility_layout.csv`：

```powershell
python .\draw_pipeline_layout.py `
  --layout-csv outputs\full_run\layout\academic_analysis_publication\facility_layout.csv `
  --output-dir outputs\full_run\layout\academic_analysis_publication\pipeline_drawings `
  --basename pipeline_routing_refined
```

该脚本会生成：

- 管线总图；
- 分系统管线图；
- 管段明细表；
- 管线系统汇总表。

绘图风格面向学术展示：黑色实心节点、细线管路、空心小箭头、底部图例和尽量避免遮挡的长度标注。

## 高程与水头分析

使用典型距离假设：

```powershell
python .\elevation_analysis.py `
  --output-dir exports\elevation `
  --prefix elevation
```

使用实际优化布局坐标：

```powershell
python .\elevation_analysis.py `
  --layout-csv outputs\full_run\layout\academic_analysis_publication\facility_layout.csv `
  --output-dir exports\elevation `
  --prefix elevation
```

高程模块主要分析主水处理、深度处理和污泥处理流程，校核可用水头、沿程损失、局部损失、安全余量和重力流可行性。非处理构筑物不纳入高程模块计算。

## 强化学习机制对比实验

`compare_rl_mechanism.py` 用于成对比较：

- 启用强化学习引导变异的原始优化算法；
- 关闭强化学习选择和 Q 表更新后的基线遗传算法。

运行示例：

```powershell
python .\compare_rl_mechanism.py `
  --runs 3 `
  --population 48 `
  --generations 35 `
  --seed-start 20260527 `
  --output-dir outputs\rl_comparison
```

输出包括：

- `rl_comparison_results.csv`
- `rl_comparison_history.csv`
- `rl_comparison_summary.csv`
- `rl_paired_improvements.csv`
- `SVG/PDF/TIFF/PNG` 对比图；
- 每个随机种子的运行日志。

最近一次成对实验表明，强化学习机制能够改善平均最优适应度和邻接关系满足率，但会增加运行时间，并且对部分管线长度指标存在一定权衡。因此，强化学习机制更适合作为提高综合搜索质量的增强模块，而不是对所有子目标都绝对占优的替代算法。

## 设施数据导出

```powershell
python .\export_facilities_to_excel.py `
  --output-dir exports\data `
  --basename facilities `
  --formats xlsx,csv
```

## 推荐完整运行流程

```powershell
$runDir = "outputs\full_run"

python .\water_plant_layout_v3.py `
  --population 200 `
  --generations 150 `
  --seed 42 `
  --output-prefix "$runDir\layout\academic_analysis"

python .\export_facilities_to_excel.py `
  --output-dir "$runDir\data" `
  --basename facilities `
  --formats xlsx,csv

python .\elevation_analysis.py `
  --layout-csv "$runDir\layout\academic_analysis_publication\facility_layout.csv" `
  --output-dir "$runDir\elevation" `
  --prefix elevation

python .\draw_pipeline_layout.py `
  --layout-csv "$runDir\layout\academic_analysis_publication\facility_layout.csv" `
  --output-dir "$runDir\layout\academic_analysis_publication\pipeline_drawings" `
  --basename pipeline_routing_refined
```

## 局限性

- 当前模型尚未纳入真实风向、地形、地质、道路交通、分期建设和环境敏感点等外部约束；
- 管线长度属于基于矩形边界和正交管廊假设的工程估算；
- 高程分析适合方案阶段的可行性判断和风险识别，详细水力设计仍需单独复核；
- 生成结果目录默认不提交到 Git，如需归档某次运行结果，建议单独打包保存。

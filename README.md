# Water Plant Facility Layout Optimization

This repository contains a research-oriented optimization workflow for water
treatment plant facility layout planning. The project combines a residual
rectangle placement strategy with an enhanced genetic algorithm, reinforcement
learning guided mutation, pipeline routing estimation, vertical elevation and
headloss analysis, and publication-quality visualization outputs.

The main implementation is `water_plant_layout_v3.py`. All auxiliary analysis
scripts read facility definitions, process links, pipeline parameters, and
elevation data from the main file so that the project has one consistent data
source.

## Research Objective

The project addresses the problem of arranging water plant facilities under
limited land availability while considering:

- facility footprint and site utilization;
- process adjacency and functional zoning;
- safety spacing between structures;
- pipeline routing length and weighted pipeline cost;
- vertical elevation compatibility and hydraulic headloss;
- reproducible comparison between RL-guided and non-RL optimization.

It is intended for algorithm research, early-stage engineering comparison, and
layout visualization. It does not replace detailed construction drawing design,
terrain modeling, or professional hydraulic network design.

## Core Method

The optimizer uses an RRP-Enhanced GA framework:

1. Residual rectangle placement is used to generate feasible rectangular
   facility layouts and reduce invalid packing.
2. The genetic algorithm searches globally over facility ordering, orientation,
   and placement structure.
3. Reinforcement learning guides mutation action selection during evolution.
4. A multi-objective fitness model evaluates land area, pipe routing cost,
   process adjacency, safety spacing, zoning compactness, entrance logic, and
   elevation feasibility.
5. Post-processing scripts generate layout figures, pipeline diagrams,
   elevation reports, and comparison experiments.

The core algorithm remains the residual rectangle and genetic algorithm fusion.
Reinforcement learning is implemented as a secondary search enhancement rather
than a replacement of the main optimization mechanism.

## Repository Structure

```text
.
|-- water_plant_layout_v3.py           # Main data, optimizer, metrics, and layout figures
|-- elevation_analysis.py              # Elevation and hydraulic headloss analysis
|-- draw_pipeline_layout.py            # Clean pipeline-only route diagrams
|-- compare_rl_mechanism.py            # Paired RL vs non-RL comparison experiment
|-- output_pipeline_lengths_by_type.py # Pipeline length summary by system
|-- export_facilities_to_excel.py      # Facility data export
|-- requirements.txt                   # Python dependencies
|-- README.md                          # English documentation
`-- README_CN.md                       # Chinese documentation
```

Generated results are written to `outputs/` or `exports/`. These directories are
ignored by Git because they may contain large TIFF/PDF/PNG files and run-specific
CSV reports.

## Requirements

Python 3.10 or later is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Required packages:

- `numpy`
- `pandas`
- `matplotlib`
- `openpyxl`
- `numba`

If CuPy and CUDA are available, the main program can attempt GPU acceleration.
Otherwise it automatically falls back to CPU/Numba execution.

## Run the Main Optimizer

```powershell
python .\water_plant_layout_v3.py `
  --population 200 `
  --generations 150 `
  --mutation-rate 0.35 `
  --seed 42 `
  --output-prefix outputs\full_run\layout\academic_analysis
```

Common options:

| Option | Description |
|---|---|
| `--population` | GA population size. Default: `200` |
| `--generations` | Number of optimization generations. Default: `150` |
| `--mutation-rate` | Initial mutation rate. Default: `0.35` |
| `--seed` | Random seed for reproducibility |
| `--no-cache` | Disable fitness cache |
| `--skip-plots` | Run optimization without exporting figures |
| `--output-prefix` | Prefix for generated figures and CSV reports |

Main outputs include:

- optimized layout figures in `SVG`, `PDF`, `TIFF`, and `PNG`;
- facility coordinate table;
- adjacency satisfaction report;
- safety spacing report;
- pipeline segment and system summaries;
- zone summary;
- figure caption and quality-assurance note.

## Pipeline Length Analysis

Pipeline lengths are computed from the optimized facility layout. The model uses
facility boundary clearance instead of simple center-to-center Euclidean distance
so that the result better approximates orthogonal pipe gallery routing.

For a pair of facilities:

```text
center_manhattan = |cx2 - cx1| + |cy2 - cy1|
edge_clearance = gap_x + gap_y
elbow_allowance = 1.5 m if both x and y clearances exist, otherwise 0
base_external = edge_clearance + terminal_allowance + elbow_allowance
routed_length = max(terminal_allowance, base_external) x detour_factor
weighted_length = routed_length x system_weight
```

Pipeline system parameters:

| System | Terminal allowance | Detour factor | Weight |
|---|---:|---:|---:|
| Main water | 4.0 m | 1.08 | 3.0 |
| Advanced treatment | 4.0 m | 1.10 | 2.4 |
| Sludge | 5.0 m | 1.18 | 2.2 |
| Backwash | 3.0 m | 1.15 | 1.5 |
| Chemical dosing | 2.5 m | 1.25 | 2.0 |
| Power cable | 2.0 m | 1.12 | 1.0 |

Run an independent pipeline summary:

```powershell
python .\output_pipeline_lengths_by_type.py `
  --population 80 `
  --generations 50 `
  --seed 42 `
  --output-dir exports\pipeline `
  --basename pipeline_lengths
```

Fast sampling mode:

```powershell
python .\output_pipeline_lengths_by_type.py `
  --fast `
  --samples 80 `
  --output-dir exports\pipeline `
  --basename pipeline_lengths
```

## Pipeline Route Visualization

Use the optimized `facility_layout.csv` produced by the main optimizer:

```powershell
python .\draw_pipeline_layout.py `
  --layout-csv outputs\full_run\layout\academic_analysis_publication\facility_layout.csv `
  --output-dir outputs\full_run\layout\academic_analysis_publication\pipeline_drawings `
  --basename pipeline_routing_refined
```

The script creates:

- a clean combined pipeline network figure;
- subsystem pipeline figures;
- pipeline segment CSV;
- pipeline system summary CSV.

The visual style is designed for academic reporting: black solid facility nodes,
thin colored routes, small open arrows, bottom legends, and non-overlapping
labels where possible.

## Elevation and Headloss Analysis

Run with default representative distances:

```powershell
python .\elevation_analysis.py `
  --output-dir exports\elevation `
  --prefix elevation
```

Run with actual optimized layout coordinates:

```powershell
python .\elevation_analysis.py `
  --layout-csv outputs\full_run\layout\academic_analysis_publication\facility_layout.csv `
  --output-dir exports\elevation `
  --prefix elevation
```

The elevation module evaluates main water treatment, advanced treatment, and
sludge treatment routes. It checks available head, pipe headloss, local loss,
safety margin, and gravity-flow feasibility. Non-process buildings are excluded
from the elevation calculation.

## RL Mechanism Comparison

The script `compare_rl_mechanism.py` performs a paired experiment comparing:

- original optimizer with RL-guided mutation;
- baseline optimizer where RL action selection and Q-table updates are disabled.

Example:

```powershell
python .\compare_rl_mechanism.py `
  --runs 3 `
  --population 48 `
  --generations 35 `
  --seed-start 20260527 `
  --output-dir outputs\rl_comparison
```

Outputs include:

- `rl_comparison_results.csv`;
- `rl_comparison_history.csv`;
- `rl_comparison_summary.csv`;
- `rl_paired_improvements.csv`;
- overview and advantage figures in `SVG`, `PDF`, `TIFF`, and `PNG`;
- run logs for each paired seed.

The latest paired test showed that RL guidance improved the average best fitness
and adjacency satisfaction, but increased runtime and slightly worsened some
pipeline length indicators. Therefore RL should be interpreted as a search
quality enhancement with computational cost, not as a uniformly superior method
for every sub-objective.

## Facility Data Export

```powershell
python .\export_facilities_to_excel.py `
  --output-dir exports\data `
  --basename facilities `
  --formats xlsx,csv
```

## Recommended End-to-End Workflow

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

## Notes and Limitations

- The model currently does not include real wind direction, terrain, geological
  conditions, construction phasing, traffic organization, or environmental
  sensitive points.
- Pipeline lengths are engineering approximations based on rectangular facility
  boundaries and orthogonal routing assumptions.
- Elevation analysis is suitable for early-stage feasibility comparison and risk
  identification, but detailed hydraulic design should be performed separately.
- Generated outputs are intentionally ignored by Git. Archive important runs as
  separate result packages if they need to be preserved.

## License

No license file is currently provided. Please add a license before public reuse
or redistribution.

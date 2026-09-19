# Supplementary Materials for the JCCE Manuscript

This directory contains the supplementary experimental outputs and figure files supporting the revised manuscript, *A Computational Framework for Multi-Objective Layout Optimization of Water Treatment Plants Using Residual-Rectangle Packing and Evolutionary Search*.

## Contents

- `formal_experiments/`: 60 completed fixed-budget experimental runs across six configurations and ten random seeds per configuration. Each `.json.gz` file contains one completed run. `summary.json`, `analysis.json`, and `manifest.json` provide the aggregated results, statistical analysis, and experimental configuration, respectively.
- `cp_area_reference/`: Outputs, solver log, configuration, and verification record for the reduced area-only constraint-programming reference model.
- `figures_and_source_data/`: Figure files in PDF, PNG, SVG, and TIFF formats, together with `source_data.json` and `figure_contract.json` used to generate and document the manuscript figures.

Intermediate `.progress.json` records and exploratory runs are intentionally excluded because they are not final experimental results reported in the manuscript.

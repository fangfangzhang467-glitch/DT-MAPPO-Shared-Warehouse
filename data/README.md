# Data Directory

This directory contains only reference inputs and generated experiment artifacts needed by the reproducibility workflow.

## Version-controlled reference files

- `warehouse_layout.csv` — deterministic 9,600-slot warehouse topology with zone, aisle, coordinates, rack level, capacity, and storage category.
- `tenant_profiles.csv` — six tenant profiles corresponding to the demand volatility, turnover, inventory share, and product-profile characteristics used by the study.
- `product_compatibility.csv` — explicit tenant/category compatibility matrix.
- `scenario_manifest.yaml` — robustness and ablation definitions copied from `config.yaml`.
- `reference_events.csv.gz` — deterministic reference realization of tenant inbound/outbound demand for reproducibility checks.
- `checksums.sha256` — SHA-256 hashes of the reference files.

The reference event stream is not claimed to be measured industrial data. The manuscript describes a synthetic industry-like dataset; this repository therefore provides a deterministic generator and a fixed generated realization.

## Runtime files

Training logs, checkpoints, and evaluation outputs are written into this directory at runtime. They are ignored by Git by default so the source release stays lightweight. Final result tables may be included in a DOI archive if required by the journal.

# luad-survival-benchmark

A reproducible, leakage-controlled pipeline for developing and externally validating
transcriptomic survival panels in lung adenocarcinoma (LUAD).

The pipeline selects a gene panel from TCGA-LUAD RNA sequencing data, writes the panel
and the full model specification to a checksummed lock, and then scores three
independent GEO microarray cohorts without any further access to their outcomes. Six
survival learners — including a prior-data fitted survival foundation model — receive
identical inputs, so measured differences are attributable to the learner rather than to
the feature set.

**Status:** research code accompanying a manuscript in preparation. Not a clinical tool.

---

## What the pipeline does

| Stage | Output |
| --- | --- |
| Audit raw inputs and build a patient-level cohort | Deduplicated cohort, exclusion tables, SHA-256 manifest |
| Map identifiers and normalize counts | GENCODE v36 mapping, outcome-blind DESeq2 VST matrix |
| Harmonize external cohorts | Per-cohort standardized matrices, shared gene universe |
| Select and lock a panel | Gene panel plus `model_spec.json` with a file checksum |
| Score external cohorts | Discrimination, paired uncertainty, calibration metrics |
| Evaluate biology post-lock | Matched tumor-normal tests, stage trends, enrichment queries |
| Render outputs | Figures, tables, and formatted documents |

Leakage controls that the code enforces rather than documents:

- Normalization and variance filtering run with `blind = TRUE` and no access to outcomes.
- The candidate gene universe is restricted to genes measurable on every validation
  platform **before** any outcome is examined.
- External scoring recomputes the panel-file checksum and aborts if it has changed.
- Analyses added after the lock are tagged `post_lock_supplementary` in their outputs.

---

## Requirements

- Python 3.11+
- R 4.6.0 with DESeq2 1.52.0 (normalization stages only)
- CUDA-capable GPU (optional; required for the neural and foundation-model estimators)

Install the Python package and its analysis dependencies:

```bash
pip install -e ".[analysis,test]"
```

The pipeline calls `python` and `Rscript` from `PATH`. To use specific interpreters,
set `LUAD_PYTHON` and `LUAD_RSCRIPT`, or pass `-Python` and `-Rscript` to the PowerShell
driver. Cohort roles, seeds and model hyperparameters live in `configs/analysis.toml`.

---

## Quickstart

Run the whole pipeline end to end.

Linux or macOS:

```bash
./scripts/run_pipeline.sh
```

Windows:

```powershell
./scripts/run_pipeline.ps1
```

By default the run resumes from an existing model lock. To repeat feature selection and
replace the lock, pass `-ForceRefit` (PowerShell) or set `FORCE_REFIT=1` (shell).

Other options: `-BootstrapR` / `BOOTSTRAP_R=1` installs the project-local R library,
`-BootstrapReplicates <n>` / `BOOTSTRAP_REPLICATES=<n>` changes the selection-stability
resampling depth, and `-SkipTests` / `SKIP_TESTS=1` omits the test run at the end.

Run the test suite:

```bash
python -m pytest
```

---

## Running individual stages

Each stage is a subcommand of the installed `luad-biomarker` entry point and takes the
same config file:

```bash
luad-biomarker build-cohort --config configs/analysis.toml
luad-biomarker prepare-expression --config configs/analysis.toml
luad-biomarker build-annotation --config configs/analysis.toml
luad-biomarker prepare-geo GSE72094 --config configs/analysis.toml
luad-biomarker run-analysis --config configs/analysis.toml --bootstrap-replicates 200
luad-biomarker score-external --config configs/analysis.toml
luad-biomarker extended-validation --config configs/analysis.toml
luad-biomarker supplementary-models --config configs/analysis.toml
luad-biomarker biological-validation --config configs/analysis.toml
luad-biomarker make-figures --config configs/analysis.toml
luad-biomarker make-reporting --config configs/analysis.toml
```

`run-analysis` performs selection and writes the lock; `score-external` resumes from an
existing lock without repeating selection.

---

## Repository layout

```
configs/          analysis.toml — cohort roles, paths, seeds, model hyperparameters
R/                DESeq2 normalization and model-matrix export
src/luad_biomarker/
  cohort.py       patient-level cohort construction and eligibility
  annotation.py   GENCODE identifier mapping
  expression.py   count filtering and matrix export
  geo.py          external cohort download and harmonization
  modeling.py     panel selection, model fitting, model lock
  validation.py   discrimination, paired bootstrap, calibration
  biology.py      tumor-normal, stage, enrichment and interaction queries
  figures.py      all figure rendering
  reporting.py    table generation and reporting audit
scripts/          portable pipeline drivers
tests/            unit tests
artifacts/        all generated output (see below)
```

Generated files are confined to `artifacts/`, `external_data/` and `external_models/`.
Source data files at the repository root are treated as read-only inputs and are excluded
from version control.

---

## Outputs

```
artifacts/
  lock/model_spec.json          panel, checksum, seeds, algorithms, cohort roles
  results/                      metrics, predictions, criteria evaluation
  biology/                      tumor-normal, stage, enrichment, interaction results
  figures/                      SVG, PNG and PDF for every figure
  tables/                       main and supplementary tables as CSV
  audit/                        per-stage machine-readable audit records
  manifests/                    SHA-256 manifests of inputs and generated artifacts
```

`artifacts/lock/model_spec.json` is the only generated file tracked in git. It records
the locked 12-gene panel, its SHA-256, the random seed and the cohort roles used for
external scoring, so a reader can inspect exactly what was validated without rerunning
the pipeline.

---

## Reproducibility

- Every stage writes a JSON audit record under `artifacts/audit/`.
- `artifacts/manifests/final_artifact_manifest.csv` records SHA-256 for every generated file.
- Seeds are fixed in `configs/analysis.toml` and recorded in the model lock.
- External API responses (g:Profiler, STRING) are cached so results can be re-derived offline.
- Unit tests cover cohort construction, aliquot selection, identifier mapping and GEO
  series parsing against synthetic fixtures, so they run without any patient data.

---

## Data availability

This repository contains code and derived results only. No patient-level source data are
redistributed.

- **TCGA-LUAD**: obtain from the [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/).
  The pipeline expects the STAR counts, VST matrix, metadata and methods audit tables at
  the repository root; filenames are configured in `configs/analysis.toml`.
- **GEO cohorts**: GSE72094, GSE68465 and GSE31210 are downloaded by `prepare-geo` into
  `external_data/geo/`.
- **GENCODE v36**: downloaded into `external_data/gencode/`.

---

## Citation

If you use this code, please cite the accompanying manuscript. Citation metadata will be
added here on publication.

## License

Released under the MIT License. See [LICENSE](LICENSE).

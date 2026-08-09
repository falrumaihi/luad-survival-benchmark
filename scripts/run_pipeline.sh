#!/usr/bin/env bash
# Run the full LUAD survival benchmark end to end on Linux or macOS.
#
# Interpreter locations come from the LUAD_PYTHON and LUAD_RSCRIPT environment
# variables, falling back to python3 and Rscript on PATH.
#
#   ./scripts/run_pipeline.sh                 # resume from an existing lock
#   FORCE_REFIT=1 ./scripts/run_pipeline.sh   # repeat selection and relock
set -euo pipefail

PYTHON="${LUAD_PYTHON:-python3}"
RSCRIPT="${LUAD_RSCRIPT:-Rscript}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-200}"
FORCE_REFIT="${FORCE_REFIT:-0}"
BOOTSTRAP_R="${BOOTSTRAP_R:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$workspace/configs/analysis.toml"
export PYTHONPATH="$workspace/src"

for tool in "$PYTHON" "$RSCRIPT"; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "Interpreter not found on PATH: $tool" >&2
        echo "Set LUAD_PYTHON / LUAD_RSCRIPT to the correct executables." >&2
        exit 1
    }
done

required=(
    "TCGA_TCGA-LUAD_STAR_Counts.csv"
    "TCGA-LUAD_vst_normalized_matrix.csv"
    "TCGA_TCGA-LUAD_Metadata.csv"
    "TCGA-LUAD_methods_audit_table.csv"
    "external_data/gencode/gencode.v36.annotation.gtf.gz"
)
for relative in "${required[@]}"; do
    [ -e "$workspace/$relative" ] || {
        echo "Required input is missing: $relative" >&2
        echo "See the data availability section of README.md." >&2
        exit 1
    }
done

stage() { "$PYTHON" -m luad_biomarker.cli "$@"; }

stage build-cohort --config "$config"
stage prepare-expression --config "$config"
stage build-annotation --config "$config"

counts="$workspace/artifacts/interim/tcga_luad_selected_raw_counts.csv"
vst="$workspace/artifacts/interim/tcga_luad_selected_vst.rds"
audit="$workspace/artifacts/audit/normalization_audit.json"

[ "$BOOTSTRAP_R" = "1" ] && "$RSCRIPT" "$workspace/R/bootstrap_dependencies.R" "$workspace"
"$RSCRIPT" "$workspace/R/01_normalize_counts.R" "$workspace" "$counts" "$vst" "$audit"
"$RSCRIPT" "$workspace/R/02_export_model_matrix.R" "$workspace"

for cohort in GSE72094 GSE68465 GSE31210; do
    stage prepare-geo "$cohort" --config "$config"
done

if [ "$FORCE_REFIT" = "1" ] || [ ! -f "$workspace/artifacts/lock/model_spec.json" ]; then
    stage run-analysis --config "$config" --bootstrap-replicates "$BOOTSTRAP_REPLICATES"
else
    echo "Resuming from the existing model lock."
    stage score-external --config "$config"
fi

stage extended-validation --config "$config"
stage supplementary-models --config "$config"
stage biological-validation --config "$config"
stage make-figures --config "$config"
stage make-reporting --config "$config"

[ "$SKIP_TESTS" = "1" ] || "$PYTHON" -m pytest "$workspace/tests"

echo
echo "Pipeline complete. Results are under artifacts/; see artifacts/results/publication_gate.json."

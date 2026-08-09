<#
.SYNOPSIS
    Run the full LUAD survival benchmark end to end.

.DESCRIPTION
    Executes every analysis stage in order: cohort construction, expression
    preparation, GENCODE annotation, R normalization, external cohort
    harmonization, panel selection and locking, external scoring, extended
    validation, post-lock sensitivity models, biological evaluation, figures
    and tables.

    Interpreter locations are taken from -Python and -Rscript, falling back to
    the LUAD_PYTHON and LUAD_RSCRIPT environment variables and then to
    "python" and "Rscript" on PATH.

.EXAMPLE
    ./scripts/run_pipeline.ps1

.EXAMPLE
    ./scripts/run_pipeline.ps1 -ForceRefit -BootstrapReplicates 500
#>
param(
    [string]$Python  = $(if ($env:LUAD_PYTHON)  { $env:LUAD_PYTHON }  else { "python" }),
    [string]$Rscript = $(if ($env:LUAD_RSCRIPT) { $env:LUAD_RSCRIPT } else { "Rscript" }),
    [switch]$BootstrapR,
    [switch]$ForceRefit,
    [switch]$SkipTests,
    [int]$BootstrapReplicates = 200
)

$ErrorActionPreference = "Stop"
$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$config = Join-Path $workspace "configs\analysis.toml"

foreach ($tool in @($Python, $Rscript)) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Interpreter not found on PATH: $tool. Pass -Python/-Rscript or set LUAD_PYTHON/LUAD_RSCRIPT."
    }
}

$requiredInputs = @(
    "TCGA_TCGA-LUAD_STAR_Counts.csv",
    "TCGA-LUAD_vst_normalized_matrix.csv",
    "TCGA_TCGA-LUAD_Metadata.csv",
    "TCGA-LUAD_methods_audit_table.csv",
    "external_data\gencode\gencode.v36.annotation.gtf.gz"
)
foreach ($relative in $requiredInputs) {
    if (-not (Test-Path -LiteralPath (Join-Path $workspace $relative))) {
        throw "Required input is missing: $relative. See the data availability section of README.md."
    }
}

function Invoke-Stage {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $env:PYTHONPATH = Join-Path $workspace "src"
    & $Python -m luad_biomarker.cli @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage failed with exit code ${LASTEXITCODE}: $Arguments" }
}

Invoke-Stage build-cohort --config $config
Invoke-Stage prepare-expression --config $config
Invoke-Stage build-annotation --config $config

# --- R normalization -------------------------------------------------------
$counts      = Join-Path $workspace "artifacts\interim\tcga_luad_selected_raw_counts.csv"
$vstOutput   = Join-Path $workspace "artifacts\interim\tcga_luad_selected_vst.rds"
$auditOutput = Join-Path $workspace "artifacts\audit\normalization_audit.json"

if ($BootstrapR) {
    & $Rscript (Join-Path $workspace "R\bootstrap_dependencies.R") $workspace
    if ($LASTEXITCODE -ne 0) { throw "R dependency bootstrap failed with exit code $LASTEXITCODE" }
}
& $Rscript (Join-Path $workspace "R\01_normalize_counts.R") $workspace $counts $vstOutput $auditOutput
if ($LASTEXITCODE -ne 0) { throw "R normalization failed with exit code $LASTEXITCODE" }

& $Rscript (Join-Path $workspace "R\02_export_model_matrix.R") $workspace
if ($LASTEXITCODE -ne 0) { throw "R model-matrix export failed with exit code $LASTEXITCODE" }

# --- External cohorts ------------------------------------------------------
foreach ($cohort in @("GSE72094", "GSE68465", "GSE31210")) {
    Invoke-Stage prepare-geo $cohort --config $config
}

# --- Selection, lock and scoring -------------------------------------------
$modelLock = Join-Path $workspace "artifacts\lock\model_spec.json"
if ($ForceRefit -or -not (Test-Path -LiteralPath $modelLock)) {
    Invoke-Stage run-analysis --config $config --bootstrap-replicates "$BootstrapReplicates"
} else {
    Write-Host "Resuming from the existing model lock: $modelLock"
    Invoke-Stage score-external --config $config
}

Invoke-Stage extended-validation --config $config
Invoke-Stage supplementary-models --config $config
Invoke-Stage biological-validation --config $config
Invoke-Stage make-figures --config $config
Invoke-Stage make-reporting --config $config

if (-not $SkipTests) {
    & $Python -m pytest (Join-Path $workspace "tests")
    if ($LASTEXITCODE -ne 0) { throw "Tests failed with exit code $LASTEXITCODE" }
}

Write-Host ""
Write-Host "Pipeline complete. Results are under artifacts/; see artifacts/results/publication_gate.json."

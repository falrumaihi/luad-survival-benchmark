from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .cohort import build_patient_cohort, read_metadata, scan_counts_for_barcodes
from .config import load_config
from .expression import extract_selected_raw_counts
from .annotation import build_gencode_mapping
from .geo import COHORT_SPECS, prepare_geo_cohort
from .modeling import run_analysis, score_locked_external
from .validation import run_extended_validation
from .supplementary import run_supplementary_models
from .figures import make_all_figures
from .reporting import generate_reporting_package
from .biology import run_biological_validation
from .io_utils import write_csv_atomic, write_json_atomic
from .provenance import file_manifest, run_metadata


COHORT_FIELDS = [
    "patient_id",
    "selected_barcode",
    "os_days",
    "os_event",
    "vital_status",
    "stage",
    "stage_group",
    "age_years",
    "sex",
    "race",
    "ethnicity",
    "smoking_status",
    "pack_years",
    "primary_diagnosis",
    "reported_diagnosis_luad_like",
    "prior_treatment",
    "primary_aliquot_count",
]

ALIQUOT_FIELDS = [
    "patient_id",
    "barcode",
    "sample_submitter_id",
    "vial_rank",
    "raw_library_size",
    "nonzero_genes",
    "selected",
    "selection_rule",
]

EXCLUSION_FIELDS = ["patient_id", "reason", "os_days"]
MANIFEST_FIELDS = ["path", "bytes", "modified_utc", "sha256"]


def _duplicate_primary_barcodes(metadata_rows: list[dict[str, str]], sample_type: str) -> set[str]:
    by_patient: dict[str, list[dict[str, str]]] = {}
    for row in metadata_rows:
        if row.get("sample_type") != sample_type:
            continue
        pid = row.get("submitter_id") or row["barcode"][:12]
        by_patient.setdefault(pid, []).append(row)
    return {
        row["barcode"]
        for rows in by_patient.values()
        if len(rows) > 1
        for row in rows
    }


def build_cohort_command(config_path: str) -> int:
    config = load_config(config_path)
    cohort_cfg = config.raw["cohort"]
    metadata_rows, metadata_fields = read_metadata(config.metadata)
    required_metadata_fields = {
        "barcode",
        "submitter_id",
        "sample_type",
        "vital_status",
        "days_to_death",
        "days_to_last_follow_up",
    }
    missing_fields = required_metadata_fields.difference(metadata_fields)
    if missing_fields:
        raise ValueError(f"Metadata is missing required fields: {sorted(missing_fields)}")

    duplicate_barcodes = _duplicate_primary_barcodes(
        metadata_rows,
        cohort_cfg["eligible_sample_type"],
    )
    counts_audit = scan_counts_for_barcodes(config.raw_counts, duplicate_barcodes)
    result = build_patient_cohort(
        metadata_rows,
        counts_audit,
        eligible_sample_type=cohort_cfg["eligible_sample_type"],
        minimum_os_days_exclusive=float(cohort_cfg["minimum_os_days_exclusive"]),
        preferred_vial=cohort_cfg["prefer_vial"],
    )

    artifacts = config.artifacts
    write_csv_atomic(
        artifacts / "interim" / "tcga_luad_patient_cohort.csv",
        result.cohort_rows,
        COHORT_FIELDS,
    )
    write_csv_atomic(
        artifacts / "audit" / "primary_aliquot_selection.csv",
        result.aliquot_rows,
        ALIQUOT_FIELDS,
    )
    write_csv_atomic(
        artifacts / "audit" / "cohort_exclusions.csv",
        result.excluded_rows,
        EXCLUSION_FIELDS,
    )
    write_json_atomic(artifacts / "audit" / "cohort_audit.json", result.audit)

    manifest_rows = file_manifest(
        [config.metadata, config.raw_counts, config.vst_matrix, config.methods_audit],
        config.workspace,
    )
    write_csv_atomic(
        artifacts / "manifests" / "input_manifest.csv",
        manifest_rows,
        MANIFEST_FIELDS,
    )
    write_json_atomic(
        artifacts / "manifests" / "run_metadata.json",
        run_metadata(sys.argv, config.config_path),
    )

    print(
        "Built TCGA-LUAD patient cohort: "
        f"n={result.audit['eligible_os_patients']}, "
        f"events={result.audit['eligible_os_events']}, "
        f"genes={result.audit['counts_gene_rows']}"
    )
    return 0


def prepare_expression_command(config_path: str) -> int:
    config = load_config(config_path)
    cohort_path = config.artifacts / "interim" / "tcga_luad_patient_cohort.csv"
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Patient cohort not found: {cohort_path}. Run build-cohort first."
        )

    import csv

    with cohort_path.open("r", encoding="utf-8", newline="") as handle:
        cohort_rows = list(csv.DictReader(handle))
    selected = [row["selected_barcode"] for row in cohort_rows]
    output_path = config.artifacts / "interim" / "tcga_luad_selected_raw_counts.csv"
    audit = extract_selected_raw_counts(config.raw_counts, selected, output_path)
    write_json_atomic(config.artifacts / "audit" / "expression_audit.json", audit)
    output_manifest = file_manifest([cohort_path, output_path], config.workspace)
    write_csv_atomic(
        config.artifacts / "manifests" / "expression_output_manifest.csv",
        output_manifest,
        MANIFEST_FIELDS,
    )
    print(
        "Prepared selected raw-count matrix: "
        f"samples={audit['selected_samples']}, genes={audit['gene_rows']}"
    )
    return 0


def build_annotation_command(config_path: str) -> int:
    config = load_config(config_path)
    annotation_cfg = config.raw["annotations"]
    gtf_path = (config.workspace / annotation_cfg["gencode_gtf"]).resolve()
    if not gtf_path.exists():
        raise FileNotFoundError(
            f"GENCODE GTF not found: {gtf_path}. Download the configured source first."
        )
    selected_counts = config.artifacts / "interim" / "tcga_luad_selected_raw_counts.csv"
    mapping_path = config.artifacts / "interim" / "gencode_v36_gene_mapping.csv"
    audit = build_gencode_mapping(gtf_path, selected_counts, mapping_path)
    audit["source_url"] = annotation_cfg["gencode_source_url"]
    audit["gencode_release"] = annotation_cfg["gencode_release"]
    audit["gencode_assembly"] = annotation_cfg["gencode_assembly"]
    write_json_atomic(config.artifacts / "audit" / "annotation_audit.json", audit)
    annotation_manifest = file_manifest([gtf_path, mapping_path], config.workspace)
    write_csv_atomic(
        config.artifacts / "manifests" / "annotation_manifest.csv",
        annotation_manifest,
        MANIFEST_FIELDS,
    )
    print(
        "Built GENCODE mapping: "
        f"mapped={audit['selected_exact_mapped']}/{audit['selected_gene_rows']}"
    )
    return 0


def prepare_geo_command(config_path: str, accession: str) -> int:
    config = load_config(config_path)
    platform_file = COHORT_SPECS[accession]["platform_file"]
    audit = prepare_geo_cohort(
        accession=accession,
        matrix_path=config.workspace
        / "external_data"
        / "geo"
        / accession
        / f"{accession}_series_matrix.txt.gz",
        platform_path=config.workspace
        / "external_data"
        / "geo"
        / "platforms"
        / platform_file,
        target_matrix_path=config.artifacts
        / "interim"
        / "tcga_luad_model_matrix.csv.gz",
        output_dir=config.artifacts / "external",
    )
    print(
        f"Prepared {accession}: n={audit['eligible_os_samples']}, "
        f"events={audit['events']}, genes={audit['expression_features']}"
    )
    return 0


def run_analysis_command(config_path: str, bootstrap_replicates: int) -> int:
    config = load_config(config_path)
    summary = run_analysis(config.artifacts, bootstrap_replicates=bootstrap_replicates)
    print(
        "Completed locked analysis: "
        f"panel={','.join(summary['panel_genes'])}; "
        f"external_cohorts={len(summary['external_cohorts'])}"
    )
    return 0


def score_external_command(config_path: str) -> int:
    config = load_config(config_path)
    summary = score_locked_external(config.artifacts)
    print(
        "Completed locked external scoring: "
        f"cohorts={','.join(summary['external_cohorts'])}"
    )
    return 0


def extended_validation_command(config_path: str) -> int:
    config = load_config(config_path)
    gate = run_extended_validation(config.artifacts)
    print(
        "Completed calibration and publication gate: "
        f"passed={gate['gate_passed']}; interpretation={gate['required_interpretation']}"
    )
    return 0


def supplementary_command(config_path: str) -> int:
    config = load_config(config_path)
    audit = run_supplementary_models(config.artifacts)
    print(
        "Completed post-lock supplementary challengers: "
        f"gpu_models={','.join(audit['gpu_models'])}"
    )
    return 0


def figures_command(config_path: str) -> int:
    config = load_config(config_path)
    records = make_all_figures(config.artifacts)
    print(f"Generated {len(records) // 3} figures in SVG, PNG, and PDF")
    return 0


def reporting_command(config_path: str) -> int:
    config = load_config(config_path)
    report = generate_reporting_package(config.artifacts, config.workspace)
    print(
        "Generated manuscript reporting package: "
        f"tables={report['table_count']}; gate_passed={report['publication_gate_passed']}"
    )
    return 0


def biology_command(config_path: str) -> int:
    config = load_config(config_path)
    report = run_biological_validation(config.artifacts, config.workspace)
    print(
        "Completed post-selection biological validation: "
        f"paired={report['paired_patients']}; paired_FDR_genes={len(report['paired_fdr_significant_genes'])}; "
        f"pathway_terms={report['significant_functional_terms']}"
    )
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="luad-biomarker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser(
        "build-cohort",
        help="Validate raw inputs and build the patient-level TCGA-LUAD cohort",
    )
    build.add_argument("--config", required=True, type=str)
    prepare = subparsers.add_parser(
        "prepare-expression",
        help="Extract the selected patient-level raw-count matrix and audit gene identifiers",
    )
    prepare.add_argument("--config", required=True, type=str)
    annotation = subparsers.add_parser(
        "build-annotation",
        help="Build and audit the exact GENCODE v36 gene mapping",
    )
    annotation.add_argument("--config", required=True, type=str)
    geo = subparsers.add_parser(
        "prepare-geo",
        help="Harmonize an external GEO expression/survival cohort",
    )
    geo.add_argument("accession", choices=sorted(COHORT_SPECS))
    geo.add_argument("--config", required=True, type=str)
    analysis = subparsers.add_parser(
        "run-analysis",
        help="Select the panel, run CPU/GPU models, lock, and score external cohorts",
    )
    analysis.add_argument("--config", required=True, type=str)
    analysis.add_argument("--bootstrap-replicates", type=int, default=200)
    score = subparsers.add_parser(
        "score-external",
        help="Resume external validation from an immutable model lock",
    )
    score.add_argument("--config", required=True, type=str)
    extended = subparsers.add_parser(
        "extended-validation",
        help="Run Uno/AUC, paired bootstrap, CSD calibration, and publication gate",
    )
    extended.add_argument("--config", required=True, type=str)
    supplementary = subparsers.add_parser(
        "supplementary-models",
        help="Run post-lock XGBoost, clinical-PFN, and stage-only challengers",
    )
    supplementary.add_argument("--config", required=True, type=str)
    figures = subparsers.add_parser(
        "make-figures",
        help="Render all manuscript figures in the supplied reference style",
    )
    figures.add_argument("--config", required=True, type=str)
    reporting = subparsers.add_parser(
        "make-reporting",
        help="Write the main and supplementary tables and the artifact manifest",
    )
    reporting.add_argument("--config", required=True, type=str)
    biology = subparsers.add_parser(
        "biological-validation",
        help="Run post-selection tumor-normal, stage, pathway, and interaction validation",
    )
    biology.add_argument("--config", required=True, type=str)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "build-cohort":
        return build_cohort_command(args.config)
    if args.command == "prepare-expression":
        return prepare_expression_command(args.config)
    if args.command == "build-annotation":
        return build_annotation_command(args.config)
    if args.command == "prepare-geo":
        return prepare_geo_command(args.config, args.accession)
    if args.command == "run-analysis":
        return run_analysis_command(args.config, args.bootstrap_replicates)
    if args.command == "score-external":
        return score_external_command(args.config)
    if args.command == "extended-validation":
        return extended_validation_command(args.config)
    if args.command == "supplementary-models":
        return supplementary_command(args.config)
    if args.command == "make-figures":
        return figures_command(args.config)
    if args.command == "make-reporting":
        return reporting_command(args.config)
    if args.command == "biological-validation":
        return biology_command(args.config)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

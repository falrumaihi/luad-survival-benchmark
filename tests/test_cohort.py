from __future__ import annotations

import csv
from pathlib import Path

from luad_biomarker.cohort import (
    build_patient_cohort,
    scan_counts_for_barcodes,
    select_primary_aliquot,
)
from luad_biomarker.expression import extract_selected_raw_counts
from luad_biomarker.annotation import build_gencode_mapping, parse_gtf_attributes


def _row(
    barcode: str,
    patient: str,
    sample_type: str = "Primary Tumor",
    vital_status: str = "Alive",
    death: str = "NA",
    followup: str = "100",
    stage: str = "Stage IA",
) -> dict[str, str]:
    return {
        "barcode": barcode,
        "submitter_id": patient,
        "sample_submitter_id": barcode[:16],
        "sample_type": sample_type,
        "vital_status": vital_status,
        "days_to_death": death,
        "days_to_last_follow_up": followup,
        "ajcc_pathologic_stage": stage,
        "age_at_index": "65",
        "age_at_diagnosis": "NA",
        "gender": "female",
        "race": "white",
        "ethnicity": "not hispanic or latino",
        "tobacco_smoking_status": "Current Smoker",
        "pack_years_smoked": "30",
        "primary_diagnosis": "Adenocarcinoma, NOS",
        "prior_treatment": "No",
    }


def _write_counts(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "TCGA-AA-0001-01A-11R-0000-01",
                "TCGA-AA-0001-01B-11R-0000-01",
                "TCGA-AA-0002-01A-11R-0000-01",
                "gene",
            ]
        )
        writer.writerow([10, 100, 2, "ENSG1.1"])
        writer.writerow([0, 100, 3, "ENSG2.1"])


def test_counts_scan_and_vial_priority(tmp_path: Path) -> None:
    counts = tmp_path / "counts.csv"
    _write_counts(counts)
    audit = scan_counts_for_barcodes(
        counts,
        {
            "TCGA-AA-0001-01A-11R-0000-01",
            "TCGA-AA-0001-01B-11R-0000-01",
        },
    )
    assert audit.gene_rows == 2
    assert audit.requested_library_sizes["TCGA-AA-0001-01B-11R-0000-01"] == 200

    rows = [
        _row("TCGA-AA-0001-01A-11R-0000-01", "TCGA-AA-0001"),
        _row("TCGA-AA-0001-01B-11R-0000-01", "TCGA-AA-0001"),
    ]
    selected = select_primary_aliquot(rows, audit.requested_library_sizes)
    assert selected["barcode"] == "TCGA-AA-0001-01A-11R-0000-01"


def test_patient_cohort_endpoint_and_exclusions(tmp_path: Path) -> None:
    counts = tmp_path / "counts.csv"
    _write_counts(counts)
    audit = scan_counts_for_barcodes(
        counts,
        {
            "TCGA-AA-0001-01A-11R-0000-01",
            "TCGA-AA-0001-01B-11R-0000-01",
        },
    )
    rows = [
        _row(
            "TCGA-AA-0001-01A-11R-0000-01",
            "TCGA-AA-0001",
            vital_status="Dead",
            death="450",
            followup="400",
        ),
        _row(
            "TCGA-AA-0001-01B-11R-0000-01",
            "TCGA-AA-0001",
            vital_status="Dead",
            death="450",
            followup="400",
        ),
        _row(
            "TCGA-AA-0002-01A-11R-0000-01",
            "TCGA-AA-0002",
            vital_status="Alive",
            followup="0",
        ),
    ]
    result = build_patient_cohort(rows, audit)
    assert len(result.cohort_rows) == 1
    assert result.cohort_rows[0]["os_days"] == 450
    assert result.cohort_rows[0]["os_event"] == 1
    assert result.cohort_rows[0]["stage_group"] == "I"
    assert result.audit["duplicate_primary_tumor_patients"] == 1
    assert result.audit["exclusion_counts"] == {"nonpositive_os_time": 1}


def test_selected_expression_extraction(tmp_path: Path) -> None:
    counts = tmp_path / "counts.csv"
    _write_counts(counts)
    output = tmp_path / "selected.csv"
    audit = extract_selected_raw_counts(
        counts,
        ["TCGA-AA-0002-01A-11R-0000-01"],
        output,
    )
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows == [
        ["gene", "TCGA-AA-0002-01A-11R-0000-01"],
        ["ENSG1.1", "2"],
        ["ENSG2.1", "3"],
    ]
    assert audit["selected_samples"] == 1
    assert audit["gene_rows"] == 2


def test_gencode_exact_mapping_preserves_par_y(tmp_path: Path) -> None:
    counts = tmp_path / "selected.csv"
    with counts.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gene", "S1"])
        writer.writerow(["ENSG00000000003.15", 10])
        writer.writerow(["ENSG00000002586.20_PAR_Y", 4])
    gtf = tmp_path / "gencode.gtf.gz"
    import gzip

    with gzip.open(gtf, "wt", encoding="utf-8") as handle:
        handle.write(
            'chrX\tHAVANA\tgene\t1\t10\t.\t+\t.\tgene_id "ENSG00000000003.15"; '
            'gene_type "protein_coding"; gene_name "TSPAN6";\n'
        )
        handle.write(
            'chrY\tHAVANA\tgene\t20\t30\t.\t-\t.\tgene_id "ENSG00000002586.20_PAR_Y"; '
            'gene_type "protein_coding"; gene_name "CD99";\n'
        )
    output = tmp_path / "mapping.csv"
    audit = build_gencode_mapping(gtf, counts, output)
    assert audit["selected_exact_mapped"] == 2
    assert audit["selected_exact_missing"] == 0
    attrs = parse_gtf_attributes('gene_id "ENSG1.1"; gene_name "A";')
    assert attrs == {"gene_id": "ENSG1.1", "gene_name": "A"}

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .io_utils import clean_text, parse_float


@dataclass(frozen=True)
class CountsAudit:
    sample_barcodes: tuple[str, ...]
    gene_rows: int
    requested_library_sizes: dict[str, int]
    requested_nonzero_genes: dict[str, int]


@dataclass(frozen=True)
class CohortBuildResult:
    cohort_rows: list[dict[str, Any]]
    aliquot_rows: list[dict[str, Any]]
    excluded_rows: list[dict[str, Any]]
    audit: dict[str, Any]


def read_metadata(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata file has no header: {path}")
        rows = list(reader)
        return rows, list(reader.fieldnames)


def scan_counts_for_barcodes(path: Path, requested: Iterable[str]) -> CountsAudit:
    requested_set = set(requested)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Empty counts matrix: {path}") from exc
        if not header or header[-1].strip().lower() != "gene":
            raise ValueError("Counts matrix must store the gene identifier in the last column")

        samples = header[:-1]
        if len(samples) != len(set(samples)):
            raise ValueError("Counts matrix contains duplicate sample barcodes")
        missing = requested_set.difference(samples)
        if missing:
            raise ValueError(f"Requested barcodes absent from counts matrix: {sorted(missing)}")

        indices = {barcode: samples.index(barcode) for barcode in requested_set}
        library_sizes = {barcode: 0 for barcode in requested_set}
        nonzero = {barcode: 0 for barcode in requested_set}
        gene_rows = 0
        expected_columns = len(header)
        for row_number, row in enumerate(reader, start=2):
            if len(row) != expected_columns:
                raise ValueError(
                    f"Counts row {row_number} has {len(row)} columns; expected {expected_columns}"
                )
            gene_rows += 1
            for barcode, index in indices.items():
                try:
                    value = int(row[index])
                except ValueError as exc:
                    raise ValueError(
                        f"Non-integer raw count at row {row_number}, sample {barcode}: {row[index]!r}"
                    ) from exc
                if value < 0:
                    raise ValueError(
                        f"Negative raw count at row {row_number}, sample {barcode}: {value}"
                    )
                library_sizes[barcode] += value
                nonzero[barcode] += int(value > 0)

    return CountsAudit(
        sample_barcodes=tuple(samples),
        gene_rows=gene_rows,
        requested_library_sizes=library_sizes,
        requested_nonzero_genes=nonzero,
    )


def patient_id(row: dict[str, str]) -> str:
    submitter = clean_text(row.get("submitter_id"))
    if submitter:
        return submitter
    barcode = clean_text(row.get("barcode"))
    if barcode and len(barcode) >= 12:
        return barcode[:12]
    raise ValueError("Metadata row has neither submitter_id nor a valid barcode")


def _vial_rank(barcode: str, preferred_vial: str) -> int:
    parts = barcode.split("-")
    if len(parts) < 4 or len(parts[3]) < 3:
        return 100
    vial = parts[3][2].upper()
    if vial == preferred_vial.upper():
        return 0
    if "A" <= vial <= "Z":
        return 1 + ord(vial) - ord("A")
    return 100


def select_primary_aliquot(
    primary_rows: list[dict[str, str]],
    library_sizes: dict[str, int],
    preferred_vial: str = "A",
) -> dict[str, str]:
    if not primary_rows:
        raise ValueError("Cannot select from an empty primary-aliquot list")

    def key(row: dict[str, str]) -> tuple[int, int, str]:
        barcode = row["barcode"]
        return (
            _vial_rank(barcode, preferred_vial),
            -library_sizes.get(barcode, -1),
            barcode,
        )

    return min(primary_rows, key=key)


def _coalesce(rows: list[dict[str, str]], field: str) -> tuple[str | None, list[str]]:
    values = [clean_text(row.get(field)) for row in rows]
    unique = sorted({value for value in values if value is not None})
    return (unique[0] if unique else None), unique


def _stage_group(stage: str | None) -> str | None:
    if stage is None:
        return None
    normalized = stage.upper().replace(" ", "")
    if normalized.startswith("STAGEIV"):
        return "IV"
    if normalized.startswith("STAGEIII"):
        return "III"
    if normalized.startswith("STAGEII"):
        return "II"
    if normalized.startswith("STAGEI"):
        return "I"
    return None


def _is_luad_like_diagnosis_label(diagnosis: str | None) -> bool | None:
    if diagnosis is None:
        return None
    value = diagnosis.lower()
    accepted_tokens = (
        "adenocarcinoma",
        "acinar cell carcinoma",
        "bronchiolo-alveolar",
        "bronchio-alveolar",
        "solid carcinoma",
        "invasive micropapillary carcinoma",
    )
    return any(token in value for token in accepted_tokens)


def build_patient_cohort(
    metadata_rows: list[dict[str, str]],
    counts_audit: CountsAudit,
    eligible_sample_type: str = "Primary Tumor",
    minimum_os_days_exclusive: float = 0.0,
    preferred_vial: str = "A",
) -> CohortBuildResult:
    metadata_barcodes = [clean_text(row.get("barcode")) for row in metadata_rows]
    if any(barcode is None for barcode in metadata_barcodes):
        raise ValueError("Every metadata row must have a barcode")
    if len(metadata_barcodes) != len(set(metadata_barcodes)):
        raise ValueError("Metadata contains duplicate full aliquot barcodes")

    count_barcodes = set(counts_audit.sample_barcodes)
    metadata_barcode_set = set(metadata_barcodes)
    if count_barcodes != metadata_barcode_set:
        only_counts = sorted(count_barcodes - metadata_barcode_set)
        only_metadata = sorted(metadata_barcode_set - count_barcodes)
        raise ValueError(
            "Counts/metadata barcode mismatch: "
            f"counts_only={only_counts[:10]}, metadata_only={only_metadata[:10]}"
        )

    by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata_rows:
        by_patient[patient_id(row)].append(row)

    cohort_rows: list[dict[str, Any]] = []
    aliquot_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    conflict_counts: Counter[str] = Counter()
    primary_patient_count = 0
    duplicate_primary_patients = 0

    for pid in sorted(by_patient):
        patient_rows = by_patient[pid]
        primary_rows = [row for row in patient_rows if row.get("sample_type") == eligible_sample_type]
        if not primary_rows:
            excluded_rows.append({"patient_id": pid, "reason": "no_primary_tumor"})
            continue
        primary_patient_count += 1
        duplicate_primary_patients += int(len(primary_rows) > 1)
        selected = select_primary_aliquot(
            primary_rows,
            counts_audit.requested_library_sizes,
            preferred_vial=preferred_vial,
        )

        for row in primary_rows:
            barcode = row["barcode"]
            aliquot_rows.append(
                {
                    "patient_id": pid,
                    "barcode": barcode,
                    "sample_submitter_id": clean_text(row.get("sample_submitter_id")),
                    "vial_rank": _vial_rank(barcode, preferred_vial),
                    "raw_library_size": counts_audit.requested_library_sizes.get(barcode),
                    "nonzero_genes": counts_audit.requested_nonzero_genes.get(barcode),
                    "selected": barcode == selected["barcode"],
                    "selection_rule": (
                        "sole_primary" if len(primary_rows) == 1 else "vial_then_library_size_then_barcode"
                    ),
                }
            )

        fields = [
            "vital_status",
            "days_to_death",
            "days_to_last_follow_up",
            "ajcc_pathologic_stage",
            "age_at_index",
            "age_at_diagnosis",
            "gender",
            "race",
            "ethnicity",
            "tobacco_smoking_status",
            "pack_years_smoked",
            "primary_diagnosis",
            "prior_treatment",
        ]
        values: dict[str, str | None] = {}
        for field in fields:
            value, unique = _coalesce(patient_rows, field)
            values[field] = value
            if len(unique) > 1:
                conflict_counts[field] += 1

        vital_status = values["vital_status"]
        event = 1 if vital_status and vital_status.lower() == "dead" else 0
        os_days = (
            parse_float(values["days_to_death"])
            if event == 1
            else parse_float(values["days_to_last_follow_up"])
        )
        if vital_status is None:
            excluded_rows.append({"patient_id": pid, "reason": "missing_vital_status"})
            continue
        if os_days is None:
            excluded_rows.append({"patient_id": pid, "reason": "missing_os_time"})
            continue
        if os_days <= minimum_os_days_exclusive:
            excluded_rows.append(
                {"patient_id": pid, "reason": "nonpositive_os_time", "os_days": os_days}
            )
            continue

        age_years = parse_float(values["age_at_index"])
        if age_years is None:
            age_days = parse_float(values["age_at_diagnosis"])
            age_years = age_days / 365.25 if age_days is not None else None

        diagnosis = values["primary_diagnosis"]
        cohort_rows.append(
            {
                "patient_id": pid,
                "selected_barcode": selected["barcode"],
                "os_days": os_days,
                "os_event": event,
                "vital_status": vital_status,
                "stage": values["ajcc_pathologic_stage"],
                "stage_group": _stage_group(values["ajcc_pathologic_stage"]),
                "age_years": age_years,
                "sex": values["gender"],
                "race": values["race"],
                "ethnicity": values["ethnicity"],
                "smoking_status": values["tobacco_smoking_status"],
                "pack_years": parse_float(values["pack_years_smoked"]),
                "primary_diagnosis": diagnosis,
                "reported_diagnosis_luad_like": _is_luad_like_diagnosis_label(diagnosis),
                "prior_treatment": values["prior_treatment"],
                "primary_aliquot_count": len(primary_rows),
            }
        )

    sample_type_counts = Counter(row.get("sample_type") for row in metadata_rows)
    stage_counts = Counter(row["stage_group"] or "missing" for row in cohort_rows)
    diagnosis_counts = Counter(row["primary_diagnosis"] or "missing" for row in cohort_rows)
    audit = {
        "metadata_rows": len(metadata_rows),
        "metadata_columns": len(metadata_rows[0]) if metadata_rows else 0,
        "counts_samples": len(counts_audit.sample_barcodes),
        "counts_gene_rows": counts_audit.gene_rows,
        "unique_patients": len(by_patient),
        "primary_tumor_patients": primary_patient_count,
        "duplicate_primary_tumor_patients": duplicate_primary_patients,
        "eligible_os_patients": len(cohort_rows),
        "eligible_os_events": sum(int(row["os_event"]) for row in cohort_rows),
        "eligible_os_censored": sum(1 - int(row["os_event"]) for row in cohort_rows),
        "paired_primary_normal_patients": sum(
            any(row.get("sample_type") == eligible_sample_type for row in rows)
            and any(row.get("sample_type") == "Solid Tissue Normal" for row in rows)
            for rows in by_patient.values()
        ),
        "sample_type_counts": dict(sorted(sample_type_counts.items(), key=lambda item: str(item[0]))),
        "stage_group_counts_eligible": dict(sorted(stage_counts.items())),
        "reported_diagnosis_luad_like_false": sum(
            row["reported_diagnosis_luad_like"] is False for row in cohort_rows
        ),
        "reported_diagnosis_missing": sum(
            row["reported_diagnosis_luad_like"] is None for row in cohort_rows
        ),
        "reported_diagnosis_usage": (
            "audit_only_not_used_for_eligibility_or_modeling_due_to_implausible_flattened_labels"
        ),
        "top_primary_diagnoses": diagnosis_counts.most_common(15),
        "patient_level_field_conflicts": dict(sorted(conflict_counts.items())),
        "exclusion_counts": dict(Counter(row["reason"] for row in excluded_rows)),
    }
    return CohortBuildResult(
        cohort_rows=cohort_rows,
        aliquot_rows=aliquot_rows,
        excluded_rows=excluded_rows,
        audit=audit,
    )

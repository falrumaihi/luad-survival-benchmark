from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd


COHORT_SPECS: dict[str, dict[str, str]] = {
    "GSE72094": {
        "platform": "GPL10379",
        "platform_file": "GPL15048.data.txt",
        "platform_note": "GPL15048 is an alternate definition of canonical GPL10379",
        "time_field": "survival_time_in_days",
        "time_multiplier": "1",
        "event_field": "vital_status",
        "disease_field": "",
        "disease_value": "",
    },
    "GSE68465": {
        "platform": "GPL96",
        "platform_file": "GPL96.annot.gz",
        "platform_note": "Affymetrix HG-U133A",
        "time_field": "months_to_last_contact_or_death",
        "time_multiplier": "30.4375",
        "event_field": "vital_status",
        "disease_field": "disease_state",
        "disease_value": "lung adenocarcinoma",
    },
    "GSE31210": {
        "platform": "GPL570",
        "platform_file": "GPL570.annot.gz",
        "platform_note": "Affymetrix HG-U133 Plus 2.0",
        "time_field": "days_before_death/censor",
        "time_multiplier": "1",
        "event_field": "death",
        "disease_field": "",
        "disease_value": "",
    },
}


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def parse_series_metadata(path: Path) -> tuple[pd.DataFrame, int]:
    sample_rows: list[list[str]] = []
    table_skiprows = -1
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("!series_matrix_table_begin"):
                table_skiprows = line_number + 1
                break
            if line.startswith("!Sample_"):
                sample_rows.append(next(csv.reader([line.rstrip("\n")], delimiter="\t")))
    if table_skiprows < 0:
        raise ValueError(f"No series matrix table marker in {path}")

    fixed: dict[str, list[str]] = {}
    characteristics: list[list[str]] = []
    for row in sample_rows:
        key, values = row[0].removeprefix("!Sample_"), row[1:]
        if key == "characteristics_ch1":
            characteristics.append(values)
        elif key not in fixed:
            fixed[key] = values
    if "geo_accession" not in fixed:
        raise ValueError(f"No sample accessions in {path}")
    n_samples = len(fixed["geo_accession"])
    if any(len(values) != n_samples for values in fixed.values()):
        raise ValueError(f"Inconsistent sample metadata width in {path}")

    records: list[dict[str, str]] = []
    for index in range(n_samples):
        record = {key: values[index] for key, values in fixed.items()}
        for values in characteristics:
            value = values[index]
            if not value or ":" not in value:
                continue
            field, content = value.split(":", 1)
            field = field.strip().lower().replace(" ", "_")
            if field and field not in record:
                record[field] = content.strip()
        records.append(record)
    metadata = pd.DataFrame.from_records(records).set_index("geo_accession", drop=False)
    return metadata, table_skiprows


def load_platform_map(path: Path, target_symbols: set[str]) -> tuple[dict[str, str], dict[str, Any]]:
    header_line = -1
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle):
            if line.startswith("ID\t"):
                header_line = line_number
                break
    if header_line < 0:
        raise ValueError(f"No platform annotation table header in {path}")
    table = pd.read_csv(
        path, sep="\t", compression="infer", skiprows=header_line, dtype=str, low_memory=False
    )
    symbol_column = "Gene symbol" if "Gene symbol" in table else "GeneSymbol"
    if "ID" not in table or symbol_column not in table:
        raise ValueError(f"Required ID/Gene symbol fields absent in {path}")
    symbols = table[symbol_column].fillna("").str.strip()
    unambiguous = symbols.ne("") & ~symbols.str.contains(r"\s*///\s*", regex=True)
    useful = table.loc[
        unambiguous & symbols.isin(target_symbols), ["ID", symbol_column]
    ].drop_duplicates("ID")
    mapping = dict(zip(useful["ID"], useful[symbol_column], strict=True))
    audit = {
        "platform_annotation_rows": int(len(table)),
        "unambiguous_probe_rows": int(unambiguous.sum()),
        "target_probe_rows": int(len(useful)),
        "target_symbols_mapped": int(useful[symbol_column].nunique()),
    }
    return mapping, audit


def load_target_expression(
    matrix_path: Path,
    table_skiprows: int,
    probe_to_symbol: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    retained: list[pd.DataFrame] = []
    matrix_probe_rows = 0
    for chunk in pd.read_csv(
        matrix_path,
        sep="\t",
        compression="infer",
        skiprows=table_skiprows,
        chunksize=2000,
        low_memory=False,
    ):
        id_col = chunk.columns[0]
        ids = chunk[id_col].astype(str)
        chunk = chunk.loc[~ids.str.startswith("!"), :]
        matrix_probe_rows += len(chunk)
        keep = chunk[id_col].isin(probe_to_symbol)
        if not keep.any():
            continue
        selected = chunk.loc[keep].copy()
        selected.insert(1, "gene_symbol", selected[id_col].map(probe_to_symbol))
        retained.append(selected)
    if not retained:
        raise ValueError(f"No mapped target probes found in {matrix_path}")
    probes = pd.concat(retained, ignore_index=True)
    id_col = probes.columns[0]
    values = probes.drop(columns=[id_col, "gene_symbol"]).apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"Non-numeric or missing expression values in {matrix_path}")
    values.insert(0, "gene_symbol", probes["gene_symbol"].to_numpy())
    collapsed = values.groupby("gene_symbol", sort=True).median(numeric_only=True)
    audit = {
        "matrix_probe_rows": int(matrix_probe_rows),
        "retained_probe_rows": int(len(probes)),
        "collapsed_gene_symbols": int(len(collapsed)),
    }
    return collapsed, audit


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.replace({"--": np.nan, "NA": np.nan, "na": np.nan, "": np.nan}),
        errors="coerce",
    )


def prepare_geo_cohort(
    accession: str,
    matrix_path: Path,
    platform_path: Path,
    target_matrix_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if accession not in COHORT_SPECS:
        raise ValueError(f"Unsupported GEO cohort: {accession}")
    spec = COHORT_SPECS[accession]
    target_symbols = set(pd.read_csv(target_matrix_path, nrows=0).columns) - {"selected_barcode"}
    metadata, skiprows = parse_series_metadata(matrix_path)
    probe_map, platform_audit = load_platform_map(platform_path, target_symbols)
    expression, expression_audit = load_target_expression(matrix_path, skiprows, probe_map)

    time_field = spec["time_field"]
    if time_field not in metadata:
        raise ValueError(
            f"{accession} lacks expected survival field {time_field}; "
            f"fields={sorted(metadata.columns)}"
        )
    time_days = _numeric(metadata[time_field]) * float(spec["time_multiplier"])
    default_vital = pd.Series(index=metadata.index, dtype=str)
    vital = metadata.get(spec["event_field"], default_vital).fillna("").str.lower()
    event = vital.eq("dead")
    eligible = time_days.gt(0) & vital.isin(["alive", "dead"])
    if spec["disease_field"]:
        default_disease = pd.Series(index=metadata.index, dtype=str)
        disease = metadata.get(spec["disease_field"], default_disease).fillna("").str.lower()
        eligible &= disease.eq(spec["disease_value"])

    sample_ids = [sample for sample in metadata.index[eligible] if sample in expression.columns]
    if not sample_ids:
        raise ValueError(f"No eligible samples overlap expression columns for {accession}")
    x = expression.loc[:, sample_ids].T
    transformed = False
    if float(np.nanpercentile(x.to_numpy(), 99)) > 100.0:
        x = np.log2(x.clip(lower=0.0) + 1.0)
        transformed = True

    med = x.median(axis=0)
    mad = (x - med).abs().median(axis=0) * 1.4826
    sd = x.std(axis=0, ddof=1).replace(0.0, 1.0)
    scale = mad.where(mad > 1e-8, sd).replace(0.0, 1.0)
    x = (x - med) / scale
    x.index.name = "sample_id"

    phenotype = metadata.loc[sample_ids].copy()
    phenotype.insert(1, "os_days", time_days.loc[sample_ids].astype(float))
    phenotype.insert(2, "os_event", event.loc[sample_ids].astype(int))
    output_dir.mkdir(parents=True, exist_ok=True)
    expression_output = output_dir / f"{accession}_expression.csv.gz"
    phenotype_output = output_dir / f"{accession}_phenotype.csv"
    x.reset_index().to_csv(expression_output, index=False, compression="gzip")
    phenotype.reset_index(drop=True).to_csv(phenotype_output, index=False)

    audit = {
        "accession": accession,
        "platform_used": spec["platform"],
        "platform_note": spec["platform_note"],
        "series_samples": int(len(metadata)),
        "eligible_os_samples": int(len(sample_ids)),
        "events": int(event.loc[sample_ids].sum()),
        "expression_features": int(x.shape[1]),
        "input_log2_applied": transformed,
        "standardization": "cohort-local median/MAD; SD fallback; outcome-free",
        "outcome_used_for_expression_processing": False,
        **platform_audit,
        **expression_audit,
    }
    with (output_dir / f"{accession}_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    return audit

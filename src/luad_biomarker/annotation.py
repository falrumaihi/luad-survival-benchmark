from __future__ import annotations

import csv
import gzip
import os
from pathlib import Path
import tempfile
from typing import Any


def parse_gtf_attributes(text: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for item in text.strip().rstrip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        key, separator, value = item.partition(" ")
        if not separator:
            continue
        attributes[key] = value.strip().strip('"')
    return attributes


def _selected_gene_ids(counts_path: Path) -> list[str]:
    genes: list[str] = []
    with counts_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if not header or header[0] != "gene":
            raise ValueError("Selected count matrix must have gene as its first column")
        for row_number, row in enumerate(reader, start=2):
            if not row or not row[0]:
                raise ValueError(f"Missing gene ID at selected-count row {row_number}")
            genes.append(row[0])
    if len(genes) != len(set(genes)):
        raise ValueError("Selected count matrix has duplicate exact gene IDs")
    return genes


def build_gencode_mapping(
    gtf_path: Path,
    selected_counts_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    selected_ids = _selected_gene_ids(selected_counts_path)
    selected_set = set(selected_ids)
    records: dict[str, dict[str, str]] = {}

    opener = gzip.open if gtf_path.suffix == ".gz" else open
    with opener(gtf_path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            attrs = parse_gtf_attributes(fields[8])
            gene_id = attrs.get("gene_id")
            if not gene_id:
                raise ValueError(f"GENCODE gene record lacks gene_id at line {line_number}")
            if gene_id in records:
                raise ValueError(f"Duplicate GENCODE gene record: {gene_id}")
            records[gene_id] = {
                "gene_id": gene_id,
                "gene_id_unversioned": gene_id.split(".", 1)[0],
                "gene_symbol": attrs.get("gene_name", ""),
                "gene_type": attrs.get("gene_type", attrs.get("gene_biotype", "")),
                "chromosome": fields[0],
                "start": fields[3],
                "end": fields[4],
                "strand": fields[6],
                "selected_matrix": str(gene_id in selected_set),
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    fields = [
        "gene_id",
        "gene_id_unversioned",
        "gene_symbol",
        "gene_type",
        "chromosome",
        "start",
        "end",
        "strand",
        "selected_matrix",
    ]
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for gene_id in sorted(records):
                writer.writerow(records[gene_id])
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    mapped = selected_set.intersection(records)
    missing = sorted(selected_set - records.keys())
    selected_records = [records[gene_id] for gene_id in mapped]
    gene_types: dict[str, int] = {}
    for record in selected_records:
        gene_type = record["gene_type"] or "missing"
        gene_types[gene_type] = gene_types.get(gene_type, 0) + 1
    symbol_counts: dict[str, int] = {}
    for record in selected_records:
        symbol = record["gene_symbol"]
        if symbol:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    duplicated_symbols = {symbol: n for symbol, n in symbol_counts.items() if n > 1}

    return {
        "gtf_file": gtf_path.name,
        "gtf_gene_records": len(records),
        "selected_gene_rows": len(selected_ids),
        "selected_exact_mapped": len(mapped),
        "selected_exact_missing": len(missing),
        "selected_missing_examples": missing[:50],
        "selected_gene_type_counts": dict(sorted(gene_types.items())),
        "selected_missing_symbols": sum(not record["gene_symbol"] for record in selected_records),
        "duplicated_gene_symbols": len(duplicated_symbols),
        "duplicated_gene_symbol_examples": list(sorted(duplicated_symbols.items()))[:50],
        "mapping_policy": "exact_versioned_gene_id_including_PAR_Y_suffix",
    }

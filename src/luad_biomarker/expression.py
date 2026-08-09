from __future__ import annotations

from collections import Counter
import csv
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence


ENSEMBL_GENE = re.compile(r"^ENSG\d{11}(?:\.\d+)?$")


def strip_ensembl_version(gene_id: str) -> str:
    return gene_id.split(".", 1)[0]


def extract_selected_raw_counts(
    source_path: Path,
    selected_barcodes: Sequence[str],
    output_path: Path,
) -> dict[str, Any]:
    if not selected_barcodes:
        raise ValueError("At least one selected barcode is required")
    if len(selected_barcodes) != len(set(selected_barcodes)):
        raise ValueError("Selected barcodes must be unique")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)

    exact_ids: set[str] = set()
    unversioned_counts: Counter[str] = Counter()
    special_gene_ids: list[str] = []
    versioned_ensembl = 0
    ensembl_gene_rows = 0
    gene_rows = 0
    zero_total_rows = 0
    negative_value_count = 0

    try:
        with source_path.open("r", encoding="utf-8-sig", newline="") as source, temp_path.open(
            "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.reader(source)
            writer = csv.writer(target)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"Empty count matrix: {source_path}") from exc
            if not header or header[-1].lower() != "gene":
                raise ValueError("Expected gene identifier in the last source column")
            source_samples = header[:-1]
            sample_to_index = {sample: index for index, sample in enumerate(source_samples)}
            missing = [barcode for barcode in selected_barcodes if barcode not in sample_to_index]
            if missing:
                raise ValueError(f"Selected barcodes absent from count matrix: {missing[:10]}")
            indices = [sample_to_index[barcode] for barcode in selected_barcodes]
            writer.writerow(["gene", *selected_barcodes])

            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ValueError(
                        f"Source count row {row_number} has {len(row)} columns; expected {len(header)}"
                    )
                gene_id = row[-1].strip()
                if not gene_id:
                    raise ValueError(f"Missing gene identifier at row {row_number}")
                if gene_id in exact_ids:
                    raise ValueError(f"Duplicate exact gene identifier: {gene_id}")
                exact_ids.add(gene_id)
                unversioned_counts[strip_ensembl_version(gene_id)] += 1
                gene_rows += 1

                if ENSEMBL_GENE.match(gene_id):
                    ensembl_gene_rows += 1
                    versioned_ensembl += int("." in gene_id)
                elif len(special_gene_ids) < 30:
                    special_gene_ids.append(gene_id)

                values = [row[index] for index in indices]
                numeric_values: list[int] = []
                for barcode, raw_value in zip(selected_barcodes, values):
                    try:
                        value = int(raw_value)
                    except ValueError as exc:
                        raise ValueError(
                            f"Non-integer count at row {row_number}, sample {barcode}: {raw_value!r}"
                        ) from exc
                    negative_value_count += int(value < 0)
                    numeric_values.append(value)
                zero_total_rows += int(sum(numeric_values) == 0)
                writer.writerow([gene_id, *numeric_values])

        if negative_value_count:
            raise ValueError(f"Selected count matrix contains {negative_value_count} negative values")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    duplicate_unversioned = {
        gene_id: count for gene_id, count in unversioned_counts.items() if count > 1
    }
    return {
        "source_file": source_path.name,
        "output_file": output_path.name,
        "source_samples": len(source_samples),
        "selected_samples": len(selected_barcodes),
        "gene_rows": gene_rows,
        "exact_unique_gene_ids": len(exact_ids),
        "ensembl_gene_rows": ensembl_gene_rows,
        "versioned_ensembl_gene_rows": versioned_ensembl,
        "non_ensembl_or_summary_rows": gene_rows - ensembl_gene_rows,
        "special_gene_id_examples": special_gene_ids,
        "duplicate_ids_after_version_stripping": len(duplicate_unversioned),
        "duplicate_unversioned_examples": list(sorted(duplicate_unversioned.items()))[:30],
        "zero_total_rows_across_selected_samples": zero_total_rows,
        "negative_value_count": negative_value_count,
        "orientation": "genes_by_rows_samples_by_columns_gene_first",
    }


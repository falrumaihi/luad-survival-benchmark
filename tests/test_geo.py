from __future__ import annotations

import gzip
from pathlib import Path

from luad_biomarker.geo import load_platform_map, parse_series_metadata


def test_series_characteristics_become_fields(tmp_path: Path) -> None:
    path = tmp_path / "series.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write('!Sample_geo_accession\t"GSM1"\t"GSM2"\n')
        handle.write('!Sample_characteristics_ch1\t"vital_status: Alive"\t"vital_status: Dead"\n')
        handle.write('!Sample_characteristics_ch1\t"survival_time_in_days: 10"\t"survival_time_in_days: 20"\n')
        handle.write("!series_matrix_table_begin\n")
        handle.write('"ID_REF"\t"GSM1"\t"GSM2"\n')
    metadata, skiprows = parse_series_metadata(path)
    assert skiprows == 4
    assert metadata.loc["GSM2", "vital_status"] == "Dead"
    assert metadata.loc["GSM1", "survival_time_in_days"] == "10"


def test_platform_mapping_excludes_ambiguous_symbols(tmp_path: Path) -> None:
    path = tmp_path / "platform.annot.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("^Annotation\n# fields\n")
        handle.write("ID\tGene symbol\tGene title\n")
        handle.write("probe1\tTP53\ttumor protein p53\n")
        handle.write("probe2\tEGFR /// ERBB2\tambiguous\n")
    mapping, audit = load_platform_map(path, {"TP53", "EGFR", "ERBB2"})
    assert mapping == {"probe1": "TP53"}
    assert audit["target_symbols_mapped"] == 1

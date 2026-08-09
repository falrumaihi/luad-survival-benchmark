from __future__ import annotations

import csv
import json
import math
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from .io_utils import write_json_atomic


def _bh(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _extract_panel_vst(workspace: Path, panel: list[str]) -> pd.DataFrame:
    mapping = pd.read_csv(workspace / "artifacts" / "interim" / "gencode_v36_gene_mapping.csv")
    positions: dict[int, str] = {}
    for gene in panel:
        matches = mapping.index[mapping["gene_symbol"].eq(gene)].tolist()
        if len(matches) != 1:
            raise ValueError(f"Expected one exact GENCODE row for {gene}; found {len(matches)}")
        positions[int(matches[0])] = gene
    path = workspace / "TCGA-LUAD_vst_normalized_matrix.csv"
    values: dict[str, list[float]] = {}
    with path.open("rb") as handle:
        header = next(handle).decode("utf-8-sig").rstrip("\r\n")
        header_fields = next(csv.reader([header]))
        if header_fields[-1].lower() != "gene":
            raise ValueError("Expected the final VST column to be named 'gene'")
        barcodes = header_fields[:-1]
        for row_index, raw_line in enumerate(handle):
            gene = positions.get(row_index)
            if gene is None:
                continue
            line = raw_line.decode("utf-8").rstrip("\r\n")
            parsed = next(csv.reader([line]))
            observed_gene_id = parsed[-1]
            expected_gene_id = str(mapping.iloc[row_index]["gene_id"])
            if observed_gene_id != expected_gene_id:
                raise ValueError(
                    f"VST/GENCODE row-order mismatch at {row_index}: "
                    f"observed {observed_gene_id}, expected {expected_gene_id}"
                )
            values[gene] = [float(value) for value in parsed[:-1]]
            if len(values) == len(panel):
                break
    if set(values) != set(panel):
        raise ValueError(f"Missing panel rows in VST matrix: {sorted(set(panel) - set(values))}")
    frame = pd.DataFrame(values, index=barcodes)
    frame.index.name = "barcode"
    return frame


def _paired_tumor_normal(workspace: Path, expression: pd.DataFrame, panel: list[str]) -> pd.DataFrame:
    metadata = pd.read_csv(workspace / "TCGA_TCGA-LUAD_Metadata.csv", low_memory=False)
    metadata["patient_id"] = metadata["submitter_id"].fillna(metadata["barcode"].str.slice(0, 12))
    selected = pd.read_csv(workspace / "artifacts" / "audit" / "primary_aliquot_selection.csv")
    selected = selected.loc[selected["selected"].astype(str).str.lower().eq("true")]
    primary = dict(zip(selected["patient_id"], selected["barcode"], strict=False))
    normal_rows = metadata.loc[metadata["sample_type"].eq("Solid Tissue Normal")].sort_values("barcode")
    normal = normal_rows.groupby("patient_id")["barcode"].first().to_dict()
    pairs = sorted(set(primary) & set(normal))
    records: list[dict[str, Any]] = []
    for gene in panel:
        tumor = np.asarray([expression.loc[primary[patient], gene] for patient in pairs], dtype=float)
        adjacent = np.asarray([expression.loc[normal[patient], gene] for patient in pairs], dtype=float)
        difference = tumor - adjacent
        statistic, p_value = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
        records.append(
            {
                "gene": gene,
                "paired_patients": len(pairs),
                "median_tumor_vst": float(np.median(tumor)),
                "median_normal_vst": float(np.median(adjacent)),
                "median_paired_difference": float(np.median(difference)),
                "paired_cohen_dz": float(np.mean(difference) / np.std(difference, ddof=1)) if np.std(difference, ddof=1) > 0 else math.nan,
                "wilcoxon_statistic": float(statistic),
                "p_value": float(p_value),
            }
        )
    frame = pd.DataFrame(records)
    frame["fdr_bh"] = _bh(frame["p_value"].to_numpy(float))
    frame["tumor_direction"] = np.where(frame["median_paired_difference"] > 0, "up_in_tumor", "down_in_tumor")
    return frame


def _stage_trends(workspace: Path, panel: list[str]) -> pd.DataFrame:
    cohort = pd.read_csv(workspace / "artifacts" / "interim" / "tcga_luad_patient_cohort.csv")
    expression = pd.read_csv(workspace / "artifacts" / "interim" / "tcga_luad_model_matrix.csv.gz", index_col=0)
    expression.index = expression.index.astype(str)
    cohort = cohort.set_index("selected_barcode").loc[expression.index]
    stage = cohort["stage_group"].map({"I": 1.0, "II": 2.0, "III": 3.0, "IV": 4.0})
    valid = stage.notna()
    records = []
    for gene in panel:
        rho, p_value = spearmanr(stage.loc[valid].to_numpy(float), expression.loc[valid, gene].to_numpy(float))
        records.append({"gene": gene, "stage_patients": int(valid.sum()), "spearman_rho": float(rho), "p_value": float(p_value)})
    frame = pd.DataFrame(records)
    frame["fdr_bh"] = _bh(frame["p_value"].to_numpy(float))
    return frame


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "LUAD-biomarker-study/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": "LUAD-biomarker-study/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def _functional_evidence(artifacts: Path, panel: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    biology = artifacts / "biology"
    biology.mkdir(parents=True, exist_ok=True)
    gp_raw = _post_json(
        "https://biit.cs.ut.ee/gprofiler/api/gost/profile/",
        {
            "organism": "hsapiens",
            "query": panel,
            "sources": ["GO:BP", "GO:MF", "GO:CC", "REAC", "KEGG", "WP", "CORUM"],
            "user_threshold": 0.05,
            "significance_threshold_method": "g_SCS",
            "no_evidences": False,
        },
    )
    write_json_atomic(biology / "gprofiler_response.json", gp_raw)
    pathways = pd.DataFrame(gp_raw.get("result", []))
    pathway_columns = ["source", "native", "name", "p_value", "term_size", "query_size", "intersection_size", "effective_domain_size", "intersection"]
    keep = [column for column in pathway_columns if column in pathways]
    pathways = pathways.loc[:, keep] if not pathways.empty else pd.DataFrame(columns=pathway_columns)

    string_network = _get_json(
        "https://string-db.org/api/json/network",
        {"identifiers": "\r".join(panel), "species": 9606, "required_score": 400, "caller_identity": "luad_biomarker_study"},
    )
    write_json_atomic(biology / "string_network_response.json", string_network)
    edges = pd.DataFrame(string_network)
    edge_columns = ["preferredName_A", "preferredName_B", "score", "nscore", "fscore", "pscore", "ascore", "escore", "dscore", "tscore"]
    edge_keep = [column for column in edge_columns if column in edges]
    edges = edges.loc[:, edge_keep] if not edges.empty else pd.DataFrame(columns=edge_columns)
    string_enrichment = _get_json(
        "https://string-db.org/api/json/ppi_enrichment",
        {"identifiers": "\r".join(panel), "species": 9606, "caller_identity": "luad_biomarker_study"},
    )
    write_json_atomic(biology / "string_enrichment_response.json", string_enrichment)
    summary = string_enrichment[0] if string_enrichment else {}
    return pathways, edges, summary


def run_biological_validation(artifacts: Path, workspace: Path) -> dict[str, Any]:
    panel = pd.read_csv(artifacts / "results" / "stable_gene_panel.csv")["gene"].tolist()
    expression = _extract_panel_vst(workspace, panel)
    paired = _paired_tumor_normal(workspace, expression, panel)
    stage = _stage_trends(workspace, panel)
    pathways, edges, string_summary = _functional_evidence(artifacts, panel)
    directions = pd.read_csv(artifacts / "results" / "gene_effect_directions.csv")
    external = directions.loc[~directions["cohort"].eq("TCGA-LUAD")].groupby("gene")["matches_tcga_direction"].sum()
    stability = pd.read_csv(artifacts / "results" / "stable_gene_panel.csv").set_index("gene")
    degree: dict[str, int] = {gene: 0 for gene in panel}
    for row in edges.itertuples(index=False):
        degree[row.preferredName_A] = degree.get(row.preferredName_A, 0) + 1
        degree[row.preferredName_B] = degree.get(row.preferredName_B, 0) + 1
    evidence = paired.set_index("gene").join(stage.set_index("gene"), lsuffix="_paired", rsuffix="_stage")
    evidence["selection_frequency"] = stability["selection_frequency"]
    evidence["external_direction_matches"] = external
    evidence["string_degree"] = pd.Series(degree)
    evidence["paired_tumor_normal_supported"] = evidence["fdr_bh_paired"] < 0.05
    evidence["stage_trend_supported"] = evidence["fdr_bh_stage"] < 0.05
    evidence["stable_at_0_60"] = evidence["selection_frequency"] >= 0.60
    evidence["direction_supported_2_of_3"] = evidence["external_direction_matches"] >= 2
    evidence["network_supported"] = evidence["string_degree"] > 0
    support_columns = ["paired_tumor_normal_supported", "stage_trend_supported", "stable_at_0_60", "direction_supported_2_of_3", "network_supported"]
    evidence["evidence_dimensions_supported"] = evidence[support_columns].sum(axis=1)
    evidence = evidence.reset_index().sort_values(["evidence_dimensions_supported", "selection_frequency"], ascending=[False, False])

    biology = artifacts / "biology"
    expression.to_csv(biology / "panel_expression_all_tcga_aliquots.csv.gz")
    paired.to_csv(biology / "paired_tumor_normal_results.csv", index=False)
    stage.to_csv(biology / "stage_trend_results.csv", index=False)
    pathways.to_csv(biology / "pathway_enrichment.csv", index=False)
    edges.to_csv(biology / "string_network_edges.csv", index=False)
    evidence.to_csv(biology / "biological_gene_evidence.csv", index=False)
    summary = {
        "panel_genes": panel,
        "paired_patients": int(paired["paired_patients"].iloc[0]),
        "paired_fdr_significant_genes": paired.loc[paired["fdr_bh"] < 0.05, "gene"].tolist(),
        "stage_fdr_significant_genes": stage.loc[stage["fdr_bh"] < 0.05, "gene"].tolist(),
        "significant_functional_terms": int(len(pathways)),
        "string_edges": int(len(edges)),
        "string_ppi_enrichment": string_summary,
        "interpretation": "post-selection biological plausibility only; does not modify the locked panel or failed predictive publication gate",
    }
    write_json_atomic(artifacts / "audit" / "biological_validation_audit.json", summary)
    return summary

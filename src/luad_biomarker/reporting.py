from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import write_json_atomic


DISPLAY_MODEL = {
    "Clinical Cox": "Age/sex Cox",
    "Random survival forest": "Random survival forest",
    "Gradient boosting survival": "Gradient boosting survival",
}


# Supplementary tables are written straight from the analysis outputs, numbered
# in the order they are first cited in the manuscript.
SUPPLEMENTARY_TABLE_SOURCES = {
    "Table_S1_gene_directions": "results/gene_effect_directions.csv",
    "Table_S2_internal_cv_descriptive": "results/internal_cv_metrics.csv",
    "Table_S3_paired_bootstrap": "results/paired_bootstrap_comparisons.csv",
    "Table_S4_calibration": "results/csd_calibration_metrics.csv",
    "Table_S5_extended_discrimination": "results/extended_discrimination_metrics.csv",
    "Table_S6_paired_tumor_normal": "biology/paired_tumor_normal_results.csv",
    "Table_S7_stage_trends": "biology/stage_trend_results.csv",
    "Table_S8_pathway_enrichment": "biology/pathway_enrichment.csv",
    "Table_S9_string_network": "biology/string_network_edges.csv",
    "Table_S10_post_lock_models": "results/supplementary_models_metrics.csv",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(fields) + " |"
    rule = "| " + " | ".join("---" for _ in fields) + " |"
    body = ["| " + " | ".join(clean(row.get(field, "")) for field in fields) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def _cohort_summary(artifacts: Path) -> list[dict[str, Any]]:
    tcga_audit = _read_json(artifacts / "audit" / "cohort_audit.json")
    rows = [
        {
            "Cohort": "TCGA-LUAD",
            "Role": "Discovery/training",
            "Platform": "RNA-seq (STAR counts)",
            "Patients": tcga_audit["eligible_os_patients"],
            "Deaths": tcga_audit["eligible_os_events"],
            "Notes": "Primary tumors; patient-level aliquot selection",
        }
    ]
    metrics = pd.read_csv(artifacts / "results" / "external_validation_metrics.csv")
    platforms = {
        "GSE72094": "GPL15048 microarray",
        "GSE68465": "GPL96 microarray",
        "GSE31210": "GPL570 microarray",
    }
    roles = {
        "GSE72094": "Primary external",
        "GSE68465": "Primary external",
        "GSE31210": "Early-stage stress test",
    }
    notes = {
        "GSE72094": "Resected LUAD; 398 OS-eligible",
        "GSE68465": "Multi-site LUAD validation cohort",
        "GSE31210": "Stage I-II LUAD",
    }
    for cohort in ("GSE72094", "GSE68465", "GSE31210"):
        first = metrics.loc[metrics["cohort"].eq(cohort)].iloc[0]
        rows.append(
            {
                "Cohort": cohort,
                "Role": roles[cohort],
                "Platform": platforms[cohort],
                "Patients": int(first["n"]),
                "Deaths": int(first["events"]),
                "Notes": notes[cohort],
            }
        )
    return rows


def _external_table(artifacts: Path) -> list[dict[str, Any]]:
    metrics = pd.read_csv(artifacts / "results" / "external_validation_metrics.csv")
    order = [
        "SurvivalPFN",
        "Panel Cox",
        "Random survival forest",
        "Gradient boosting survival",
        "DeepSurv",
        "Clinical Cox",
    ]
    metrics["model_order"] = metrics["model"].map({name: i for i, name in enumerate(order)})
    metrics = metrics.sort_values(["cohort", "model_order"])
    rows: list[dict[str, Any]] = []
    for row in metrics.itertuples(index=False):
        rows.append(
            {
                "Cohort": row.cohort,
                "Model": DISPLAY_MODEL.get(row.model, row.model),
                "C-index (95% CI)": f"{row.c_index:.3f} ({row.ci_lower:.3f}-{row.ci_upper:.3f})",
                "n/events": f"{int(row.n)}/{int(row.events)}",
            }
        )
    return rows


def _panel_table(artifacts: Path) -> list[dict[str, Any]]:
    panel = pd.read_csv(artifacts / "results" / "stable_gene_panel.csv")
    effects = pd.read_csv(artifacts / "results" / "gene_effect_directions.csv")
    rows: list[dict[str, Any]] = []
    for item in panel.itertuples(index=False):
        e = effects.loc[effects["gene"].eq(item.gene)]
        tcga_direction = e.loc[e["cohort"].eq("TCGA-LUAD"), "direction"].iloc[0]
        external = e.loc[~e["cohort"].eq("TCGA-LUAD"), "matches_tcga_direction"]
        rows.append(
            {
                "Rank": int(item.panel_rank),
                "Gene": item.gene,
                "Selection frequency": f"{item.selection_frequency:.3f}",
                "TCGA direction": str(tcga_direction).replace("_", " "),
                "External direction matches": f"{int(external.sum())}/3",
                "Passes 0.60": "Yes" if item.selection_frequency >= 0.60 else "No",
            }
        )
    return rows


def _gate_table(artifacts: Path) -> list[dict[str, Any]]:
    gate = _read_json(artifacts / "results" / "publication_gate.json")
    labels = {
        "1_pfn_beats_clinical_both_primary": "SurvivalPFN point estimate exceeds age/sex Cox in both primary cohorts",
        "2_paired_ci_excludes_zero_one_primary": "Paired bootstrap delta-C CI excludes zero in at least one primary cohort",
        "3_csd_ibs_preserved_both_primary": "CSD preserves or improves IBS in both primary cohorts",
        "4_gene_direction_concordance_at_least_75pct": "Primary-cohort gene-direction concordance is at least 75%",
        "5_all_panel_genes_stability_at_least_0_60": "Every panel gene has selection frequency at least 0.60",
        "6_frozen_early_stage_scored": "Frozen model is scored in the early-stage cohort",
    }
    return [
        {"Criterion": labels[key], "Result": "Pass" if value else "Fail"}
        for key, value in gate["conditions"].items()
    ]


def _biological_table(artifacts: Path) -> list[dict[str, Any]]:
    evidence = pd.read_csv(artifacts / "biology" / "biological_gene_evidence.csv")
    evidence = evidence.sort_values("selection_frequency", ascending=False)
    rows: list[dict[str, Any]] = []
    for row in evidence.itertuples(index=False):
        rows.append(
            {
                "Gene": row.gene,
                "Paired dz": f"{row.paired_cohen_dz:.2f}",
                "Tumor-normal FDR": f"{row.fdr_bh_paired:.3g}",
                "Stage rho": f"{row.spearman_rho:.2f}",
                "Stage FDR": f"{row.fdr_bh_stage:.3g}",
                "External direction matches": f"{int(row.external_direction_matches)}/3",
                "Stability": f"{row.selection_frequency:.3f}",
            }
        )
    return rows


def _write_table_bundle(artifacts: Path, manuscript: Path) -> dict[str, str]:
    table_dir = artifacts / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "Table_1_cohort_summary": (_cohort_summary(artifacts), ["Cohort", "Role", "Platform", "Patients", "Deaths", "Notes"]),
        "Table_2_panel_stability": (_panel_table(artifacts), ["Rank", "Gene", "Selection frequency", "TCGA direction", "External direction matches", "Passes 0.60"]),
        "Table_3_locked_external_validation": (_external_table(artifacts), ["Cohort", "Model", "C-index (95% CI)", "n/events"]),
        "Table_4_biological_validation": (
            _biological_table(artifacts),
            ["Gene", "Paired dz", "Tumor-normal FDR", "Stage rho", "Stage FDR", "External direction matches", "Stability"],
        ),
        "Table_5_publication_gate": (_gate_table(artifacts), ["Criterion", "Result"]),
        "Table_6_translational_readiness": (
            [
                {
                    "Domain": "Intended use",
                    "Evidence in this study": "Research-stage postoperative overall-survival risk estimation in LUAD",
                    "Management implication": "May motivate future adjunctive risk-stratification studies; not a treatment-selection test",
                    "Status": "Defined, not clinically validated",
                },
                {
                    "Domain": "Analytical transport",
                    "Evidence in this study": "Locked RNA-seq panel scored on three independent microarray platforms",
                    "Management implication": "Supports technical portability of the measured features",
                    "Status": "Supported retrospectively",
                },
                {
                    "Domain": "Biological association",
                    "Evidence in this study": "11/12 genes differed in 58 matched tumor-normal pairs",
                    "Management implication": "Links most panel genes to LUAD tissue state",
                    "Status": "Supported; post-selection",
                },
                {
                    "Domain": "Prognostic discrimination",
                    "Evidence in this study": "Three external cohorts; SurvivalPFN C-index 0.583-0.665",
                    "Management implication": "Shows modest risk-ranking ability across cohorts",
                    "Status": "Externally evaluated",
                },
                {
                    "Domain": "Incremental value",
                    "Evidence in this study": "No paired SurvivalPFN-versus-panel Cox delta-C interval excluded zero",
                    "Management implication": "No demonstrated advantage over the conventional molecular comparator",
                    "Status": "Not established",
                },
                {
                    "Domain": "Calibration",
                    "Evidence in this study": "CSD improved one cohort and worsened two by IBS",
                    "Management implication": "Absolute risk estimates require cohort-specific validation",
                    "Status": "Inconsistent",
                },
                {
                    "Domain": "Mechanistic coherence",
                    "Evidence in this study": "No significant pathway term or STRING interaction",
                    "Management implication": "Panel should not be presented as a unified biological module",
                    "Status": "Not supported",
                },
                {
                    "Domain": "Clinical utility",
                    "Evidence in this study": "No prospective decision-impact, treatment-response, or cost-effectiveness study",
                    "Management implication": "Cannot guide adjuvant therapy or surveillance",
                    "Status": "Not evaluated",
                },
            ],
            ["Domain", "Evidence in this study", "Management implication", "Status"],
        ),
    }
    markdown: dict[str, str] = {}
    for name, (rows, fields) in tables.items():
        _write_csv(table_dir / f"{name}.csv", rows, fields)
        md = _markdown_table(rows, fields)
        (table_dir / f"{name}.md").write_text(md + "\n", encoding="utf-8")
        markdown[name] = md

    for name, relative in SUPPLEMENTARY_TABLE_SOURCES.items():
        source = artifacts / relative
        try:
            frame = pd.read_csv(source)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame({"result": ["No significant entries"]})
        frame.to_csv(table_dir / f"{name}.csv", index=False)
    return markdown


def _manuscript_compliance(manuscript: Path) -> dict[str, Any] | None:
    """Summarise the assembled manuscript, when one is present.

    Manuscript documents are authored and assembled outside this package. They
    are optional: the analysis pipeline is usable on its own, so this returns
    ``None`` when no assembled manuscript is present rather than failing the run.
    """
    assembled = manuscript / "manuscript.md"
    supplementary = manuscript / "supplementary_source.md"
    if not assembled.exists():
        return None

    text = assembled.read_text(encoding="utf-8")
    if "## References" not in text or "## Simple Summary" not in text:
        return None

    simple_summary = text.split("## Simple Summary", 1)[1].split("## Abstract", 1)[0]
    main_body, references = text.split("## References", 1)
    supplementary_figures = (
        supplementary.read_text(encoding="utf-8").count("![Supplementary Figure ")
        if supplementary.exists()
        else 0
    )
    compliance = {
        "title": text.splitlines()[0].lstrip("# "),
        "simple_summary_words": len(simple_summary.split()),
        "main_text_words_excluding_references": len(main_body.split()),
        "main_figure_count": text.count("![Figure "),
        "supplementary_figure_count": supplementary_figures,
        "reference_count": sum(
            1 for line in references.splitlines() if line[:1].isdigit() and ". " in line
        ),
    }
    compliance["passes_automated_checks"] = bool(
        compliance["simple_summary_words"] <= 200
        and compliance["main_figure_count"] == 7
        and compliance["supplementary_figure_count"] == 6
        and 55 <= compliance["reference_count"] <= 60
    )
    return compliance


def _artifact_manifest(workspace: Path, artifacts: Path, manuscript: Path) -> Path:
    roots = [artifacts / "audit", artifacts / "figures", artifacts / "lock", artifacts / "results", artifacts / "tables", manuscript]
    rows: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            transient = any(part.startswith(("qa_", "lo_profile", "lo_direct")) for part in path.parts)
            if not path.is_file() or transient:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({"path": path.relative_to(workspace).as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    out = artifacts / "manifests" / "final_artifact_manifest.csv"
    _write_csv(out, rows, ["path", "bytes", "sha256"])
    return out


def generate_reporting_package(artifacts: Path, workspace: Path | None = None) -> dict[str, Any]:
    """Write the table bundle and the artifact manifest, then audit both.

    Manuscript documents are optional. When an assembled manuscript is present
    its structure is summarised in the audit; otherwise the analysis outputs are
    produced on their own.
    """
    workspace = (workspace or artifacts.parent).resolve()
    manuscript = workspace / "manuscript"
    manuscript.mkdir(parents=True, exist_ok=True)
    tables = _write_table_bundle(artifacts, manuscript)
    manifest = _artifact_manifest(workspace, artifacts, manuscript)
    report: dict[str, Any] = {
        "table_count": len(tables) + len(SUPPLEMENTARY_TABLE_SOURCES),
        "manifest": str(manifest),
        "publication_gate_passed": _read_json(
            artifacts / "results" / "publication_gate.json"
        )["gate_passed"],
    }
    compliance = _manuscript_compliance(manuscript)
    if compliance is not None:
        report["manuscript_compliance"] = compliance
    write_json_atomic(artifacts / "audit" / "reporting_audit.json", report)
    return report

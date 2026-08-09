from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .figure_style import (
    C_CLINICAL,
    C_CONFORMAL,
    C_GRID,
    C_INK,
    C_SURVIVALPFN,
    COHORT_COLORS,
    MODEL_COLORS,
    NPG,
    apply_style,
    panel_label,
    save,
)


MODEL_COLORS.update(
    {
        "Panel Cox": "#3C5488",
        "Gradient boosting survival": "#F39B7F",
        "SurvivalPFN-split": C_SURVIVALPFN,
        "CSD-SurvivalPFN": C_CONFORMAL,
        "XGBoost Cox": "#F39B7F",
        "SurvivalPFN + age/sex": "#DC0000",
        "Stage-only Cox": "#91D1C2",
    }
)


def _figure_1(artifacts: Path) -> list[dict[str, str | int]]:
    audit = json.loads((artifacts / "audit" / "cohort_audit.json").read_text())
    external = [
        json.loads((artifacts / "external" / f"{cohort}_audit.json").read_text())
        for cohort in ("GSE72094", "GSE68465", "GSE31210")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.axis("off")
    steps = [
        ("601 RNA aliquots", "raw STAR counts + metadata"),
        ("517 primary-tumor patients", "12 duplicate-primary cases resolved"),
        ("504 OS-eligible patients", "182 deaths; patient-level discovery"),
        ("12-gene locked panel", "200 parallel bootstrap selections"),
        ("3 external cohorts", "1,066 patients; 384 deaths"),
    ]
    y_positions = np.linspace(0.92, 0.08, len(steps))
    for index, ((title, subtitle), y) in enumerate(zip(steps, y_positions, strict=True)):
        ax.text(
            0.5,
            y,
            f"{title}\n{subtitle}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.42", "facecolor": NPG[index], "alpha": 0.18, "edgecolor": NPG[index], "linewidth": 1.4},
        )
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, y_positions[index + 1] + 0.07),
                xytext=(0.5, y - 0.07),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "-|>", "color": C_INK, "lw": 1.2},
            )
    ax.set_title("Leakage-controlled study flow", pad=8)
    panel_label(ax, "a", dx=-0.05, dy=1.03)

    ax = axes[1]
    cohorts = ["TCGA-LUAD", *[item["accession"] for item in external]]
    sample_counts = [audit["eligible_os_patients"], *[item["eligible_os_samples"] for item in external]]
    event_counts = [audit["eligible_os_events"], *[item["events"] for item in external]]
    positions = np.arange(len(cohorts))
    ax.bar(positions, sample_counts, color=[COHORT_COLORS[c] for c in cohorts], alpha=0.88, label="Censored")
    ax.bar(positions, event_counts, color=C_SURVIVALPFN, alpha=0.85, label="Deaths")
    for position, total, events in zip(positions, sample_counts, event_counts, strict=True):
        ax.text(position, total + 13, f"n={total}\n{events} events", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(positions, cohorts, rotation=28, ha="right")
    ax.set_ylabel("Patients")
    ax.set_ylim(0, max(sample_counts) * 1.22)
    ax.set_title("Discovery and validation cohorts")
    ax.legend(loc="upper right")
    panel_label(ax, "b")
    fig.tight_layout(w_pad=2.0)
    return save(fig, artifacts / "figures", "Figure_1_study_design")


def _figure_2(artifacts: Path) -> list[dict[str, str | int]]:
    panel = pd.read_csv(artifacts / "results" / "stable_gene_panel.csv").sort_values("selection_frequency")
    effects = pd.read_csv(artifacts / "results" / "gene_effect_directions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.1), gridspec_kw={"width_ratios": [0.9, 1.25]})
    ax = axes[0]
    colors = [C_SURVIVALPFN if value >= 0.60 else C_CLINICAL for value in panel["selection_frequency"]]
    ax.barh(panel["gene"], panel["selection_frequency"], color=colors)
    ax.axvline(0.60, color=C_INK, linestyle="--", linewidth=1.1, label="Prespecified 0.60")
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Bootstrap selection frequency")
    ax.set_title("Six genes met the stability threshold")
    ax.legend(loc="lower right")
    panel_label(ax, "a")

    ax = axes[1]
    matrix = effects.pivot(index="gene", columns="cohort", values="cox_coefficient").loc[
        panel.sort_values("selection_frequency", ascending=False)["gene"],
        ["TCGA-LUAD", "GSE72094", "GSE68465", "GSE31210"],
    ]
    maximum = float(np.quantile(np.abs(matrix.to_numpy()), 0.95))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=sns.diverging_palette(220, 15, as_cmap=True),
        center=0,
        vmin=-maximum,
        vmax=maximum,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Univariable Cox coefficient", "shrink": 0.75},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Effect directions were mostly transportable")
    ax.tick_params(axis="x", rotation=30)
    panel_label(ax, "b", dx=-0.22)
    fig.tight_layout(w_pad=2.0)
    return save(fig, artifacts / "figures", "Figure_2_panel_stability")


def _figure_3(artifacts: Path) -> list[dict[str, str | int]]:
    metrics = pd.read_csv(artifacts / "results" / "external_validation_metrics.csv")
    model_order = [
        "SurvivalPFN",
        "Panel Cox",
        "Random survival forest",
        "Gradient boosting survival",
        "DeepSurv",
        "Clinical Cox",
    ]
    display_order = [
        "SurvivalPFN",
        "Panel Cox",
        "Random survival forest",
        "Gradient boosting survival",
        "DeepSurv",
        "Age/sex Cox",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.55), sharey=True)
    for letter, cohort, ax in zip("abc", ("GSE72094", "GSE68465", "GSE31210"), axes, strict=True):
        frame = metrics[metrics["cohort"].eq(cohort)].set_index("model").loc[model_order].reset_index()
        y = np.arange(len(frame))[::-1]
        for yi, row in zip(y, frame.itertuples(), strict=True):
            ax.errorbar(
                row.c_index,
                yi,
                xerr=[[row.c_index - row.ci_lower], [row.ci_upper - row.c_index]],
                fmt="o",
                color=MODEL_COLORS.get(row.model, C_INK),
                ecolor=MODEL_COLORS.get(row.model, C_INK),
                capsize=2.2,
                markersize=5,
                linewidth=1.3,
            )
        ax.axvline(0.5, color=C_INK, linestyle=":", linewidth=1)
        ax.set_xlim(0.43, 0.79)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(display_order)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("Harrell C-index (95% CI)")
        ax.set_title(f"{cohort}\nn={frame['n'].iloc[0]}, events={frame['events'].iloc[0]}")
        panel_label(ax, letter, dx=-0.24 if ax is axes[0] else -0.14)
    fig.suptitle("SurvivalPFN ranked first, but most model CIs overlapped", y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout(w_pad=1.0)
    return save(fig, artifacts / "figures", "Figure_3_external_validation")


def _figure_4(artifacts: Path) -> list[dict[str, str | int]]:
    """Paired uncertainty and calibration transport in a single non-redundant figure."""
    paired = pd.read_csv(artifacts / "results" / "paired_bootstrap_comparisons.csv")
    csd = pd.read_csv(artifacts / "results" / "csd_calibration_metrics.csv")
    order = ["GSE72094", "GSE68465", "GSE31210"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1))

    ax = axes[0, 0]
    paired = paired.copy()
    paired["label"] = paired["cohort"] + "\nvs " + paired["comparator"].str.replace("Clinical Cox", "age/sex Cox")
    y = np.arange(len(paired))[::-1]
    for yi, row in zip(y, paired.itertuples(), strict=True):
        ax.errorbar(
            row.delta_harrell_c,
            yi,
            xerr=[[row.delta_harrell_c - row.ci_lower], [row.ci_upper - row.delta_harrell_c]],
            fmt="o",
            color=COHORT_COLORS[row.cohort],
            capsize=2,
            linewidth=1.2,
        )
    ax.axvline(0, color=C_INK, linestyle="--", linewidth=1)
    ax.set_yticks(y, paired["label"], fontsize=7.3)
    ax.set_xlabel("Paired ΔC (SurvivalPFN − comparator)")
    ax.set_title("Shared-patient paired differences", fontsize=9)
    panel_label(ax, "a", dx=-0.34)

    ibs = csd.pivot(index="cohort", columns="model", values="integrated_brier_score").loc[order]
    dcal = csd.pivot(index="cohort", columns="model", values="d_calibration_p").loc[order]
    pos = np.arange(len(order))
    width = 0.34

    ax = axes[0, 1]
    ax.bar(pos - width / 2, ibs["SurvivalPFN-split"], width, color=C_SURVIVALPFN, label="Raw")
    ax.bar(pos + width / 2, ibs["CSD-SurvivalPFN"], width, color=C_CONFORMAL, label="CSD")
    ax.set_xticks(pos, order, rotation=28, ha="right")
    ax.set_ylabel("Integrated Brier score")
    ax.set_title("Overall prediction error", fontsize=9)
    ax.legend()
    panel_label(ax, "b", dx=-0.20)

    ax = axes[1, 0]
    ax.plot(pos, dcal["SurvivalPFN-split"], "o-", color=C_SURVIVALPFN, label="Raw")
    ax.plot(pos, dcal["CSD-SurvivalPFN"], "o-", color=C_CONFORMAL, label="CSD")
    ax.axhline(0.05, color=C_INK, linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(pos, order, rotation=28, ha="right")
    ax.set_ylabel("D-calibration p-value")
    ax.set_title("Distribution calibration", fontsize=9)
    ax.legend(loc="lower left", fontsize=7)
    panel_label(ax, "c", dx=-0.20)

    delta = ibs["CSD-SurvivalPFN"] - ibs["SurvivalPFN-split"]
    ax = axes[1, 1]
    reverse = order[::-1]
    colors = [C_CONFORMAL if delta.loc[cohort] <= 0 else C_CLINICAL for cohort in reverse]
    ax.barh(reverse, delta.loc[reverse], color=colors)
    ax.axvline(0, color=C_INK, linestyle="--", linewidth=1)
    ax.set_xlim(-0.0045, 0.019)
    for yi, cohort in enumerate(reverse):
        value = delta.loc[cohort]
        label_x = value + 0.0005 if value >= 0 else 0.0005
        ax.text(label_x, yi, f"{value:+.3f}", ha="left", va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("CSD minus raw IBS")
    ax.set_title("Calibration benefit was cohort-specific", fontsize=9)
    panel_label(ax, "d", dx=-0.24)

    fig.tight_layout(w_pad=1.6, h_pad=2.0)
    return save(fig, artifacts / "figures", "Figure_4_uncertainty_and_calibration")


def _supplementary_figure_6_post_lock(artifacts: Path) -> list[dict[str, str | int]]:
    supplementary = pd.read_csv(artifacts / "results" / "supplementary_models_metrics.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.7))
    frame = supplementary[supplementary["model"].isin(["XGBoost Cox", "SurvivalPFN + age/sex", "Stage-only Cox"])]
    labels = frame["cohort"] + "\n" + frame["model"].str.replace("SurvivalPFN + age/sex", "PFN + age/sex")
    y = np.arange(len(frame))[::-1]
    for yi, (row, label) in enumerate(zip(frame.itertuples(), labels, strict=True)):
        position = y[yi]
        ax.errorbar(
            row.c_index,
            position,
            xerr=[[row.c_index - row.ci_lower], [row.ci_upper - row.c_index]],
            fmt="o",
            color=MODEL_COLORS.get(row.model, C_INK),
            capsize=2,
            linewidth=1.2,
        )
    ax.axvline(0.5, color=C_INK, linestyle=":", linewidth=1)
    ax.set_yticks(y, labels, fontsize=7.2)
    ax.set_xlim(0.43, 0.79)
    ax.set_xlabel("External C-index (95% CI)")
    ax.set_title("Post-lock sensitivity models", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return save(fig, artifacts / "figures", "Supplementary_Figure_S6_post_lock_models")


def _figure_6_discrimination(artifacts: Path) -> list[dict[str, str | int]]:
    metrics = pd.read_csv(artifacts / "results" / "extended_discrimination_metrics.csv")
    metrics["model"] = metrics["model"].replace({"Clinical Cox": "Age/sex Cox"})
    model_order = ["SurvivalPFN", "Panel Cox", "Random survival forest", "Gradient boosting survival", "DeepSurv", "Age/sex Cox"]
    columns = ["uno_c_index", "auc_1y", "auc_3y", "auc_5y"]
    labels = ["Uno C", "AUC 1 y", "AUC 3 y", "AUC 5 y"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.25), sharey=True)
    for letter, cohort, ax in zip("abc", ["GSE72094", "GSE68465", "GSE31210"], axes, strict=True):
        frame = metrics.loc[metrics["cohort"].eq(cohort)].set_index("model").loc[model_order, columns]
        frame.columns = labels
        annotations = frame.map(lambda value: "NE" if pd.isna(value) else f"{value:.3f}")
        plot_frame = frame.fillna(0.49)
        cmap = sns.light_palette(C_SURVIVALPFN, as_cmap=True)
        cmap.set_under("#D9D9D9")
        sns.heatmap(
            plot_frame,
            ax=ax,
            cmap=cmap,
            vmin=0.5,
            vmax=0.72,
            annot=annotations,
            fmt="",
            annot_kws={"fontsize": 7, "fontweight": "bold"},
            linewidths=0.5,
            linecolor="white",
            cbar=(ax is axes[-1]),
            cbar_kws={"label": "Discrimination", "shrink": 0.7},
        )
        ax.set_title(cohort)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=35)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)
        panel_label(ax, letter, dx=-0.22)
    fig.suptitle("Method ranking varied across time horizons and cohorts", y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout(w_pad=0.8)
    return save(fig, artifacts / "figures", "Figure_5_extended_discrimination")


def _supplementary_figure_4_directions(artifacts: Path) -> list[dict[str, str | int]]:
    effects = pd.read_csv(artifacts / "results" / "gene_effect_directions.csv")
    wide = effects.pivot(index="gene", columns="cohort", values="cox_coefficient")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), sharex=True, sharey=True)
    for letter, cohort, ax in zip("abc", ["GSE72094", "GSE68465", "GSE31210"], axes, strict=True):
        x = wide["TCGA-LUAD"]
        y = wide[cohort]
        matched = np.sign(x) == np.sign(y)
        ax.scatter(x[matched], y[matched], color=C_SURVIVALPFN, s=32, label="Same direction", zorder=3)
        ax.scatter(x[~matched], y[~matched], color=C_CLINICAL, s=32, label="Opposite direction", zorder=3)
        ax.axhline(0, color=C_GRID, linewidth=1)
        ax.axvline(0, color=C_GRID, linewidth=1)
        for gene in wide.index:
            ax.annotate(gene, (x.loc[gene], y.loc[gene]), xytext=(2, 2), textcoords="offset points", fontsize=5.8)
        ax.set_title(f"{cohort}\n{matched.mean():.0%} direction agreement", fontsize=9)
        ax.set_xlabel("TCGA Cox coefficient")
        if ax is axes[0]:
            ax.set_ylabel("External Cox coefficient")
            ax.legend(loc="lower right", fontsize=6.5)
        panel_label(ax, letter, dx=-0.26, dy=1.02)
    fig.suptitle("Gene-level direction transport was strong but not universal", y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout(w_pad=1.1)
    return save(fig, artifacts / "figures", "Supplementary_Figure_S4_gene_direction_transport")


def _figure_7_biology(artifacts: Path) -> list[dict[str, str | int]]:
    evidence = pd.read_csv(artifacts / "biology" / "biological_gene_evidence.csv")
    evidence = evidence.sort_values("selection_frequency", ascending=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.15), gridspec_kw={"width_ratios": [1, 0.85, 1.25]})

    ax = axes[0]
    colors = [C_SURVIVALPFN if value > 0 else C_CLINICAL for value in evidence["paired_cohen_dz"]]
    ax.barh(evidence["gene"], evidence["paired_cohen_dz"], color=colors)
    ax.axvline(0, color=C_INK, linewidth=1)
    for yi, row in enumerate(evidence.itertuples(index=False)):
        if row.fdr_bh_paired < 0.05:
            ax.text(row.paired_cohen_dz + (0.04 if row.paired_cohen_dz >= 0 else -0.04), yi, "*",
                    ha="left" if row.paired_cohen_dz >= 0 else "right", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Paired tumor-normal effect (Cohen dz)")
    ax.set_title("11/12 genes differed in 58 pairs", fontsize=9)
    panel_label(ax, "a", dx=-0.28, dy=1.02)

    ax = axes[1]
    stage_colors = [C_SURVIVALPFN if q < 0.05 else C_GRID for q in evidence["fdr_bh_stage"]]
    ax.barh(evidence["gene"], evidence["spearman_rho"], color=stage_colors)
    ax.axvline(0, color=C_INK, linewidth=1)
    ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("Stage Spearman rho")
    ax.set_title("Only FAIM2 passed stage FDR", fontsize=9)
    panel_label(ax, "b", dx=-0.30, dy=1.02)

    ax = axes[2]
    # Panels a and b are barh plots, which place the first row at the BOTTOM; seaborn
    # heatmap places the first row at the TOP. Reverse the index so the evidence matrix
    # rows line up with the gene labels shown on panel a.
    matrix = evidence.set_index("gene")[[
        "paired_tumor_normal_supported",
        "stage_trend_supported",
        "stable_at_0_60",
        "direction_supported_2_of_3",
        "network_supported",
    ]].astype(int).iloc[::-1]
    matrix.columns = ["Tumor-normal", "Stage", "Stable", "Direction", "PPI"]
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=sns.color_palette(["#E5E5E5", C_CONFORMAL], as_cmap=True),
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.6,
        linecolor="white",
        annot=matrix.replace({0: "-", 1: "+"}),
        fmt="",
        annot_kws={"fontsize": 8, "fontweight": "bold"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0, labelsize=7.5)
    ax.set_title("Independent evidence dimensions", fontsize=9)
    panel_label(ax, "c", dx=-0.34, dy=1.02)
    fig.suptitle("Matched-tissue evidence was strong; the panel is not one pathway", y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout(w_pad=1.3)
    return save(fig, artifacts / "figures", "Figure_6_biological_validation")


def _figure_8_gate(artifacts: Path) -> list[dict[str, str | int]]:
    gate = json.loads((artifacts / "results" / "publication_gate.json").read_text())
    labels = [
        "PFN > age/sex\nin both primary cohorts",
        "Paired CI excludes 0\nin >=1 primary cohort",
        "CSD preserves IBS\nin both primary cohorts",
        "Gene directions\n>=75% concordant",
        "All genes stable\nat >=0.60",
        "Frozen early-stage\ncohort scored",
    ]
    values = list(gate["conditions"].values())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    y = np.arange(len(values))[::-1]
    colors = [C_CONFORMAL if value else C_CLINICAL for value in values]
    ax.barh(y, [1] * len(values), color=colors, height=0.72)
    for idx, value in enumerate(values):
        ax.text(0.5, y[idx], "PASS" if value else "FAIL", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.set_yticks(y, labels, fontsize=7.2)
    ax.set_xlim(0, 1.02)
    ax.set_xticks([])
    ax.set_title("Six prespecified criteria, applied conjunctively")
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "a", dx=-0.42, dy=1.02)

    ax = axes[1]
    ax.axis("off")
    ax.text(0.5, 0.90, "EVIDENCE GRADE", ha="center", va="center", fontsize=10, fontweight="bold", color="#3C5488")
    ax.text(0.5, 0.69, "4 of 6 met", ha="center", va="center", fontsize=15, fontweight="bold", color="#3C5488",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#3C5488", "alpha": 0.12, "edgecolor": "#3C5488", "linewidth": 1.4})
    ax.annotate("", xy=(0.5, 0.45), xytext=(0.5, 0.59), xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "-|>", "color": C_INK, "lw": 1.3})
    ax.text(0.5, 0.27, "Externally validated\ncomparative benchmark\n\nRisk ranking transports;\nabsolute risk needs\nlocal recalibration",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#E8EDF4", "edgecolor": "#3C5488", "linewidth": 1.4})
    panel_label(ax, "b", dx=-0.05)
    fig.tight_layout(w_pad=1.2)
    return save(fig, artifacts / "figures", "Figure_7_evidence_synthesis")


def _kaplan_meier(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(time)
    time = time[order]
    event = event[order]
    unique_event_times = np.unique(time[event.astype(bool)])
    survival = 1.0
    xs = [0.0]
    ys = [1.0]
    for value in unique_event_times:
        at_risk = np.sum(time >= value)
        deaths = np.sum((time == value) & event.astype(bool))
        if at_risk > 0:
            survival *= 1.0 - deaths / at_risk
        xs.append(float(value) / 365.25)
        ys.append(float(survival))
    return np.asarray(xs), np.asarray(ys)


def _supplementary_figure_1_cohort_survival(artifacts: Path) -> list[dict[str, str | int]]:
    cohorts = {
        "TCGA-LUAD": pd.read_csv(artifacts / "interim" / "tcga_luad_patient_cohort.csv"),
        "GSE72094": pd.read_csv(artifacts / "external" / "GSE72094_phenotype.csv"),
        "GSE68465": pd.read_csv(artifacts / "external" / "GSE68465_phenotype.csv"),
        "GSE31210": pd.read_csv(artifacts / "external" / "GSE31210_phenotype.csv"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=False, sharey=True)
    for letter, (name, frame), ax in zip("abcd", cohorts.items(), axes.ravel(), strict=True):
        time = pd.to_numeric(frame["os_days"], errors="coerce").to_numpy(float)
        event = pd.to_numeric(frame["os_event"], errors="coerce").to_numpy(int)
        valid = np.isfinite(time) & (time > 0)
        x, y = _kaplan_meier(time[valid], event[valid])
        ax.step(x, y, where="post", color=COHORT_COLORS[name], linewidth=2)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Follow-up (years)")
        ax.set_ylabel("Overall survival probability")
        ax.set_title(f"{name}: n={valid.sum()}, events={event[valid].sum()}")
        panel_label(ax, letter, dx=-0.15)
    fig.suptitle("Observed survival and censoring support differed across cohorts", y=1.01, fontsize=11, fontweight="bold")
    fig.tight_layout(w_pad=1.4, h_pad=1.5)
    return save(fig, artifacts / "figures", "Supplementary_Figure_S1_cohort_survival")


def _supplementary_figure_2_preprocessing(artifacts: Path) -> list[dict[str, str | int]]:
    normalization = json.loads((artifacts / "audit" / "normalization_audit.json").read_text())
    model_matrix = json.loads((artifacts / "audit" / "model_matrix_audit.json").read_text())
    external = [
        json.loads((artifacts / "external" / f"{cohort}_audit.json").read_text())
        for cohort in ("GSE72094", "GSE68465", "GSE31210")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3))
    ax = axes[0]
    labels = ["Input", "Count-filtered", "Protein coding", "Model features"]
    values = [normalization["input_gene_rows"], normalization["retained_gene_rows"], model_matrix["unique_protein_coding_symbols"], model_matrix["output_features"]]
    ax.bar(labels, values, color=NPG[:4])
    ax.set_yscale("log")
    ax.set_ylabel("Genes (log scale)")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Outcome-blind feature reduction")
    panel_label(ax, "a", dx=-0.18)

    ax = axes[1]
    cohort_names = [item["accession"] for item in external]
    genes = [item["expression_features"] for item in external]
    ax.bar(cohort_names, genes, color=[COHORT_COLORS[name] for name in cohort_names])
    for idx, value in enumerate(genes):
        ax.text(idx, value + 30, f"{value:,}", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("Mapped gene symbols")
    ax.tick_params(axis="x", rotation=28)
    ax.set_title("External platform coverage")
    panel_label(ax, "b", dx=-0.18)

    ax = axes[2]
    sample_counts = [item["eligible_os_samples"] for item in external]
    event_counts = [item["events"] for item in external]
    pos = np.arange(len(cohort_names))
    ax.bar(pos, sample_counts, color=[COHORT_COLORS[name] for name in cohort_names], alpha=0.85, label="Patients")
    ax.bar(pos, event_counts, color=C_SURVIVALPFN, alpha=0.9, label="Deaths")
    ax.set_xticks(pos, cohort_names, rotation=28)
    ax.set_ylabel("Count")
    ax.set_title("External endpoint information")
    ax.legend()
    panel_label(ax, "c", dx=-0.18)
    fig.tight_layout(w_pad=1.5)
    fig.subplots_adjust(bottom=0.24)
    return save(fig, artifacts / "figures", "Supplementary_Figure_S2_preprocessing_audit")


def _supplementary_figure_5_dcalibration(artifacts: Path) -> list[dict[str, str | int]]:
    csd = pd.read_csv(artifacts / "results" / "csd_calibration_metrics.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), sharex=True)
    for col, cohort in enumerate(("GSE72094", "GSE68465", "GSE31210")):
        for row_idx, (model, display, color) in enumerate(
            (("SurvivalPFN-split", "Raw", C_SURVIVALPFN), ("CSD-SurvivalPFN", "CSD", C_CONFORMAL))
        ):
            record = csd.loc[csd["cohort"].eq(cohort) & csd["model"].eq(model)].iloc[0]
            histogram = np.asarray(json.loads(record["d_calibration_histogram"]), dtype=float)
            expected = histogram.sum() / len(histogram)
            ax = axes[row_idx, col]
            ax.bar(np.arange(1, 11), histogram, color=color, alpha=0.85)
            ax.axhline(expected, color=C_INK, linestyle="--", linewidth=1)
            ax.set_title(f"{cohort} - {display}\np={record['d_calibration_p']:.3g}")
            ax.set_xlabel("D-calibration bin")
            if col == 0:
                ax.set_ylabel("Observed/weighted count")
            panel_label(ax, chr(ord("a") + row_idx * 3 + col), dx=0.01, dy=1.02)
    fig.suptitle("Full D-calibration histograms show cohort-specific behavior", y=1.01, fontsize=11, fontweight="bold")
    fig.tight_layout(w_pad=1.0, h_pad=1.5)
    return save(fig, artifacts / "figures", "Supplementary_Figure_S5_dcalibration_histograms")


def _supplementary_figure_3_selection(artifacts: Path) -> list[dict[str, str | int]]:
    panel = pd.read_csv(artifacts / "results" / "stable_gene_panel.csv")
    internal = pd.read_csv(artifacts / "results" / "internal_cv_metrics.csv")
    internal["model"] = internal["model"].replace({"Clinical Cox": "Age/sex Cox"})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ax.scatter(panel["selection_frequency"], panel["median_abs_coefficient"],
               c=[C_SURVIVALPFN if value >= 0.60 else C_CLINICAL for value in panel["selection_frequency"]], s=45)
    for row in panel.itertuples(index=False):
        ax.annotate(row.gene, (row.selection_frequency, row.median_abs_coefficient), xytext=(3, 2), textcoords="offset points", fontsize=6.2)
    ax.axvline(0.60, color=C_INK, linestyle="--", linewidth=1)
    ax.set_xlabel("Selection frequency")
    ax.set_ylabel("Median absolute coefficient")
    ax.set_title("Selection frequency vs effect magnitude")
    panel_label(ax, "a", dx=-0.18)

    ax = axes[1]
    order = internal.groupby("model")["c_index"].mean().sort_values(ascending=False).index
    sns.boxplot(data=internal, y="model", x="c_index", order=order, hue="model",
                palette={model: MODEL_COLORS.get("Clinical Cox" if model == "Age/sex Cox" else model, C_INK) for model in order},
                legend=False, width=0.65, linewidth=1, fliersize=2, ax=ax)
    ax.axvline(0.5, color=C_INK, linestyle=":", linewidth=1)
    ax.set_xlabel("Internal fold C-index")
    ax.set_ylabel("")
    ax.set_title("Descriptive post-selection resampling")
    panel_label(ax, "b", dx=-0.30)
    fig.tight_layout(w_pad=1.8)
    return save(fig, artifacts / "figures", "Supplementary_Figure_S3_selection_internal_cv")


def make_all_figures(artifacts: Path) -> list[dict[str, str | int]]:
    apply_style()
    records: list[dict[str, str | int]] = []
    # Ordered as they are first cited in the manuscript.
    for function in (
        _figure_1,
        _figure_2,
        _figure_3,
        _figure_4,
        _figure_6_discrimination,
        _figure_7_biology,
        _figure_8_gate,
        _supplementary_figure_1_cohort_survival,
        _supplementary_figure_2_preprocessing,
        _supplementary_figure_3_selection,
        _supplementary_figure_4_directions,
        _supplementary_figure_5_dcalibration,
        _supplementary_figure_6_post_lock,
    ):
        records.extend(function(artifacts))
    manifest = pd.DataFrame(records)
    manifest.to_csv(artifacts / "manifests" / "figure_manifest.csv", index=False)
    return records

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chisquare
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.nonparametric import kaplan_meier_estimator

from .modeling import MASTER_SEED, harrell_c, load_discovery, survival_y


PRIMARY_COHORTS = ("GSE72094", "GSE68465")
ALL_COHORTS = PRIMARY_COHORTS + ("GSE31210",)


def paired_c_index_bootstrap(
    event: np.ndarray,
    time: np.ndarray,
    risk_a: np.ndarray,
    risk_b: np.ndarray,
    replicates: int = 2000,
) -> tuple[float, float, float]:
    point = harrell_c(event, time, risk_a) - harrell_c(event, time, risk_b)

    def one(seed: int) -> float:
        rng = np.random.default_rng(seed)
        for _ in range(20):
            index = rng.integers(0, len(time), len(time))
            if event[index].sum() >= 2:
                a = harrell_c(event[index], time[index], risk_a[index])
                b = harrell_c(event[index], time[index], risk_b[index])
                if np.isfinite(a) and np.isfinite(b):
                    return a - b
        return np.nan

    workers = min(8, max(1, (os.cpu_count() or 2) - 2))
    values = Parallel(n_jobs=workers, backend="loky")(
        delayed(one)(MASTER_SEED + 9000 + index) for index in range(replicates)
    )
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return point, float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))


def discrimination_tables(artifacts: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(artifacts / "results" / "external_predictions.csv.gz")
    extended: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for cohort, cohort_frame in predictions.groupby("cohort", sort=True):
        wide = cohort_frame.pivot(index="sample_id", columns="model", values="risk")
        outcome = cohort_frame.drop_duplicates("sample_id").set_index("sample_id").loc[wide.index]
        event = outcome["os_event"].to_numpy(int)
        time = outcome["os_days"].to_numpy(float)
        y = survival_y(event, time)
        event_times = time[event.astype(bool)]
        tau = float(min(np.quantile(event_times, 0.9), np.quantile(time, 0.9)))
        horizons = np.asarray([365.0, 1095.0, 1825.0])
        valid_horizons = horizons[(horizons > time.min()) & (horizons < tau)]
        for model in wide.columns:
            risk = wide[model].to_numpy(float)
            try:
                uno = float(concordance_index_ipcw(y, y, risk, tau=tau)[0])
            except ValueError:
                uno = float("nan")
            auc_values: dict[float, float] = {}
            if len(valid_horizons):
                try:
                    auc, _ = cumulative_dynamic_auc(y, y, risk, valid_horizons)
                    auc_values = dict(zip(valid_horizons, auc, strict=True))
                except ValueError:
                    auc_values = {}
            extended.append(
                {
                    "cohort": cohort,
                    "model": model,
                    "uno_c_index": uno,
                    "uno_tau_days": tau,
                    "auc_1y": auc_values.get(365.0, np.nan),
                    "auc_3y": auc_values.get(1095.0, np.nan),
                    "auc_5y": auc_values.get(1825.0, np.nan),
                }
            )
        for comparator in ("Clinical Cox", "Panel Cox"):
            delta, lower, upper = paired_c_index_bootstrap(
                event,
                time,
                wide["SurvivalPFN"].to_numpy(float),
                wide[comparator].to_numpy(float),
            )
            paired.append(
                {
                    "cohort": cohort,
                    "model": "SurvivalPFN",
                    "comparator": comparator,
                    "delta_harrell_c": delta,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "bootstrap_replicates": 2000,
                }
            )
    return pd.DataFrame(extended), pd.DataFrame(paired)


def _pfn_quantiles(distribution: Any, levels: np.ndarray) -> np.ndarray:
    probs = distribution.probs.detach().cpu().numpy()
    edges = distribution.bin_edges.detach().cpu().numpy()
    cdf = np.cumsum(probs, axis=1)
    output = np.zeros((probs.shape[0], len(levels)), dtype=float)
    for row in range(probs.shape[0]):
        for column, level in enumerate(levels):
            index = int(np.searchsorted(cdf[row], level, side="left"))
            if index >= probs.shape[1] - 1:
                finite_cdf = max(cdf[row, -2], 1e-6)
                output[row, column] = edges[row, -1] * level / finite_cdf
                continue
            lower_cdf = 0.0 if index == 0 else cdf[row, index - 1]
            upper_cdf = cdf[row, index]
            fraction = (level - lower_cdf) / max(upper_cdf - lower_cdf, 1e-8)
            output[row, column] = edges[row, index] + fraction * (
                edges[row, index + 1] - edges[row, index]
            )
    return output


def _km_conditional_samples(
    train_time: np.ndarray,
    train_event: np.ndarray,
    calibration_time: np.ndarray,
    calibration_event: np.ndarray,
    n_sample: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    km_time, km_survival = kaplan_meier_estimator(train_event.astype(bool), train_time)
    survival_before = np.r_[1.0, km_survival[:-1]]
    event_mass = np.maximum(survival_before - km_survival, 0.0)
    event_mask = event_mass > 0
    event_times = km_time[event_mask]
    event_mass = event_mass[event_mask]
    upper_tail = max(float(train_time.max()), float(calibration_time.max())) * 1.25
    samples = np.empty((len(calibration_time), n_sample), dtype=float)
    for index, (time, event) in enumerate(zip(calibration_time, calibration_event, strict=True)):
        if event:
            samples[index, :] = time
            continue
        valid = event_times > time
        candidate_times = event_times[valid]
        weights = event_mass[valid]
        residual = max(0.0, float(km_survival[-1]))
        candidate_times = np.r_[candidate_times, upper_tail]
        weights = np.r_[weights, residual]
        if weights.sum() <= 0:
            samples[index, :] = time
        else:
            weights = weights / weights.sum()
            samples[index, :] = rng.choice(candidate_times, size=n_sample, p=weights)
            samples[index, :] = np.maximum(samples[index, :], time)
    return samples


def _conformal_corrections(
    predicted_quantiles: np.ndarray,
    decensored_times: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    repeated_predictions = np.repeat(predicted_quantiles, decensored_times.shape[1], axis=0)
    scores = repeated_predictions - decensored_times.reshape(-1, 1)
    scores.sort(axis=0)
    indices = np.ceil((1.0 - levels) * (scores.shape[0] + 1)).astype(int) - 1
    indices = np.clip(indices, 0, scores.shape[0] - 1)
    return scores[indices, np.arange(len(levels))]


def quantiles_to_survival(
    quantile_times: np.ndarray, levels: np.ndarray, evaluation_times: np.ndarray
) -> np.ndarray:
    output = np.empty((len(quantile_times), len(evaluation_times)), dtype=float)
    for index, times in enumerate(quantile_times):
        times = np.maximum.accumulate(np.maximum(times, 0.0) + np.arange(len(times)) * 1e-8)
        cdf = np.interp(evaluation_times, np.r_[0.0, times], np.r_[0.0, levels], left=0.0, right=levels[-1])
        output[index] = 1.0 - cdf
    return output


def d_calibration(
    survival_at_observed: np.ndarray, event: np.ndarray, bins: int = 10
) -> tuple[float, float, list[float]]:
    histogram = np.zeros(bins, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    for value, observed in zip(np.clip(survival_at_observed, 0.0, 1.0), event, strict=True):
        if observed:
            index = min(int(value * bins), bins - 1)
            histogram[index] += 1.0
        elif value > 0:
            for index in range(bins):
                overlap = max(0.0, min(value, edges[index + 1]) - edges[index])
                histogram[index] += overlap / value
    expected = np.full(bins, histogram.sum() / bins)
    statistic, p_value = chisquare(histogram, expected)
    return float(statistic), float(p_value), histogram.tolist()


def run_csd_survivalpfn(artifacts: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.model_selection import train_test_split
    from sksurv.metrics import brier_score, integrated_brier_score
    from survivalpfn import SurvivalEstimator

    data = load_discovery(artifacts)
    panel = pd.read_csv(artifacts / "results" / "stable_gene_panel.csv")["gene"].tolist()
    x = data.expression.loc[:, panel].to_numpy(np.float32)
    train_index, calibration_index = train_test_split(
        np.arange(len(x)),
        test_size=0.30,
        random_state=MASTER_SEED + 4,
        stratify=data.event,
    )
    estimator = SurvivalEstimator(
        device="cuda:0",
        model_path="shi-ang/SurvivalPFN",
        cache_dir=str(Path("external_models") / "survivalpfn"),
    ).fit(
        x[train_index],
        data.event[train_index].astype(np.float32),
        data.time[train_index].astype(np.float32),
    )
    levels = np.arange(0.1, 1.0, 0.1)
    calibration_dist = estimator.predict_event_distribution(x[calibration_index])
    calibration_quantiles = _pfn_quantiles(calibration_dist, levels)
    decensored = _km_conditional_samples(
        data.time[train_index],
        data.event[train_index],
        data.time[calibration_index],
        data.event[calibration_index],
        n_sample=200,
        seed=MASTER_SEED + 5,
    )
    corrections = _conformal_corrections(calibration_quantiles, decensored, levels)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "method": "CSD KM-sampling correction applied to SurvivalPFN quantiles",
        "official_method_repository_commit": "08af6b5e950691daa27446bff74adb83c49659ba",
        "training_n": len(train_index),
        "calibration_n": len(calibration_index),
        "calibration_events": int(data.event[calibration_index].sum()),
        "km_samples_per_calibration_patient": 200,
        "quantile_levels": levels.tolist(),
        "quantile_corrections_days": corrections.tolist(),
    }
    for accession in ALL_COHORTS:
        expression = pd.read_csv(
            artifacts / "external" / f"{accession}_expression.csv.gz"
        ).set_index("sample_id")
        phenotype = pd.read_csv(artifacts / "external" / f"{accession}_phenotype.csv")
        x_test = expression.loc[:, panel].to_numpy(np.float32)
        event = phenotype["os_event"].to_numpy(int)
        time = phenotype["os_days"].to_numpy(float)
        dist = estimator.predict_event_distribution(x_test)
        raw_quantiles = _pfn_quantiles(dist, levels)
        csd_quantiles = np.maximum.accumulate(
            np.maximum(raw_quantiles - corrections[None, :], 0.0), axis=1
        )
        observed_grid = time.astype(np.float32)
        raw_survival_observed = dist.survival_at(
            __import__("torch").as_tensor(observed_grid, device="cuda:0")
        ).detach().cpu().numpy()
        csd_survival_observed = np.asarray(
            [
                quantiles_to_survival(csd_quantiles[i : i + 1], levels, np.asarray([time[i]]))[0, 0]
                for i in range(len(time))
            ]
        )
        tau = min(float(np.quantile(time, 0.9)), 1825.0)
        grid = np.linspace(max(30.0, float(np.quantile(time, 0.05))), tau, 30)
        raw_survival_grid = estimator.S(x_test, grid.astype(np.float32))
        csd_survival_grid = quantiles_to_survival(csd_quantiles, levels, grid)
        y = survival_y(event, time)
        raw_ibs = float(integrated_brier_score(y, y, raw_survival_grid, grid))
        csd_ibs = float(integrated_brier_score(y, y, csd_survival_grid, grid))
        for name, survival_observed, survival_grid in (
            ("SurvivalPFN-split", raw_survival_observed, raw_survival_grid),
            ("CSD-SurvivalPFN", csd_survival_observed, csd_survival_grid),
        ):
            statistic, p_value, histogram = d_calibration(survival_observed, event)
            _, brier = brier_score(y, y, survival_grid, grid)
            median_risk = -np.trapezoid(survival_grid, grid, axis=1)
            rows.append(
                {
                    "cohort": accession,
                    "model": name,
                    "n": len(time),
                    "events": int(event.sum()),
                    "c_index": harrell_c(event, time, median_risk),
                    "integrated_brier_score": raw_ibs if name.startswith("SurvivalPFN") else csd_ibs,
                    "brier_1y": float(brier[np.argmin(np.abs(grid - 365.0))]),
                    "brier_3y": float(brier[np.argmin(np.abs(grid - 1095.0))]),
                    "d_calibration_chisq": statistic,
                    "d_calibration_p": p_value,
                    "d_calibration_histogram": json.dumps(histogram),
                }
            )
    return pd.DataFrame(rows), details


def gene_effect_directions(artifacts: Path) -> pd.DataFrame:
    data = load_discovery(artifacts)
    panel = pd.read_csv(artifacts / "results" / "stable_gene_panel.csv")
    rows: list[dict[str, Any]] = []
    datasets: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {
        "TCGA-LUAD": (data.expression, data.event, data.time)
    }
    for accession in ALL_COHORTS:
        expression = pd.read_csv(
            artifacts / "external" / f"{accession}_expression.csv.gz"
        ).set_index("sample_id")
        phenotype = pd.read_csv(artifacts / "external" / f"{accession}_phenotype.csv")
        datasets[accession] = (
            expression,
            phenotype["os_event"].to_numpy(int),
            phenotype["os_days"].to_numpy(float),
        )
    for gene in panel["gene"]:
        signs: dict[str, int] = {}
        for cohort, (expression, event, time) in datasets.items():
            model = CoxPHSurvivalAnalysis(alpha=0.1).fit(
                expression[[gene]].to_numpy(float), survival_y(event, time)
            )
            coefficient = float(model.coef_[0])
            signs[cohort] = int(np.sign(coefficient))
            rows.append(
                {
                    "gene": gene,
                    "cohort": cohort,
                    "cox_coefficient": coefficient,
                    "direction": "higher_risk" if coefficient > 0 else "lower_risk",
                }
            )
        tcga_sign = signs["TCGA-LUAD"]
        for row in rows[-len(datasets) :]:
            row["matches_tcga_direction"] = row["cohort"] == "TCGA-LUAD" or signs[row["cohort"]] == tcga_sign
    return pd.DataFrame(rows)


def publication_gate(
    artifacts: Path,
    paired: pd.DataFrame,
    csd: pd.DataFrame,
    effects: pd.DataFrame,
) -> dict[str, Any]:
    metrics = pd.read_csv(artifacts / "results" / "external_validation_metrics.csv")
    panel = pd.read_csv(artifacts / "results" / "stable_gene_panel.csv")
    primary = metrics[metrics["cohort"].isin(PRIMARY_COHORTS)]
    piv = primary.pivot(index="cohort", columns="model", values="c_index")
    condition_1 = bool((piv["SurvivalPFN"] > piv["Clinical Cox"]).all())
    paired_primary = paired[
        paired["cohort"].isin(PRIMARY_COHORTS) & paired["comparator"].eq("Clinical Cox")
    ]
    condition_2 = bool((paired_primary["ci_lower"] > 0).any())
    csd_primary = csd[csd["cohort"].isin(PRIMARY_COHORTS)].pivot(
        index="cohort", columns="model", values="integrated_brier_score"
    )
    condition_3 = bool(
        (csd_primary["CSD-SurvivalPFN"] <= csd_primary["SurvivalPFN-split"] + 0.005).all()
    )
    primary_effects = effects[effects["cohort"].isin(PRIMARY_COHORTS)]
    direction_fraction = float(primary_effects["matches_tcga_direction"].mean())
    condition_4 = direction_fraction >= 0.75
    condition_5 = bool((panel["selection_frequency"] >= 0.60).all())
    condition_6 = bool("GSE31210" in metrics["cohort"].unique())
    conditions = {
        "1_pfn_beats_clinical_both_primary": condition_1,
        "2_paired_ci_excludes_zero_one_primary": condition_2,
        "3_csd_ibs_preserved_both_primary": condition_3,
        "4_gene_direction_concordance_at_least_75pct": condition_4,
        "5_all_panel_genes_stability_at_least_0_60": condition_5,
        "6_frozen_early_stage_scored": condition_6,
    }
    return {
        "gate_passed": all(conditions.values()),
        "conditions": conditions,
        "primary_gene_direction_concordance": direction_fraction,
        "stable_genes_at_0_60": int((panel["selection_frequency"] >= 0.60).sum()),
        "panel_size": len(panel),
        "required_interpretation": (
            "positive biomarker manuscript" if all(conditions.values()) else "comparative methods study; no clinical-readiness or superiority claim"
        ),
    }


def run_extended_validation(artifacts: Path) -> dict[str, Any]:
    results = artifacts / "results"
    extended, paired = discrimination_tables(artifacts)
    extended.to_csv(results / "extended_discrimination_metrics.csv", index=False)
    paired.to_csv(results / "paired_bootstrap_comparisons.csv", index=False)
    csd, csd_details = run_csd_survivalpfn(artifacts)
    csd.to_csv(results / "csd_calibration_metrics.csv", index=False)
    with (artifacts / "audit" / "csd_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(csd_details, handle, indent=2, sort_keys=True)
    effects = gene_effect_directions(artifacts)
    effects.to_csv(results / "gene_effect_directions.csv", index=False)
    gate = publication_gate(artifacts, paired, csd, effects)
    with (results / "publication_gate.json").open("w", encoding="utf-8") as handle:
        json.dump(gate, handle, indent=2, sort_keys=True)
    return gate

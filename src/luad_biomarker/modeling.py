from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv


MASTER_SEED = 20260806
PANEL_SIZE = 12


@dataclass
class DiscoveryData:
    expression: pd.DataFrame
    clinical: pd.DataFrame
    time: np.ndarray
    event: np.ndarray


def survival_y(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    return Surv.from_arrays(event.astype(bool), time.astype(float))


def harrell_c(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    if len(np.unique(risk)) < 2 or event.sum() < 2:
        return float("nan")
    return float(concordance_index_censored(event.astype(bool), time, risk)[0])


def _robust_scale_fit(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    center = x.median(axis=0)
    scale = (x - center).abs().median(axis=0) * 1.4826
    fallback = x.std(axis=0, ddof=1).replace(0.0, 1.0)
    scale = scale.where(scale > 1e-8, fallback).replace(0.0, 1.0)
    return (x - center) / scale, center, scale


def load_discovery(artifacts: Path) -> DiscoveryData:
    expression = pd.read_csv(artifacts / "interim" / "tcga_luad_model_matrix.csv.gz")
    cohort = pd.read_csv(artifacts / "interim" / "tcga_luad_patient_cohort.csv")
    if expression["selected_barcode"].tolist() != cohort["selected_barcode"].tolist():
        raise ValueError("TCGA expression and cohort order differ")
    x = expression.set_index("selected_barcode")
    x, _, _ = _robust_scale_fit(x)
    clinical = pd.DataFrame(index=x.index)
    clinical["age"] = pd.to_numeric(cohort["age_years"], errors="coerce").to_numpy()
    clinical["male"] = cohort["sex"].fillna("").str.lower().eq("male").astype(float).to_numpy()
    clinical["age"] = clinical["age"].fillna(clinical["age"].median())
    clinical, _, _ = _robust_scale_fit(clinical)
    return DiscoveryData(
        expression=x,
        clinical=clinical,
        time=cohort["os_days"].to_numpy(float),
        event=cohort["os_event"].to_numpy(int),
    )


def common_external_features(artifacts: Path, tcga_features: set[str]) -> list[str]:
    common = set(tcga_features)
    for accession in ("GSE72094", "GSE68465", "GSE31210"):
        columns = set(
            pd.read_csv(artifacts / "external" / f"{accession}_expression.csv.gz", nrows=0).columns
        ) - {"sample_id"}
        common &= columns
    return sorted(common)


def cox_score_z(x: np.ndarray, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    order = np.argsort(-time, kind="stable")
    xs = x[order]
    es = event[order].astype(bool)
    cum_x = np.cumsum(xs, axis=0)
    cum_x2 = np.cumsum(xs * xs, axis=0)
    n_risk = np.arange(1, len(time) + 1, dtype=float)[:, None]
    mean = cum_x / n_risk
    variance = np.maximum(cum_x2 / n_risk - mean * mean, 1e-12)
    score = (xs[es] - mean[es]).sum(axis=0)
    information = variance[es].sum(axis=0)
    return score / np.sqrt(np.maximum(information, 1e-12))


def choose_alpha(x: np.ndarray, event: np.ndarray, time: np.ndarray, seed: int) -> tuple[float, pd.DataFrame]:
    y = survival_y(event, time)
    path = CoxnetSurvivalAnalysis(
        l1_ratio=0.9, n_alphas=35, alpha_min_ratio=0.02, max_iter=100000, normalize=False
    ).fit(x, y)
    alphas = path.alphas_
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = np.full((5, len(alphas)), np.nan)
    for fold, (train, test) in enumerate(splitter.split(x, event)):
        model = CoxnetSurvivalAnalysis(
            l1_ratio=0.9, alphas=alphas, max_iter=100000, normalize=False
        ).fit(x[train], y[train])
        for index, alpha in enumerate(model.alphas_):
            scores[fold, index] = harrell_c(event[test], time[test], model.predict(x[test], alpha=alpha))
    mean = np.nanmean(scores, axis=0)
    se = np.nanstd(scores, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(scores), axis=0))
    best = int(np.nanargmax(mean))
    threshold = mean[best] - se[best]
    candidates = np.flatnonzero(mean >= threshold)
    chosen = int(candidates[0])  # alphas are descending: strongest within one SE
    table = pd.DataFrame({"alpha": alphas, "mean_c_index": mean, "se_c_index": se})
    table["selected"] = False
    table.loc[chosen, "selected"] = True
    return float(alphas[chosen]), table


def _bootstrap_coxnet(
    x: np.ndarray, event: np.ndarray, time: np.ndarray, alpha: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for _ in range(20):
        index = rng.integers(0, len(time), len(time))
        if event[index].sum() >= 10 and (~event[index].astype(bool)).sum() >= 10:
            break
    try:
        model = CoxnetSurvivalAnalysis(
            l1_ratio=0.9, alphas=[alpha], max_iter=100000, normalize=False
        ).fit(x[index], survival_y(event[index], time[index]))
        return model.coef_[:, -1]
    except Exception:
        return np.zeros(x.shape[1], dtype=float)


def select_stable_panel(
    data: DiscoveryData,
    common_features: list[str],
    bootstrap_replicates: int = 200,
    n_jobs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    x_common = data.expression.loc[:, common_features]
    z = cox_score_z(x_common.to_numpy(), data.time, data.event)
    screen_count = min(250, len(common_features))
    selected_indices = np.argsort(-np.abs(z))[:screen_count]
    screened_features = x_common.columns[selected_indices].tolist()
    x = x_common.loc[:, screened_features].to_numpy(dtype=float)
    alpha, alpha_table = choose_alpha(x, data.event, data.time, MASTER_SEED + 2)
    workers = n_jobs or min(12, max(1, (os.cpu_count() or 2) - 2))
    coefficients = Parallel(n_jobs=workers, backend="loky", verbose=5)(
        delayed(_bootstrap_coxnet)(
            x, data.event, data.time, alpha, MASTER_SEED + 1000 + replicate
        )
        for replicate in range(bootstrap_replicates)
    )
    coef = np.vstack(coefficients)
    frequency = (np.abs(coef) > 1e-10).mean(axis=0)
    median_abs = np.median(np.abs(coef), axis=0)
    sign_consistency = np.maximum((coef > 0).mean(axis=0), (coef < 0).mean(axis=0))
    stability = pd.DataFrame(
        {
            "gene": screened_features,
            "selection_frequency": frequency,
            "median_abs_coefficient": median_abs,
            "sign_consistency": sign_consistency,
            "univariate_score_z": z[selected_indices],
        }
    ).sort_values(
        ["selection_frequency", "median_abs_coefficient", "gene"],
        ascending=[False, False, True],
    )
    panel = stability.head(PANEL_SIZE).copy()
    panel["panel_rank"] = np.arange(1, len(panel) + 1)
    audit = {
        "tcga_samples": len(data.time),
        "events": int(data.event.sum()),
        "cross_platform_common_features": len(common_features),
        "outcome_screen_features": screen_count,
        "elastic_net_l1_ratio": 0.9,
        "selected_alpha_one_se": alpha,
        "bootstrap_replicates": bootstrap_replicates,
        "parallel_workers": workers,
        "panel_size": len(panel),
        "panel_genes": panel["gene"].tolist(),
    }
    return panel, alpha_table, audit


def clinical_matrix(phenotype: pd.DataFrame, accession: str) -> np.ndarray:
    if accession == "TCGA-LUAD":
        age = pd.to_numeric(phenotype["age_years"], errors="coerce")
        sex = phenotype["sex"].fillna("").str.lower()
    elif accession == "GSE72094":
        age = pd.to_numeric(phenotype["age_at_diagnosis"], errors="coerce")
        sex = phenotype["gender"].fillna("").str.lower()
    else:
        age_candidates = [
            pd.to_numeric(phenotype[field], errors="coerce")
            for field in ("age", "age_(years)")
            if field in phenotype
        ]
        if not age_candidates:
            age = pd.Series(np.nan, index=phenotype.index)
        else:
            age = max(age_candidates, key=lambda values: int(values.notna().sum()))
        sex_field = "sex" if "sex" in phenotype else "gender"
        sex = phenotype[sex_field].fillna("").str.lower()
    age_median = age.median()
    age = age.fillna(0.0 if not np.isfinite(age_median) else age_median)
    male = sex.str.startswith("m").astype(float)
    matrix = np.column_stack([age.to_numpy(float), male.to_numpy(float)])
    return StandardScaler().fit_transform(matrix)


def _cpu_models(seed: int) -> dict[str, Any]:
    return {
        "Clinical Cox": CoxPHSurvivalAnalysis(alpha=0.1),
        "Panel Cox": CoxPHSurvivalAnalysis(alpha=0.1),
        "Random survival forest": RandomSurvivalForest(
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=8,
            max_features="sqrt",
            n_jobs=1,
            random_state=seed,
        ),
        "Gradient boosting survival": GradientBoostingSurvivalAnalysis(
            loss="coxph",
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            random_state=seed,
        ),
    }


def evaluate_cpu_cv(
    data: DiscoveryData, panel_genes: list[str], n_jobs: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_gene = data.expression.loc[:, panel_genes].to_numpy(float)
    x_clinical = data.clinical.to_numpy(float)
    y = survival_y(data.event, data.time)
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=MASTER_SEED + 10)
    splits = list(splitter.split(x_gene, data.event))

    def fit_split(split_number: int, train: np.ndarray, test: np.ndarray):
        rows: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        for name, model in _cpu_models(MASTER_SEED + split_number).items():
            x = x_clinical if name == "Clinical Cox" else x_gene
            model.fit(x[train], y[train])
            risk = model.predict(x[test])
            rows.append(
                {
                    "model": name,
                    "repeat": split_number // 5 + 1,
                    "fold": split_number % 5 + 1,
                    "n_test": len(test),
                    "events_test": int(data.event[test].sum()),
                    "c_index": harrell_c(data.event[test], data.time[test], risk),
                }
            )
            if split_number < 5:
                predictions.extend(
                    {
                        "sample_index": int(index),
                        "model": name,
                        "risk": float(value),
                    }
                    for index, value in zip(test, risk, strict=True)
                )
        return rows, predictions

    workers = n_jobs or min(8, max(1, (os.cpu_count() or 2) - 2))
    results = Parallel(n_jobs=workers, backend="loky", verbose=5)(
        delayed(fit_split)(number, train, test)
        for number, (train, test) in enumerate(splits)
    )
    metrics = pd.DataFrame([row for rows, _ in results for row in rows])
    predictions = pd.DataFrame([row for _, preds in results for row in preds])
    return metrics, predictions


def fit_deepsurv(
    x_train: np.ndarray,
    event_train: np.ndarray,
    time_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    epochs: int = 300,
) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 32),
        nn.SiLU(),
        nn.Dropout(0.15),
        nn.Linear(32, 16),
        nn.SiLU(),
        nn.Linear(16, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    order = np.argsort(-time_train, kind="stable")
    x_tensor = torch.as_tensor(x_train[order], dtype=torch.float32, device=device)
    event_tensor = torch.as_tensor(event_train[order].astype(bool), device=device)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    patience = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk = model(x_tensor).squeeze(1)
        log_cumulative_hazard = torch.logcumsumexp(risk, dim=0)
        loss = -(risk[event_tensor] - log_cumulative_hazard[event_tensor]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        if value < best_loss - 1e-5:
            best_loss = value
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= 40:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x_test, dtype=torch.float32, device=device)).squeeze(1).cpu().numpy()


def evaluate_gpu_cv(data: DiscoveryData, panel_genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from survivalpfn import SurvivalEstimator

    x = data.expression.loc[:, panel_genes].to_numpy(np.float32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=MASTER_SEED + 20)
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    cache = str(Path("external_models") / "survivalpfn")
    for fold, (train, test) in enumerate(splitter.split(x, data.event), start=1):
        deep_risk = fit_deepsurv(
            x[train], data.event[train], data.time[train], x[test], MASTER_SEED + 200 + fold
        )
        estimator = SurvivalEstimator(
            device="cuda:0", model_path="shi-ang/SurvivalPFN", cache_dir=cache
        )
        estimator.fit(x[train], data.event[train].astype(np.float32), data.time[train].astype(np.float32))
        pfn_time = estimator.predict_event_time(x[test], type="median")
        for name, risk in (("DeepSurv", deep_risk), ("SurvivalPFN", -pfn_time)):
            rows.append(
                {
                    "model": name,
                    "repeat": 1,
                    "fold": fold,
                    "n_test": len(test),
                    "events_test": int(data.event[test].sum()),
                    "c_index": harrell_c(data.event[test], data.time[test], risk),
                }
            )
            predictions.extend(
                {"sample_index": int(index), "model": name, "risk": float(value)}
                for index, value in zip(test, risk, strict=True)
            )
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model_lock(
    artifacts: Path, panel_path: Path, panel_genes: list[str], selection_audit: dict[str, Any]
) -> Path:
    lock_dir = artifacts / "lock"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "model_spec.json"
    payload = {
        "locked_before_external_scoring": True,
        "seed": MASTER_SEED,
        "endpoint": "overall survival",
        "panel_genes": panel_genes,
        "panel_file_sha256": _sha256(panel_path),
        "models": [
            "Clinical Cox",
            "Panel Cox",
            "Random survival forest",
            "Gradient boosting survival",
            "DeepSurv",
            "SurvivalPFN",
        ],
        "selection": selection_audit,
        "primary_external": ["GSE72094", "GSE68465"],
        "secondary_external": ["GSE31210"],
    }
    with lock_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return lock_path


def bootstrap_c_index(
    event: np.ndarray,
    time: np.ndarray,
    risk: np.ndarray,
    replicates: int = 1000,
) -> tuple[float, float, float]:
    point = harrell_c(event, time, risk)

    def one(seed: int) -> float:
        rng = np.random.default_rng(seed)
        for _ in range(20):
            index = rng.integers(0, len(time), len(time))
            if event[index].sum() >= 2:
                value = harrell_c(event[index], time[index], risk[index])
                if np.isfinite(value):
                    return value
        return np.nan

    values = Parallel(n_jobs=min(8, max(1, (os.cpu_count() or 2) - 2)), backend="loky")(
        delayed(one)(MASTER_SEED + 5000 + index) for index in range(replicates)
    )
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    return point, float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))


def evaluate_external(
    artifacts: Path, data: DiscoveryData, panel_genes: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from survivalpfn import SurvivalEstimator

    lock_path = artifacts / "lock" / "model_spec.json"
    if not lock_path.exists():
        raise RuntimeError("External scoring refused: model_spec.json is not locked")
    x_train = data.expression.loc[:, panel_genes].to_numpy(float)
    y_train = survival_y(data.event, data.time)
    cohort = pd.read_csv(artifacts / "interim" / "tcga_luad_patient_cohort.csv")
    c_train = clinical_matrix(cohort, "TCGA-LUAD")
    cpu_models = _cpu_models(MASTER_SEED + 30)
    for name, model in cpu_models.items():
        model.fit(c_train if name == "Clinical Cox" else x_train, y_train)
    pfn = SurvivalEstimator(
        device="cuda:0",
        model_path="shi-ang/SurvivalPFN",
        cache_dir=str(Path("external_models") / "survivalpfn"),
    ).fit(x_train.astype(np.float32), data.event.astype(np.float32), data.time.astype(np.float32))

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for accession in ("GSE72094", "GSE68465", "GSE31210"):
        expression = pd.read_csv(artifacts / "external" / f"{accession}_expression.csv.gz").set_index("sample_id")
        phenotype = pd.read_csv(artifacts / "external" / f"{accession}_phenotype.csv")
        if expression.index.tolist() != phenotype["geo_accession"].tolist():
            raise ValueError(f"{accession} expression/phenotype order differs")
        x_test = expression.loc[:, panel_genes].to_numpy(float)
        c_test = clinical_matrix(phenotype, accession)
        event = phenotype["os_event"].to_numpy(int)
        time = phenotype["os_days"].to_numpy(float)
        risks: dict[str, np.ndarray] = {}
        for name, model in cpu_models.items():
            risks[name] = model.predict(c_test if name == "Clinical Cox" else x_test)
        risks["DeepSurv"] = fit_deepsurv(
            x_train, data.event, data.time, x_test, MASTER_SEED + 300, epochs=400
        )
        risks["SurvivalPFN"] = -pfn.predict_event_time(x_test.astype(np.float32), type="median")
        for name, risk in risks.items():
            point, lower, upper = bootstrap_c_index(event, time, np.asarray(risk), 1000)
            metric_rows.append(
                {
                    "cohort": accession,
                    "model": name,
                    "n": len(time),
                    "events": int(event.sum()),
                    "c_index": point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
            prediction_rows.extend(
                {
                    "cohort": accession,
                    "sample_id": sample,
                    "os_days": float(t),
                    "os_event": int(e),
                    "model": name,
                    "risk": float(r),
                }
                for sample, t, e, r in zip(expression.index, time, event, risk, strict=True)
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def run_analysis(artifacts: Path, bootstrap_replicates: int = 200) -> dict[str, Any]:
    results_dir = artifacts / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    data = load_discovery(artifacts)
    common = common_external_features(artifacts, set(data.expression.columns))
    panel, alpha_table, selection_audit = select_stable_panel(
        data, common, bootstrap_replicates=bootstrap_replicates
    )
    panel_path = results_dir / "stable_gene_panel.csv"
    panel.to_csv(panel_path, index=False)
    alpha_table.to_csv(results_dir / "elastic_net_alpha_cv.csv", index=False)
    with (artifacts / "audit" / "feature_selection_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(selection_audit, handle, indent=2, sort_keys=True)
    panel_genes = panel["gene"].tolist()

    cpu_metrics, cpu_predictions = evaluate_cpu_cv(data, panel_genes)
    gpu_metrics, gpu_predictions = evaluate_gpu_cv(data, panel_genes)
    internal_metrics = pd.concat([cpu_metrics, gpu_metrics], ignore_index=True)
    internal_predictions = pd.concat([cpu_predictions, gpu_predictions], ignore_index=True)
    internal_metrics.to_csv(results_dir / "internal_cv_metrics.csv", index=False)
    internal_predictions.to_csv(results_dir / "internal_oof_predictions.csv", index=False)

    lock_path = write_model_lock(artifacts, panel_path, panel_genes, selection_audit)
    external_metrics, external_predictions = evaluate_external(artifacts, data, panel_genes)
    external_metrics.to_csv(results_dir / "external_validation_metrics.csv", index=False)
    external_predictions.to_csv(results_dir / "external_predictions.csv.gz", index=False, compression="gzip")
    summary = {
        "panel_genes": panel_genes,
        "model_lock": str(lock_path),
        "internal_models": sorted(internal_metrics["model"].unique()),
        "external_cohorts": sorted(external_metrics["cohort"].unique()),
        "gpu_models": ["DeepSurv", "SurvivalPFN"],
        "external_metrics_rows": len(external_metrics),
    }
    with (results_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def score_locked_external(artifacts: Path) -> dict[str, Any]:
    results_dir = artifacts / "results"
    panel_path = results_dir / "stable_gene_panel.csv"
    lock_path = artifacts / "lock" / "model_spec.json"
    if not panel_path.exists() or not lock_path.exists():
        raise RuntimeError("Locked panel artifacts are missing")
    with lock_path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if _sha256(panel_path) != lock["panel_file_sha256"]:
        raise RuntimeError("Stable panel changed after model lock")
    panel_genes = pd.read_csv(panel_path)["gene"].tolist()
    if panel_genes != lock["panel_genes"]:
        raise RuntimeError("Panel gene order differs from model lock")
    data = load_discovery(artifacts)
    metrics, predictions = evaluate_external(artifacts, data, panel_genes)
    metrics.to_csv(results_dir / "external_validation_metrics.csv", index=False)
    predictions.to_csv(
        results_dir / "external_predictions.csv.gz", index=False, compression="gzip"
    )
    summary = {
        "panel_genes": panel_genes,
        "model_lock": str(lock_path),
        "external_cohorts": sorted(metrics["cohort"].unique()),
        "gpu_models": ["DeepSurv", "SurvivalPFN"],
        "external_metrics_rows": len(metrics),
        "resumed_from_lock": True,
    }
    with (results_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary

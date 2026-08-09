from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis

from .modeling import (
    MASTER_SEED,
    bootstrap_c_index,
    clinical_matrix,
    load_discovery,
    survival_y,
)


def _stage_values(phenotype: pd.DataFrame, accession: str) -> np.ndarray | None:
    if accession == "TCGA-LUAD":
        values = phenotype["stage_group"].map({"I": 1, "II": 2, "III": 3, "IV": 4})
    elif accession == "GSE72094":
        values = pd.to_numeric(
            phenotype["stage"].astype(str).str.extract(r"([1-4])", expand=False),
            errors="coerce",
        )
    elif accession == "GSE31210":
        values = phenotype["pstage_iorii"].astype(str).str.upper().map({"I": 1, "II": 2})
    else:
        return None
    if values.notna().sum() < len(values) * 0.70:
        return None
    values = values.fillna(values.median()).to_numpy(float)
    return ((values - values.mean()) / max(values.std(ddof=1), 1e-8)).reshape(-1, 1)


def run_supplementary_models(artifacts: Path) -> dict[str, Any]:
    import xgboost as xgb
    from survivalpfn import SurvivalEstimator

    results_dir = artifacts / "results"
    lock = json.loads((artifacts / "lock" / "model_spec.json").read_text(encoding="utf-8"))
    panel = lock["panel_genes"]
    data = load_discovery(artifacts)
    x_gene = data.expression.loc[:, panel].to_numpy(np.float32)
    cohort = pd.read_csv(artifacts / "interim" / "tcga_luad_patient_cohort.csv")
    x_clinical = clinical_matrix(cohort, "TCGA-LUAD").astype(np.float32)
    x_combined = np.column_stack([x_gene, x_clinical]).astype(np.float32)

    signed_time = np.where(data.event.astype(bool), data.time, -data.time)
    xgb_model = xgb.XGBRegressor(
        objective="survival:cox",
        n_estimators=600,
        max_depth=2,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=8,
        reg_lambda=5.0,
        reg_alpha=0.2,
        tree_method="hist",
        device="cuda",
        random_state=MASTER_SEED + 40,
        n_jobs=4,
    ).fit(x_gene, signed_time, verbose=False)
    pfn_clinical = SurvivalEstimator(
        device="cuda:0",
        model_path="shi-ang/SurvivalPFN",
        cache_dir=str(Path("external_models") / "survivalpfn"),
    ).fit(x_combined, data.event.astype(np.float32), data.time.astype(np.float32))

    stage_train = _stage_values(cohort, "TCGA-LUAD")
    stage_model = None
    if stage_train is not None:
        stage_model = CoxPHSurvivalAnalysis(alpha=0.1).fit(
            stage_train, survival_y(data.event, data.time)
        )

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for accession in ("GSE72094", "GSE68465", "GSE31210"):
        expression = pd.read_csv(
            artifacts / "external" / f"{accession}_expression.csv.gz"
        ).set_index("sample_id")
        phenotype = pd.read_csv(artifacts / "external" / f"{accession}_phenotype.csv")
        x_external_gene = expression.loc[:, panel].to_numpy(np.float32)
        x_external_clinical = clinical_matrix(phenotype, accession).astype(np.float32)
        x_external_combined = np.column_stack(
            [x_external_gene, x_external_clinical]
        ).astype(np.float32)
        event = phenotype["os_event"].to_numpy(int)
        time = phenotype["os_days"].to_numpy(float)
        risks: dict[str, np.ndarray] = {
            "XGBoost Cox": xgb_model.predict(x_external_gene),
            "SurvivalPFN + age/sex": -pfn_clinical.predict_event_time(
                x_external_combined, type="median"
            ),
        }
        stage_external = _stage_values(phenotype, accession)
        if stage_model is not None and stage_external is not None:
            risks["Stage-only Cox"] = stage_model.predict(stage_external)
        for model, risk in risks.items():
            point, lower, upper = bootstrap_c_index(event, time, risk, 1000)
            metric_rows.append(
                {
                    "cohort": accession,
                    "model": model,
                    "n": len(time),
                    "events": int(event.sum()),
                    "c_index": point,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "post_lock_supplementary": True,
                }
            )
            prediction_rows.extend(
                {
                    "cohort": accession,
                    "sample_id": sample,
                    "model": model,
                    "risk": float(value),
                }
                for sample, value in zip(expression.index, risk, strict=True)
            )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(results_dir / "supplementary_models_metrics.csv", index=False)
    predictions.to_csv(results_dir / "supplementary_models_predictions.csv.gz", index=False, compression="gzip")
    audit = {
        "post_lock_supplementary": True,
        "not_used_for_publication_gate": True,
        "gpu_models": ["XGBoost Cox", "SurvivalPFN + age/sex"],
        "stage_only_cohorts": sorted(
            metrics.loc[metrics["model"].eq("Stage-only Cox"), "cohort"].tolist()
        ),
    }
    (artifacts / "audit" / "supplementary_models_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    return audit

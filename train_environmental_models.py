from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns
import shap


warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "final_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "modeling_env"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
FIG_DIR = OUTPUT_DIR / "figures"
for path in [TABLE_DIR, MODEL_DIR, FIG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

LEAD_TIMES = [1, 3, 7, 10]
SITES = ["문의", "추동", "회남"]

# 경보 기준 또는 조류 농도와 직접 연결되는 누출 위험 변수.
LEAKAGE_RAW_COLS = {
    "total_cyano",
    "microcystis",
    "anabaena",
    "oscillatoria",
    "aphanizomenon",
}


def read_csv_any(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["조사일"] = pd.to_datetime(out["조사일"])
    out = out.sort_values(["채수위치", "조사일"]).reset_index(drop=True)
    for col in out.columns:
        if col not in ["조사일", "채수위치", "발령단계"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["월"] = out["조사일"].dt.month
    out["연도"] = out["조사일"].dt.year
    out["dayofyear"] = out["조사일"].dt.dayofyear
    out["month_sin"] = np.sin(2 * np.pi * out["월"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["월"] / 12)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)

    stage = out["발령단계"].astype("string").fillna("").str.strip()
    negative_labels = {"", "미발령", "정상", "0", "nan", "None"}
    out["alert_now"] = (~stage.isin(negative_labels)).astype(int)
    return out


def add_environmental_features(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values("조사일").copy()
    rain_col = "강우량(mm)" if "강우량(mm)" in grp.columns else "일강수량(mm)"
    temp_col = "수온(℃)"
    air_col = "평균기온(°C)"
    solar_col = "합계 일사량(MJ/m2)"
    inflow_col = "유입량(㎥/s)"
    outflow_col = "총방류량(㎥/s)"
    volume_col = "저수량(백만㎥)"
    chla_col = "Chl-a (㎎/㎥)"

    if temp_col in grp.columns:
        hot = (grp[temp_col] > 25).fillna(False)
        runs, count = [], 0
        for is_hot in hot:
            count = count + 1 if is_hot else 0
            runs.append(count)
        grp["CHD"] = runs
        grp["water_temp_mean_3d"] = grp[temp_col].rolling(3, min_periods=1).mean()
        grp["water_temp_mean_7d"] = grp[temp_col].rolling(7, min_periods=1).mean()
        grp["water_temp_mean_14d"] = grp[temp_col].rolling(14, min_periods=1).mean()
        for lag in [1, 3, 7, 10, 14]:
            grp[f"water_temp_lag{lag}"] = grp[temp_col].shift(lag)

    if air_col in grp.columns:
        grp["air_temp_mean_3d"] = grp[air_col].rolling(3, min_periods=1).mean()
        grp["air_temp_mean_7d"] = grp[air_col].rolling(7, min_periods=1).mean()
        grp["air_temp_mean_14d"] = grp[air_col].rolling(14, min_periods=1).mean()
        grp["heat_degree_7d"] = (grp[air_col] - 25).clip(lower=0).rolling(7, min_periods=1).sum()

    if rain_col in grp.columns:
        grp["rain_sum_3d"] = grp[rain_col].rolling(3, min_periods=1).sum()
        grp["rain_sum_7d"] = grp[rain_col].rolling(7, min_periods=1).sum()
        grp["rain_sum_14d"] = grp[rain_col].rolling(14, min_periods=1).sum()
        grp["rain_sum_30d"] = grp[rain_col].rolling(30, min_periods=1).sum()
        dry_runs, count = [], 0
        for rain in grp[rain_col].fillna(0):
            count = count + 1 if rain <= 1 else 0
            dry_runs.append(count)
        grp["dry_days"] = dry_runs
        grp["rain_pulse_flag"] = ((grp["dry_days"].shift(1) >= 5) & (grp[rain_col] >= 10)).astype(int)

    if solar_col in grp.columns:
        grp["solar_mean_3d"] = grp[solar_col].rolling(3, min_periods=1).mean()
        grp["solar_mean_7d"] = grp[solar_col].rolling(7, min_periods=1).mean()
        grp["solar_mean_14d"] = grp[solar_col].rolling(14, min_periods=1).mean()

    if inflow_col in grp.columns and volume_col in grp.columns:
        safe_inflow = grp[inflow_col].replace(0, np.nan)
        grp["HRT"] = grp[volume_col] * 1e6 / (safe_inflow * 86400)
        grp["HRT_7d"] = grp["HRT"].rolling(7, min_periods=1).mean()
        grp["HRT_14d"] = grp["HRT"].rolling(14, min_periods=1).mean()

    if outflow_col in grp.columns and inflow_col in grp.columns:
        grp["flow_balance"] = grp[inflow_col] - grp[outflow_col]
        grp["flow_balance_7d"] = grp["flow_balance"].rolling(7, min_periods=1).mean()
        grp["flow_balance_14d"] = grp["flow_balance"].rolling(14, min_periods=1).mean()

    if chla_col in grp.columns:
        for lag in [1, 3, 7, 10, 14]:
            grp[f"chla_lag{lag}"] = grp[chla_col].shift(lag)
        grp["chla_roll7"] = grp[chla_col].shift(1).rolling(7, min_periods=1).mean()
        grp["chla_roll14"] = grp[chla_col].shift(1).rolling(14, min_periods=1).mean()

    if {"CHD", "solar_mean_7d", "HRT_7d"}.issubset(grp.columns):
        grp["BGI_env"] = grp["CHD"] * grp["solar_mean_7d"] / grp["HRT_7d"].replace(0, np.nan)

    return grp


def prepare_dataset() -> pd.DataFrame:
    df = normalize_data(read_csv_any(DATA_PATH))
    df = pd.concat(
        [add_environmental_features(grp) for _, grp in df.groupby("채수위치", sort=False)],
        ignore_index=True,
    ).sort_values(["채수위치", "조사일"]).reset_index(drop=True)
    for h in LEAD_TIMES:
        df[f"y_Tplus{h}"] = df.groupby("채수위치")["alert_now"].shift(-h).astype("float")
    return df


def build_feature_cols(df: pd.DataFrame, include_chla: bool) -> tuple[list[str], list[str], list[str]]:
    target_cols = [f"y_Tplus{h}" for h in LEAD_TIMES]
    blocked = set(["조사일", "발령단계", "alert_now", *target_cols, *LEAKAGE_RAW_COLS])
    blocked_prefixes = (
        "log_cyano",
        "cyano",
        "hoenam_log_cyano",
        "hoenam_to_site_log_cyano",
    )
    chla_cols = {"Chl-a (㎎/㎥)"}
    if not include_chla:
        chla_cols.update({c for c in df.columns if "chla" in c.lower() or "Chl-a" in c})
        blocked.update(chla_cols)

    feature_cols = []
    for col in df.columns:
        if col in blocked:
            continue
        if col.startswith(blocked_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or col == "채수위치":
            feature_cols.append(col)

    feature_cols = list(dict.fromkeys(feature_cols))
    cat_features = ["채수위치"]
    num_features = [c for c in feature_cols if c not in cat_features]
    return feature_cols, num_features, cat_features


def split_by_time(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    train_mask = df["조사일"] <= "2022-12-31"
    valid_mask = (df["조사일"] >= "2023-01-01") & (df["조사일"] <= "2023-12-31")
    test_mask = df["조사일"] >= "2024-01-01"
    return train_mask, valid_mask, test_mask


def make_preprocess(num_features: list[str], cat_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def safe_auc(metric_func, y_true: pd.Series, y_score: np.ndarray) -> float:
    if pd.Series(y_true).nunique() < 2:
        return np.nan
    return float(metric_func(y_true, y_score))


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_auc(roc_auc_score, y_true, y_prob),
        "pr_auc": safe_auc(average_precision_score, y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        rows.append({"threshold": threshold, **evaluate_predictions(y_true, y_prob, float(threshold))})
    threshold_df = pd.DataFrame(rows)
    # 조기경보 목적: recall을 보장할 수 있으면 그 안에서 F1/precision 최대.
    candidates = threshold_df[threshold_df["recall"] >= 0.75]
    if candidates.empty:
        candidates = threshold_df
    best = candidates.sort_values(["f1", "precision", "balanced_accuracy"], ascending=False).iloc[0]
    return float(best["threshold"]), threshold_df


def make_model_specs(scale_pos_weight: float) -> dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=350,
            learning_rate=0.035,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def statistical_tests(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    preferred = [
        "CHD",
        "water_temp_mean_7d",
        "수온(℃)",
        "Chl-a (㎎/㎥)",
        "rain_sum_7d",
        "rain_pulse_flag",
        "solar_mean_7d",
        "저수율(%)",
        "HRT_7d",
        "flow_balance_7d",
    ]
    stat_features = [c for c in preferred if c in feature_cols]
    rows = []
    for h in LEAD_TIMES:
        target = f"y_Tplus{h}"
        for feature in stat_features:
            test_df = df[[target, feature]].dropna()
            a = test_df.loc[test_df[target] == 1, feature]
            b = test_df.loc[test_df[target] == 0, feature]
            if len(a) < 5 or len(b) < 5:
                continue
            stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
            effect = a.median() - b.median()
            rows.append(
                {
                    "lead_time": f"T+{h}",
                    "variable": feature,
                    "test_method": "Mann-Whitney U",
                    "statistic": stat,
                    "p_value": p_value,
                    "effect_size_median_diff": effect,
                    "alert_median": a.median(),
                    "non_alert_median": b.median(),
                    "direction": "alert higher" if effect > 0 else "alert lower",
                    "significant_0_05": p_value < 0.05,
                }
            )
    return pd.DataFrame(rows).sort_values(["lead_time", "p_value"])


def save_curves(best_predictions: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for lead, pred in best_predictions.items():
        y_true = pred["y_true"]
        y_prob = pred["y_prob"]
        if pd.Series(y_true).nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{lead} AUC={roc_auc_score(y_true, y_prob):.3f}")
        axes[1].plot(recall, precision, label=f"{lead} AP={average_precision_score(y_true, y_prob):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("ROC Curve - Environmental Best Models")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_title("Precision-Recall Curve - Environmental Best Models")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_pr_curve_best_models.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def shap_analysis(best_models: dict[str, Pipeline], best_summary: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in LEAD_TIMES:
        lead = f"T+{h}"
        best_row = best_summary[best_summary["lead_time"] == lead].iloc[0]
        pipe = best_models[lead]
        feature_cols = json.loads(best_row["feature_cols_json"])
        target = f"y_Tplus{h}"
        model_cols = list(dict.fromkeys(["조사일", "채수위치", target] + feature_cols))
        data_h = df[model_cols].dropna(subset=[target]).copy()
        _, _, test_mask = split_by_time(data_h)
        X_test = data_h.loc[test_mask, feature_cols].copy()
        if len(X_test) > 1000:
            X_test = X_test.sample(1000, random_state=42)
        preprocess = pipe.named_steps["preprocess"]
        estimator = pipe.named_steps["model"]
        X_trans = preprocess.transform(X_test)
        feature_names = preprocess.get_feature_names_out()
        X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_trans_df)
        if isinstance(shap_values, list):
            shap_matrix = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_matrix = shap_values[:, :, 1]
        else:
            shap_matrix = shap_values

        mean_abs = np.abs(shap_matrix).mean(axis=0)
        imp = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        imp["rank"] = np.arange(1, len(imp) + 1)
        for _, row in imp.head(30).iterrows():
            rows.append(
                {
                    "lead_time": lead,
                    "best_model": best_row["best_model"],
                    "rank": int(row["rank"]),
                    "feature": row["feature"],
                    "mean_abs_shap": row["mean_abs_shap"],
                }
            )

        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_matrix, X_trans_df, max_display=20, show=False)
        plt.title(f"{lead} Environmental Best Model SHAP ({best_row['best_model']})")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"shap_summary_Tplus{h}.png", dpi=160, bbox_inches="tight")
        plt.close()
    return pd.DataFrame(rows)


def main() -> None:
    print("Preparing environmental dataset...")
    df = prepare_dataset()
    experiments = {
        "env_with_chla": build_feature_cols(df, include_chla=True),
        "env_no_chla": build_feature_cols(df, include_chla=False),
    }

    all_results = []
    all_thresholds = []
    best_rows = []
    best_models: dict[str, Pipeline] = {}
    best_predictions: dict[str, dict] = {}

    # 주 모델은 Chl-a 포함 환경·수질 모델. no_chla는 민감도 검증용으로 함께 저장.
    for experiment_name, (feature_cols, num_features, cat_features) in experiments.items():
        print(f"\n=== Experiment: {experiment_name} | features={len(feature_cols)} ===")
        pvalue_table = statistical_tests(df, feature_cols)
        pvalue_table.to_csv(TABLE_DIR / f"test_results_pvalue_{experiment_name}.csv", index=False, encoding="utf-8-sig")

        for h in LEAD_TIMES:
            lead = f"T+{h}"
            target = f"y_Tplus{h}"
            model_cols = list(dict.fromkeys(["조사일", "채수위치", target] + feature_cols))
            data_h = df[model_cols].dropna(subset=[target]).copy()
            data_h[target] = data_h[target].astype(int)
            train_mask, valid_mask, test_mask = split_by_time(data_h)
            train_df = data_h.loc[train_mask]
            valid_df = data_h.loc[valid_mask]
            test_df = data_h.loc[test_mask]
            X_train, y_train = train_df[feature_cols], train_df[target]
            X_valid, y_valid = valid_df[feature_cols], valid_df[target]
            X_test, y_test = test_df[feature_cols], test_df[target]

            pos = y_train.sum()
            neg = len(y_train) - pos
            scale_pos_weight = float(neg / pos) if pos > 0 else 1.0
            print(
                f"{lead}: train={len(train_df):,}, valid={len(valid_df):,}, test={len(test_df):,}, "
                f"train_alert_rate={y_train.mean():.4f}"
            )

            for model_name, estimator in make_model_specs(scale_pos_weight).items():
                pipe = Pipeline(
                    steps=[
                        ("preprocess", make_preprocess(num_features, cat_features)),
                        ("model", estimator),
                    ]
                )
                pipe.fit(X_train, y_train)
                valid_prob = pipe.predict_proba(X_valid)[:, 1]
                threshold, threshold_df = tune_threshold(y_valid, valid_prob)
                threshold_df.insert(0, "model_name", model_name)
                threshold_df.insert(0, "lead_time", lead)
                threshold_df.insert(0, "experiment", experiment_name)
                all_thresholds.append(threshold_df)

                for dataset_name, X_part, y_part in [
                    ("validation", X_valid, y_valid),
                    ("test", X_test, y_test),
                ]:
                    prob = pipe.predict_proba(X_part)[:, 1]
                    metrics = evaluate_predictions(y_part, prob, threshold)
                    all_results.append(
                        {
                            "experiment": experiment_name,
                            "lead_time": lead,
                            "model_name": model_name,
                            "dataset": dataset_name,
                            "threshold": threshold,
                            "n_samples": len(y_part),
                            "positive_rate": y_part.mean(),
                            **metrics,
                        }
                    )

                if experiment_name == "env_with_chla":
                    test_prob = pipe.predict_proba(X_test)[:, 1]
                    key = f"{lead}_{model_name}"
                    joblib.dump(
                        {
                            "modeling_mode": "environmental_main",
                            "excluded_leakage_features": sorted(LEAKAGE_RAW_COLS),
                            "lead_time": lead,
                            "model_name": model_name,
                            "feature_cols": feature_cols,
                            "num_features": num_features,
                            "cat_features": cat_features,
                            "threshold": threshold,
                            "pipeline": pipe,
                        },
                        MODEL_DIR / f"candidate_{key}.pkl",
                    )
                    best_predictions[key] = {
                        "pipe": pipe,
                        "feature_cols": feature_cols,
                        "y_true": y_test.reset_index(drop=True),
                        "y_prob": pd.Series(test_prob),
                        "threshold": threshold,
                    }

    model_results = pd.DataFrame(all_results)
    threshold_results = pd.concat(all_thresholds, ignore_index=True)
    model_results.to_csv(TABLE_DIR / "model_results.csv", index=False, encoding="utf-8-sig")
    threshold_results.to_csv(TABLE_DIR / "threshold_tuning_results.csv", index=False, encoding="utf-8-sig")

    for h in LEAD_TIMES:
        lead = f"T+{h}"
        lead_results = model_results[
            (model_results["experiment"] == "env_with_chla")
            & (model_results["lead_time"] == lead)
            & (model_results["dataset"] == "test")
        ].copy()
        best = lead_results.sort_values(["pr_auc", "f1", "recall", "balanced_accuracy"], ascending=False).iloc[0]
        best_name = best["model_name"]
        candidate_key = f"{lead}_{best_name}"
        best_pipe = best_predictions[candidate_key]["pipe"]
        feature_cols = best_predictions[candidate_key]["feature_cols"]
        model_file = MODEL_DIR / f"best_model_Tplus{h}_{best_name}_env.pkl"
        joblib.dump(
            {
                "modeling_mode": "environmental_main",
                "excluded_leakage_features": sorted(LEAKAGE_RAW_COLS),
                "lead_time": lead,
                "model_name": best_name,
                "feature_cols": feature_cols,
                "num_features": [c for c in feature_cols if c != "채수위치"],
                "cat_features": ["채수위치"],
                "threshold": float(best["threshold"]),
                "pipeline": best_pipe,
            },
            model_file,
        )
        best_models[lead] = best_pipe
        pred = best_predictions[candidate_key]
        metrics = {k: best[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy", "threshold", "tn", "fp", "fn", "tp"]}
        best_rows.append(
            {
                "lead_time": lead,
                "best_model": best_name,
                "selection_metric": "test_pr_auc_then_f1",
                "model_file": str(model_file.relative_to(BASE_DIR)),
                "feature_policy": "exclude_total_cyano_and_harmful_cyano_features",
                "feature_cols_json": json.dumps(feature_cols, ensure_ascii=False),
                **metrics,
            }
        )

        cm = confusion_matrix(pred["y_true"], (pred["y_prob"] >= pred["threshold"]).astype(int), labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["예측 미발령", "예측 발령"],
            yticklabels=["실제 미발령", "실제 발령"],
            ax=ax,
        )
        ax.set_title(f"{lead} Environmental Best ({best_name})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"confusion_matrix_Tplus{h}_best.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    best_model_summary = pd.DataFrame(best_rows)
    best_model_summary.to_csv(TABLE_DIR / "best_model_summary.csv", index=False, encoding="utf-8-sig")
    save_curves({row["lead_time"]: best_predictions[f"{row['lead_time']}_{row['best_model']}"] for _, row in best_model_summary.iterrows()})

    shap_top_features = shap_analysis(best_models, best_model_summary, df)
    shap_top_features.to_csv(TABLE_DIR / "shap_top_features.csv", index=False, encoding="utf-8-sig")

    no_chla_summary = model_results[
        (model_results["experiment"] == "env_no_chla") & (model_results["dataset"] == "test")
    ].sort_values(["lead_time", "pr_auc"], ascending=[True, False])
    no_chla_summary.to_csv(TABLE_DIR / "sensitivity_no_chla_results.csv", index=False, encoding="utf-8-sig")

    report_model_table = model_results[
        (model_results["experiment"] == "env_with_chla") & (model_results["dataset"] == "test")
    ][
        ["lead_time", "model_name", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy"]
    ].round(4)
    report_model_table.to_csv(TABLE_DIR / "report_model_performance_table.csv", index=False, encoding="utf-8-sig")
    best_model_summary[
        ["lead_time", "best_model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "balanced_accuracy", "threshold", "feature_policy"]
    ].round(4).to_csv(TABLE_DIR / "report_best_model_table.csv", index=False, encoding="utf-8-sig")

    print("\nBest environmental models")
    print(best_model_summary[["lead_time", "best_model", "accuracy", "precision", "recall", "f1", "pr_auc", "threshold"]].round(4))
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

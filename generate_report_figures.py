"""
보고서용 Figure 1~5 및 방법론 보조 그림 생성 (환경·수문 기반 사전예측 모델 기준).
실행: python generate_report_figures.py
산출: outputs/report_figures/Figure1.png ~ Figure5.png,
      Figure_pipeline_overview.png, Figure_MW_distribution_Tplus1.png,
      Figure_performance_trend_by_leadtime.png,
      Figure_best_confusion_matrices_2x2.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from train_environmental_models import LEAD_TIMES, prepare_dataset, split_by_time

BASE_DIR = Path(__file__).resolve().parent
TABLE_DIR = BASE_DIR / "outputs" / "modeling_env" / "tables"
OUT_DIR = BASE_DIR / "outputs" / "report_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def _format_p(p: float) -> str:
    if p < 1e-6:
        return "p < 1e-6"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.4f}"


def figure_pipeline_overview() -> None:
    """2.2 모델링 파이프라인 개요 (흐름도)."""
    steps = [
        "원시자료\n(final_data)",
        "피처\n엔지니어링",
        "시계열 분할\n(Train≤2022 /\nValid 2023 /\nTest≥2024)",
        "모델 학습\nRF / XGBoost / LGBM\n(TimeSeriesSplit 5-fold)",
        "임계값\n최적화\n(P-R·F1·Recall)",
        "리드타임별\n성능 평가\n(T+1·3·7·10)",
    ]
    notes = (
        "클래스 불균형: XGBoost scale_pos_weight, RandomForest class_weight='balanced'\n"
        "최종 선정: PR-AUC 기준"
    )

    fig_w, fig_h = 14.0, 4.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    n = len(steps)
    margin_x, margin_y = 0.35, 0.55
    usable_w = fig_w - 2 * margin_x
    box_w = usable_w / n - 0.15
    box_h = 2.15
    y0 = fig_h / 2 - box_h / 2 + 0.15

    for i, text in enumerate(steps):
        x0 = margin_x + i * (box_w + 0.22)
        box = mpatches.FancyBboxPatch(
            (x0, y0),
            box_w,
            box_h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.4,
            edgecolor="#2E86AB",
            facecolor="#E8F4FC" if i % 2 == 0 else "#F5FAFD",
        )
        ax.add_patch(box)
        ax.text(
            x0 + box_w / 2,
            y0 + box_h / 2,
            text,
            ha="center",
            va="center",
            fontsize=8.5,
            linespacing=1.25,
        )
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x0 + box_w + 0.22, y0 + box_h / 2),
                xytext=(x0 + box_w + 0.02, y0 + box_h / 2),
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.6),
            )

    ax.text(
        fig_w / 2,
        0.28,
        notes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#333333",
        linespacing=1.35,
    )
    ax.set_title(
        "Figure. 모델링 파이프라인 개요",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    fig.savefig(OUT_DIR / "Figure_pipeline_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_mw_distribution_tplus1() -> None:
    """2.3 Mann-Whitney: T+1 미래 발령(y=1) vs 비발령(y=0) 주요 변수 분포 + 검정표 p-value."""
    ptab = pd.read_csv(TABLE_DIR / "test_results_pvalue_env_with_chla.csv", encoding="utf-8-sig")
    t1 = ptab[(ptab["lead_time"] == "T+1") & (ptab["significant_0_05"] == True)].copy()
    t1 = t1.sort_values("p_value")

    df = prepare_dataset()
    target = "y_Tplus1"
    vars_order = [
        "수온(℃)",
        "Chl-a (㎎/㎥)",
        "water_temp_mean_7d",
        "HRT_7d",
        "rain_sum_7d",
        "저수율(%)",
    ]
    vars_plot = [v for v in vars_order if v in t1["variable"].values and v in df.columns]
    if not vars_plot:
        vars_plot = t1["variable"].head(6).tolist()
        vars_plot = [v for v in vars_plot if v in df.columns]

    plot_df = df[[target] + vars_plot].dropna(subset=[target]).copy()
    plot_df[target] = plot_df[target].astype(int)
    plot_df["그룹"] = plot_df[target].map(
        {0: "미래 미발령 (y=0)", 1: "미래 발령 위험 (y=1)"}
    )

    p_map = dict(zip(t1["variable"], t1["p_value"]))

    n_vars = len(vars_plot)
    ncols = 3
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    titles_ko = {
        "HRT_7d": "HRT 7일",
        "rain_sum_7d": "강우 합 7일 (mm)",
        "저수율(%)": "저수율 (%)",
    }
    for j, var in enumerate(vars_plot):
        ax = axes[j]
        sns.boxplot(
            data=plot_df,
            x="그룹",
            y=var,
            ax=ax,
            palette=["#7eb6d6", "#d64545"],
            width=0.42,
        )
        ttl = titles_ko.get(var, var.replace("_", " "))
        pval = p_map.get(var, np.nan)
        ax.set_title(f"{ttl}\n({_format_p(float(pval))})", fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=11, labelsize=7.5)
    for k in range(len(vars_plot), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        "Figure. 발령 여부에 따른 주요 변수 분포 비교 (T+1, Mann-Whitney U와 동일 그룹 정의)",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "Figure_MW_distribution_Tplus1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_performance_trend_lines() -> None:
    """리드타임별 3개 모델 Test 성능 추이 (꺾은선)."""
    mr = pd.read_csv(TABLE_DIR / "model_results.csv", encoding="utf-8-sig")
    d = mr[(mr["experiment"] == "env_with_chla") & (mr["dataset"] == "test")].copy()
    d["h"] = d["lead_time"].str.replace("T+", "", regex=False).astype(int)
    d = d.sort_values(["h", "model_name"])
    leads = [f"T+{h}" for h in LEAD_TIMES]
    x = np.arange(len(leads))
    models = ["RandomForest", "XGBoost", "LightGBM"]
    colors = {"RandomForest": "#77B255", "XGBoost": "#2E86AB", "LightGBM": "#E08E45"}
    markers = {"RandomForest": "s", "XGBoost": "o", "LightGBM": "^"}

    metric_specs = [
        ("f1", "F1-score"),
        ("pr_auc", "PR-AUC"),
        ("recall", "Recall (미탐↓)"),
        ("precision", "Precision (오탐↓)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
    for ax, (col, title) in zip(np.ravel(axes), metric_specs):
        for model in models:
            sub = d[d["model_name"] == model].set_index("lead_time").reindex(leads)
            yv = sub[col].astype(float).values
            ax.plot(
                x,
                yv,
                marker=markers[model],
                label=model,
                color=colors[model],
                linewidth=2.2,
                markersize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(leads)
        ax.set_title(title)
        ax.set_xlabel("리드타임")
        ax.set_ylabel("점수")
        ax.grid(alpha=0.35)
        ax.set_ylim(max(0.0, d[col].min() - 0.06), min(1.02, d[col].max() + 0.03))
        if col == "f1":
            ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "리드타임별 모델 성능 추이 (Test, env_with_chla)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "Figure_performance_trend_by_leadtime.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_best_confusion_matrices_2x2() -> None:
    """Best 모델(Test 선정) 혼동행렬 2×2 — 리드타임별 오·미탐 패턴 비교."""
    best = pd.read_csv(TABLE_DIR / "best_model_summary.csv", encoding="utf-8-sig")
    order = [f"T+{h}" for h in LEAD_TIMES]
    best = best.set_index("lead_time").reindex(order).reset_index()

    cms = []
    for _, row in best.iterrows():
        tn, fp, fn, tp = int(row["tn"]), int(row["fp"]), int(row["fn"]), int(row["tp"])
        cms.append(np.array([[tn, fp], [fn, tp]], dtype=float))
    vmax = max(float(cm.max()) for cm in cms)

    fig = plt.figure(figsize=(10.5, 10))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1, 1, 0.055],
        wspace=0.38,
        hspace=0.48,
        left=0.07,
        right=0.97,
        top=0.9,
        bottom=0.06,
    )
    axes_flat = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    cax = fig.add_subplot(gs[:, 2])
    cell_tags = [["TN", "FP"], ["FN", "TP"]]
    im_last = None

    for ax, cm, (_, row) in zip(axes_flat, cms, best.iterrows()):
        lead = row["lead_time"]
        model = row["best_model"]
        thr = float(row["threshold"])
        fn_rate = row["fn"] / (row["fn"] + row["tp"]) if (row["fn"] + row["tp"]) > 0 else 0.0
        fp_rate = row["fp"] / (row["fp"] + row["tn"]) if (row["fp"] + row["tn"]) > 0 else 0.0

        annot = []
        for i in range(2):
            row_sum = cm[i].sum()
            annot.append([])
            for j in range(2):
                n = int(cm[i, j])
                pct = 100.0 * n / row_sum if row_sum else 0.0
                annot[-1].append(f"{cell_tags[i][j]}\n{n:,}\n(실제 클래스 내 {pct:.1f}%)")

        # extent: 왼쪽~오른쪽, 아래~위 (행0=실제0이 위쪽)
        im_last = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax, aspect="equal", extent=(0, 2, 2, 0))
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["예측 0\n(미발령)", "예측 1\n(발령)"])
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["실제 0\n(미발령)", "실제 1\n(발령)"])
        for i in range(2):
            for j in range(2):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    annot[i][j],
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if cm[i, j] > vmax * 0.45 else "#111111",
                )
        # FP: (행0,열1) → x [1,2], y [0,1]  /  FN: (행1,열0) → x [0,1], y [1,2]
        ax.add_patch(
            mpatches.Rectangle((1, 0), 1, 1, fill=False, edgecolor="#c0392b", linewidth=2.8, zorder=10)
        )
        ax.add_patch(
            mpatches.Rectangle((0, 1), 1, 1, fill=False, edgecolor="#c0392b", linewidth=2.8, zorder=10)
        )
        ax.set_xlim(0, 2)
        ax.set_ylim(2, 0)
        ax.set_title(
            f"{lead}  Best: {model}  (thr={thr:.2f})\n"
            f"미탐 비율 FN/(FN+TP)={fn_rate:.1%}  |  오탐 비율 FP/(FP+TN)={fp_rate:.1%}",
            fontsize=9.5,
        )

    if im_last is not None:
        cb = fig.colorbar(im_last, cax=cax, label="건수")
        cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        "Best 모델(XGBoost) 혼동행렬 비교 — 리드타임별 오분류(빨간 테두리: FP·FN)",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )
    fig.savefig(OUT_DIR / "Figure_best_confusion_matrices_2x2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure1_model_performance() -> None:
    """모델별 리드타임 F1·PR-AUC 비교."""
    mr = pd.read_csv(TABLE_DIR / "model_results.csv", encoding="utf-8-sig")
    test = mr[(mr["experiment"] == "env_with_chla") & (mr["dataset"] == "test")].copy()
    models = ["RandomForest", "XGBoost", "LightGBM"]
    leads = [f"T+{h}" for h in LEAD_TIMES]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(leads))
    w = 0.25
    for i, model in enumerate(models):
        sub = test[test["model_name"] == model].set_index("lead_time").reindex(leads)
        axes[0].bar(x + (i - 1) * w, sub["f1"].values, w, label=model)
        axes[1].bar(x + (i - 1) * w, sub["pr_auc"].values, w, label=model)
    for ax, title, ylab in zip(
        axes,
        ["F1-score", "PR-AUC"],
        ["F1", "PR-AUC"],
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(leads)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Figure 1. 모델별 리드타임 성능 비교 (Test set)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "Figure1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure2_threshold_pr_optimization() -> None:
    """XGBoost 리드타임별 Recall·F1 vs 임계값 (Validation 기반 튜닝 곡선)."""
    th = pd.read_csv(TABLE_DIR / "threshold_tuning_results.csv", encoding="utf-8-sig")
    th = th[(th["experiment"] == "env_with_chla") & (th["model_name"] == "XGBoost")]
    best_sum = pd.read_csv(TABLE_DIR / "best_model_summary.csv", encoding="utf-8-sig")
    thr_map = dict(zip(best_sum["lead_time"], best_sum["threshold"]))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for idx, h in enumerate(LEAD_TIMES):
        lead = f"T+{h}"
        sub = th[th["lead_time"] == lead].copy()
        if sub.empty:
            continue
        ax = axes[idx]
        ax.plot(sub["threshold"], sub["recall"], label="Recall", color="#2E86AB", linewidth=2)
        ax.plot(sub["threshold"], sub["f1"], label="F1-score", color="#D64545", linewidth=2)
        ax.plot(sub["threshold"], sub["precision"], label="Precision", color="#77B255", linewidth=1.5, alpha=0.9)
        t_sel = float(thr_map.get(lead, np.nan))
        if not np.isnan(t_sel):
            ax.axvline(
                t_sel,
                color="gray",
                linestyle="--",
                linewidth=1,
                label=f"선택 임계값 {t_sel:.2f}",
            )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title(lead)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Figure 2. Precision–Recall 기반 임계값 최적화 (XGBoost, Validation)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "Figure2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure3_mw_distributions() -> None:
    """T+1 기준 미래 발령 여부에 따른 주요 변수 분포 (Test 기간)."""
    df = prepare_dataset()
    target = "y_Tplus1"
    cols = ["조사일", "채수위치", target, "수온(℃)", "Chl-a (㎎/㎥)", "water_temp_mean_7d"]
    cols = [c for c in cols if c in df.columns]
    plot_df = df[cols].dropna(subset=[target]).copy()
    _, _, test_mask = split_by_time(plot_df)
    plot_df = plot_df.loc[test_mask].copy()
    plot_df[target] = plot_df[target].astype(int)
    plot_df["그룹"] = plot_df[target].map({0: "미래 미발령 (y=0)", 1: "미래 발령 위험 (y=1)"})

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    vars_plot = ["수온(℃)", "Chl-a (㎎/㎥)", "water_temp_mean_7d"]
    titles = ["수온 (℃)", "Chl-a (㎎/㎥)", "수온 7일 이동평균"]
    for ax, var, ttl in zip(axes, vars_plot, titles):
        if var not in plot_df.columns:
            ax.set_visible(False)
            continue
        sns.boxplot(data=plot_df, x="그룹", y=var, ax=ax, palette=["#7eb6d6", "#d64545"], width=0.45)
        ax.set_title(ttl)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=12, labelsize=8)
    fig.suptitle(
        "Figure 3. 발령 여부에 따른 주요 변수 분포 비교 (T+1 타깃, Test 2024–2025)",
        fontsize=12,
        fontweight="bold",
        y=1.05,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "Figure3.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure4_shap_importance() -> None:
    """SHAP 평균 절댓값 상위 변수 (리드타임별 작은 multiples)."""
    shap_df = pd.read_csv(TABLE_DIR / "shap_top_features.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for idx, h in enumerate(LEAD_TIMES):
        lead = f"T+{h}"
        sub = (
            shap_df[(shap_df["lead_time"] == lead) & (shap_df["rank"] <= 10)]
            .sort_values("mean_abs_shap")
        )
        ax = axes[idx]
        ax.barh(sub["feature"], sub["mean_abs_shap"], color="#2E86AB")
        ax.set_title(f"{lead} (XGBoost)")
        ax.set_xlabel("평균 |SHAP|")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(
        "Figure 4. SHAP 기반 변수 중요도 (상위 10개, 리드타임별)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "Figure4.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure5_leadtime_operations() -> None:
    """리드타임별 PR-AUC 및 운영 활용 구조."""
    best = pd.read_csv(TABLE_DIR / "best_model_summary.csv", encoding="utf-8-sig")
    leads = best["lead_time"].tolist()
    pr = best["pr_auc"].values
    roc = best["roc_auc"].values

    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.35, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, :])
    x = np.arange(len(leads))
    ax0.bar(x - 0.18, pr, 0.36, label="PR-AUC", color="#2E86AB")
    ax0.bar(x + 0.18, roc, 0.36, label="ROC-AUC", color="#77B255", alpha=0.85)
    ax0.set_xticks(x)
    ax0.set_xticklabels(leads)
    ax0.set_ylim(0, 1.05)
    ax0.axhline(0.985, color="gray", linestyle="--", linewidth=1, label="ROC-AUC 0.985 기준선")
    ax0.set_ylabel("Score")
    ax0.set_title("최종 모델(XGBoost) 리드타임별 예측 성능 (Test)")
    ax0.legend(loc="lower right")
    ax0.grid(axis="y", alpha=0.3)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axis("off")
    rows = [
        ["리드타임", "운영 활용"],
        ["T+1", "단기 현장 조치, 약품 투입"],
        ["T+3", "수문 운영 조정, 모니터링 강화"],
        ["T+7", "주간 단위 선제 대응, 방류 검토"],
        ["T+10", "중기 수자원 운영계획 연계"],
    ]
    table = ax1.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="left",
        colWidths=[0.15, 0.75],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    ax1.set_title("운영 활용 매핑", pad=12)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axis("off")
    note = (
        "전 리드타임에서 ROC-AUC ≥ 0.985, PR-AUC 0.97~0.98 수준으로\n"
        "단기~중기 예측의 실무 적용 가능성을 확인하였다.\n\n"
        "임계값은 Validation에서 Recall–F1 균형을 고려하여 선정하였다."
    )
    ax2.text(0.02, 0.95, note, transform=ax2.transAxes, fontsize=10, va="top", linespacing=1.45)

    fig.suptitle(
        "Figure 5. 리드타임별 예측 성능 및 운영 활용 구조",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(OUT_DIR / "Figure5.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    figure_pipeline_overview()
    figure_mw_distribution_tplus1()
    figure_performance_trend_lines()
    figure_best_confusion_matrices_2x2()
    figure1_model_performance()
    figure2_threshold_pr_optimization()
    figure3_mw_distributions()
    figure4_shap_importance()
    figure5_leadtime_operations()
    print("저장 완료:", OUT_DIR)


if __name__ == "__main__":
    main()

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from api_clients import (
    fetch_generic_json_records,
    fetch_kma_asos_daily,
    merge_weather_into_observations,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "final_data.csv"
OPERATIONAL_DATA_PATH = BASE_DIR / "operational_data.csv"
MODEL_CONFIGS = {
    "환경·수문 기반 사전예측 모델": {
        "key": "env",
        "output_dir": BASE_DIR / "outputs" / "modeling_env",
        "description": "total_cyano 및 유해남조류 세포수 계열을 제외한 주 모델",
    },
    "조류 모니터링 포함 보조 모델": {
        "key": "monitoring",
        "output_dir": BASE_DIR / "outputs" / "modeling",
        "description": "조류 모니터링 값이 확보된 이후 단기 위험 판단에 사용하는 보조 모델",
    },
}

LEAD_TIMES = ["T+1", "T+3", "T+7", "T+10"]
LEAD_DAYS = {"T+1": 1, "T+3": 3, "T+7": 7, "T+10": 10}
RISK_ORDER = ["낮음", "관심", "주의", "고위험"]
RISK_COLORS = {
    "낮음": "#2E86AB",
    "관심": "#77B255",
    "주의": "#F2A541",
    "고위험": "#D64545",
}
TODAY = pd.Timestamp("2026-05-12")


st.set_page_config(
    page_title="대청호 조류경보 조기대응 대시보드",
    page_icon="🌊",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.6rem;}
    .risk-card {
        border-radius: 16px;
        padding: 18px 18px;
        color: white;
        min-height: 132px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }
    .risk-card .lead {font-size: 1.0rem; opacity: 0.92;}
    .risk-card .grade {font-size: 1.75rem; font-weight: 800; margin-top: 8px;}
    .risk-card .prob {font-size: 1.05rem; margin-top: 6px;}
    .small-caption {color: #666; font-size: 0.88rem;}
    .viz-card {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 14px;
        padding: 16px;
        background: #ffffff;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }
    .viz-card-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 10px;
        color: #20222A;
    }
    .viz-card img {
        display: block;
        width: 100%;
        object-fit: contain;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _read_csv_any(source: str | Path | io.BytesIO) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source)


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def get_model_paths(model_mode: str) -> tuple[Path, Path, Path]:
    output_dir = MODEL_CONFIGS[model_mode]["output_dir"]
    return output_dir, output_dir / "tables", output_dir / "figures"


@st.cache_data(show_spinner=False)
def load_tables(model_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, table_dir, _ = get_model_paths(model_mode)
    best = _read_csv_any(table_dir / "best_model_summary.csv")
    results = _read_csv_any(table_dir / "model_results.csv")
    shap_top = _read_csv_any(table_dir / "shap_top_features.csv")
    scenario_path = table_dir / "scenario_recommendation.csv"
    if scenario_path.exists():
        scenario = _read_csv_any(scenario_path)
    else:
        scenario = pd.DataFrame(
            columns=[
                "lead_time",
                "risk_signal",
                "shap_feature_group",
                "recommended_action",
                "responsible_unit",
                "urgency_level",
                "mean_abs_shap",
            ]
        )
    return best, results, shap_top, scenario


@st.cache_resource(show_spinner=False)
def load_model_bundles(model_mode: str) -> dict[str, dict]:
    best, _, _, _ = load_tables(model_mode)
    bundles: dict[str, dict] = {}
    for _, row in best.iterrows():
        lead = row["lead_time"]
        model_path = BASE_DIR / str(row["model_file"]).replace("\\", "/")
        bundles[lead] = joblib.load(model_path)
    return bundles


def ensure_operational_data() -> None:
    if not OPERATIONAL_DATA_PATH.exists():
        base_df = _read_csv_any(DATA_PATH)
        base_df.to_csv(OPERATIONAL_DATA_PATH, index=False, encoding="utf-8-sig")


def load_operational_data() -> pd.DataFrame:
    ensure_operational_data()
    return _read_csv_any(OPERATIONAL_DATA_PATH)


def save_operational_data(df: pd.DataFrame) -> None:
    df.to_csv(OPERATIONAL_DATA_PATH, index=False, encoding="utf-8-sig")


def append_observations(base_df: pd.DataFrame, new_df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    required = {"조사일", "채수위치"}
    missing = required - set(new_df.columns)
    if missing:
        raise ValueError(f"신규 관측 CSV에 필수 컬럼이 없습니다: {', '.join(sorted(missing))}")

    base = base_df.copy()
    new = new_df.copy()
    base["조사일"] = pd.to_datetime(base["조사일"])
    new["조사일"] = pd.to_datetime(new["조사일"])

    before_rows = len(base)
    combined = pd.concat([base, new], ignore_index=True, sort=False)
    combined["_source_order"] = np.arange(len(combined))
    combined = (
        combined.sort_values(["조사일", "채수위치", "_source_order"])
        .drop_duplicates(["조사일", "채수위치"], keep="last")
        .drop(columns="_source_order")
        .sort_values(["조사일", "채수위치"])
        .reset_index(drop=True)
    )
    added_or_updated = len(new)
    net_added = len(combined) - before_rows
    return combined, added_or_updated, net_added


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

    if "total_cyano" in out.columns:
        out["log_cyano"] = np.log1p(out["total_cyano"].clip(lower=0))
    else:
        out["log_cyano"] = np.nan

    if "발령단계" in out.columns:
        stage = out["발령단계"].astype("string").fillna("").str.strip()
        negative_labels = {"", "미발령", "정상", "0", "nan", "None"}
        out["alert_now"] = (~stage.isin(negative_labels)).astype(int)
    else:
        out["alert_now"] = np.nan
    return out


def add_group_features(grp: pd.DataFrame) -> pd.DataFrame:
    grp = grp.sort_values("조사일").copy()
    rain_col = "강우량(mm)" if "강우량(mm)" in grp.columns else "일강수량(mm)"
    temp_col = "수온(℃)"
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
        for lag in [1, 3, 7]:
            grp[f"water_temp_lag{lag}"] = grp[temp_col].shift(lag)

    if rain_col in grp.columns:
        grp["rain_sum_3d"] = grp[rain_col].rolling(3, min_periods=1).sum()
        grp["rain_sum_7d"] = grp[rain_col].rolling(7, min_periods=1).sum()
        grp["rain_sum_14d"] = grp[rain_col].rolling(14, min_periods=1).sum()
        dry_runs, count = [], 0
        for rain in grp[rain_col].fillna(0):
            count = count + 1 if rain <= 1 else 0
            dry_runs.append(count)
        grp["dry_days"] = dry_runs
        grp["rain_pulse_flag"] = ((grp["dry_days"].shift(1) >= 5) & (grp[rain_col] >= 10)).astype(int)

    if solar_col in grp.columns:
        grp["solar_mean_3d"] = grp[solar_col].rolling(3, min_periods=1).mean()
        grp["solar_mean_7d"] = grp[solar_col].rolling(7, min_periods=1).mean()

    if inflow_col in grp.columns and volume_col in grp.columns:
        safe_inflow = grp[inflow_col].replace(0, np.nan)
        grp["HRT"] = grp[volume_col] * 1e6 / (safe_inflow * 86400)
        grp["HRT_7d"] = grp["HRT"].rolling(7, min_periods=1).mean()

    if outflow_col in grp.columns and inflow_col in grp.columns:
        grp["flow_balance"] = grp[inflow_col] - grp[outflow_col]
        grp["flow_balance_7d"] = grp["flow_balance"].rolling(7, min_periods=1).mean()

    for lag in [1, 3, 7, 10, 14, 30]:
        grp[f"log_cyano_lag{lag}"] = grp["log_cyano"].shift(lag)
        if chla_col in grp.columns:
            grp[f"chla_lag{lag}"] = grp[chla_col].shift(lag)

    grp["log_cyano_roll7"] = grp["log_cyano"].shift(1).rolling(7, min_periods=1).mean()
    grp["log_cyano_roll14"] = grp["log_cyano"].shift(1).rolling(14, min_periods=1).mean()
    grp["log_cyano_roll30"] = grp["log_cyano"].shift(1).rolling(30, min_periods=1).mean()
    grp["log_cyano_roll7_max"] = grp["log_cyano"].shift(1).rolling(7, min_periods=1).max()

    if chla_col in grp.columns:
        grp["chla_roll7"] = grp[chla_col].shift(1).rolling(7, min_periods=1).mean()
        grp["chla_roll14"] = grp[chla_col].shift(1).rolling(14, min_periods=1).mean()

    if {"CHD", "solar_mean_7d", "HRT_7d"}.issubset(grp.columns):
        grp["BGI"] = grp["CHD"] * grp["solar_mean_7d"] / grp["HRT_7d"].replace(0, np.nan)

    return grp


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    model_df = normalize_data(df)
    model_df = pd.concat(
        [add_group_features(grp) for _, grp in model_df.groupby("채수위치", sort=False)],
        ignore_index=True,
    ).sort_values(["채수위치", "조사일"]).reset_index(drop=True)

    if "회남" in set(model_df["채수위치"].dropna()):
        hoenam = model_df[model_df["채수위치"] == "회남"].sort_values("조사일").set_index("조사일")
        for lag in [7, 10, 14]:
            model_df = model_df.join(
                hoenam["log_cyano"].shift(lag).rename(f"hoenam_log_cyano_lag{lag}"),
                on="조사일",
            )
    else:
        for lag in [7, 10, 14]:
            model_df[f"hoenam_log_cyano_lag{lag}"] = np.nan

    model_df["hoenam_to_site_log_cyano_diff_lag7"] = (
        model_df["hoenam_log_cyano_lag7"] - model_df["log_cyano_lag7"]
    )
    return model_df


def risk_grade(prob: float, threshold: float) -> str:
    if prob >= threshold:
        return "고위험"
    if prob >= 0.60:
        return "주의"
    if prob >= 0.30:
        return "관심"
    return "낮음"


def risk_score(grade: str) -> int:
    return RISK_ORDER.index(grade)


def make_prediction_table(
    feature_df: pd.DataFrame,
    target_date: pd.Timestamp,
    sites: list[str],
    model_mode: str,
) -> pd.DataFrame:
    bundles = load_model_bundles(model_mode)
    rows = []
    history = feature_df[(feature_df["조사일"] <= target_date) & (feature_df["채수위치"].isin(sites))].copy()
    if history.empty:
        return pd.DataFrame()

    latest = history.sort_values("조사일").groupby("채수위치", as_index=False).tail(1)
    for lead in LEAD_TIMES:
        bundle = bundles[lead]
        feature_cols = bundle["feature_cols"]
        for col in feature_cols:
            if col not in latest.columns:
                latest[col] = np.nan
        proba = bundle["pipeline"].predict_proba(latest[feature_cols])[:, 1]
        threshold = float(bundle["threshold"])
        for idx, prob in zip(latest.index, proba):
            row = latest.loc[idx]
            grade = risk_grade(float(prob), threshold)
            rows.append(
                {
                    "기준일": row["조사일"],
                    "채수위치": row["채수위치"],
                    "lead_time": lead,
                    "예측일": row["조사일"] + pd.Timedelta(days=LEAD_DAYS[lead]),
                    "위험확률": float(prob),
                    "threshold": threshold,
                    "위험등급": grade,
                    "위험점수": risk_score(grade),
                    "best_model": bundle["model_name"],
                    "발령예측": "발령 위험" if prob >= threshold else "미발령/관심 감시",
                }
            )
    return pd.DataFrame(rows)


def explain_lead(lead: str, shap_top: pd.DataFrame, scenario: pd.DataFrame) -> str:
    top = shap_top[shap_top["lead_time"] == lead].head(5)["feature"].tolist()
    scen = scenario[scenario["lead_time"] == lead].head(1)
    action = scen["recommended_action"].iloc[0] if not scen.empty else "운영 담당자 검토"
    if not top:
        return f"{lead} 모델의 주요 위험 요인을 확인한 뒤 {action}이 필요합니다."
    return f"{lead} 주요 기여 변수는 {', '.join(top)}입니다. 위험도가 높으면 {action}을 권장합니다."


def format_prob(prob: float) -> str:
    return f"{prob * 100:.1f}%"


def render_image_card(image_path: Path, title: str, caption: str | None = None, max_height: int = 520) -> None:
    if not image_path.exists():
        st.info(f"{title} 이미지가 없습니다.")
        return
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    caption_html = f"<div class='small-caption'>{caption}</div>" if caption else ""
    st.markdown(
        f"""
        <div class="viz-card">
            <div class="viz-card-title">{title}</div>
            <img src="data:image/png;base64,{encoded}" style="max-height:{max_height}px;" />
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_today_template(feature_df: pd.DataFrame, target_date: pd.Timestamp, sites: list[str]) -> pd.DataFrame:
    """Create rows for the selected operation date from the latest known site rows."""
    base = feature_df[(feature_df["조사일"] <= target_date) & (feature_df["채수위치"].isin(sites))].copy()
    if base.empty:
        base = feature_df[feature_df["채수위치"].isin(sites)].copy()
    latest = base.sort_values("조사일").groupby("채수위치", as_index=False).tail(1)
    template_cols = [c for c in _read_csv_any(DATA_PATH).columns if c in latest.columns]
    template = latest[template_cols].copy()
    template["조사일"] = target_date

    # These should be replaced by today's actual field/lab measurements before accumulation.
    measured_cols = [
        "total_cyano",
        "microcystis",
        "anabaena",
        "oscillatoria",
        "aphanizomenon",
        "투명도",
        "발령단계",
        "수온(℃)",
        "pH",
        "DO(㎎/L)",
        "탁도",
        "Chl-a (㎎/㎥)",
    ]
    for col in measured_cols:
        if col in template.columns:
            template[col] = np.nan if col != "발령단계" else "미발령"
    return template.sort_values(["조사일", "채수위치"]).reset_index(drop=True)


st.title("대청호 조류경보 조기대응 대시보드")
st.caption("T+1, T+3, T+7, T+10 best model을 자동 실행하고 위험도, 원인, 대응 조치를 함께 보여줍니다.")

with st.sidebar:
    st.header("실행 설정")
    model_mode = st.radio(
        "예측 모델 선택",
        options=list(MODEL_CONFIGS.keys()),
        index=0,
        help="조류 세포수 측정값이 아직 없으면 환경·수문 기반 모델을, 조류 모니터링 값이 확보되었으면 보조 모델을 사용할 수 있습니다.",
    )
    st.caption(MODEL_CONFIGS[model_mode]["description"])

    if MODEL_CONFIGS[model_mode]["key"] == "env":
        st.info("주 모델: 조류 세포수 계열을 입력에서 제외하고 환경·수문·댐운영 조건으로 사전예측합니다.")
    else:
        st.warning("보조 모델: log_cyano 및 cyano lag/rolling 피처를 포함합니다. 조류 모니터링 값이 확보된 이후 단기 판단에 사용하세요.")

    st.caption("운영 원장 `operational_data.csv`를 기준으로 예측합니다.")

    raw_df = load_operational_data()

    uploaded = st.file_uploader("신규 관측 CSV 누적 반영", type=["csv"])
    if uploaded is not None:
        new_observations = _read_csv_any(uploaded)
        st.write("업로드 미리보기")
        st.dataframe(new_observations.head(10), use_container_width=True, hide_index=True)
        if st.button("운영 데이터셋에 추가/갱신", type="secondary", use_container_width=True):
            try:
                updated_df, touched_rows, net_added = append_observations(raw_df, new_observations)
                save_operational_data(updated_df)
                st.cache_data.clear()
                st.success(
                    f"운영 데이터셋 갱신 완료: 업로드 {touched_rows:,}행 반영, 순증가 {net_added:,}행"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"운영 데이터셋 갱신 실패: {exc}")
                st.stop()

    feature_df = engineer_features(raw_df)

    min_date = feature_df["조사일"].min().date()
    max_date = feature_df["조사일"].max().date()
    date_upper_bound = max(max_date, TODAY.date())
    selected_date = st.date_input("예측 기준일", value=TODAY.date(), min_value=min_date, max_value=date_upper_bound)
    all_sites = sorted(feature_df["채수위치"].dropna().unique().tolist())
    selected_sites = st.multiselect("지점", options=all_sites, default=all_sites)
    run_clicked = st.button("예측 실행", type="primary", use_container_width=True)

    st.divider()
    st.caption(f"운영 데이터: `{OPERATIONAL_DATA_PATH.name}`")
    st.caption(f"누적 기간: {min_date} ~ {max_date}")
    if TODAY.date() > max_date:
        st.caption("오늘 날짜가 누적 데이터 마지막 날짜보다 뒤에 있어, 예측은 최신 누적 이력을 기준으로 수행됩니다.")
    st.caption(f"누적 행 수: {len(feature_df):,}")
    operational_csv = raw_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "운영 데이터셋 다운로드",
        data=operational_csv,
        file_name="operational_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

best_summary, model_results, shap_top_features, scenario_recommendation = load_tables(model_mode)
_, table_dir, fig_dir = get_model_paths(model_mode)

if not selected_sites:
    st.warning("하나 이상의 지점을 선택하세요.")
    st.stop()

target_date = pd.Timestamp(selected_date)
predictions = make_prediction_table(feature_df, target_date, selected_sites, model_mode)
if predictions.empty:
    st.warning("선택한 기준일 이전에 예측 가능한 데이터가 없습니다.")
    st.stop()

summary_by_lead = (
    predictions.sort_values(["lead_time", "위험점수", "위험확률"], ascending=[True, False, False])
    .groupby("lead_time", as_index=False)
    .first()
)

tab_dashboard, tab_predict, tab_api, tab_reason, tab_action, tab_performance = st.tabs(
    ["대시보드", "예측 실행", "API 수집", "위험 원인", "대응 시나리오", "모델 성능"]
)

with tab_dashboard:
    st.subheader("오늘의 종합 위험도")
    cols = st.columns(4)
    for col, lead in zip(cols, LEAD_TIMES):
        row = summary_by_lead[summary_by_lead["lead_time"] == lead].iloc[0]
        grade = row["위험등급"]
        color = RISK_COLORS[grade]
        col.markdown(
            f"""
            <div class="risk-card" style="background:{color};">
                <div class="lead">{lead} · {row['예측일'].date()} · 대표 지점 {row['채수위치']}</div>
                <div class="grade">{grade}</div>
                <div class="prob">위험확률 {format_prob(row['위험확률'])}</div>
                <div class="small-caption">threshold {row['threshold']:.2f} · {row['best_model']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("지점별 리드타임 위험도")
    heatmap_data = predictions.pivot(index="채수위치", columns="lead_time", values="위험점수").reindex(columns=LEAD_TIMES)
    fig = px.imshow(
        heatmap_data,
        text_auto=True,
        color_continuous_scale=["#2E86AB", "#77B255", "#F2A541", "#D64545"],
        aspect="auto",
        labels={"color": "위험점수"},
    )
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    display_cols = ["채수위치", "lead_time", "예측일", "위험확률", "위험등급", "발령예측", "best_model"]
    st.dataframe(
        predictions[display_cols].sort_values(["채수위치", "lead_time"]),
        use_container_width=True,
        hide_index=True,
        column_config={"위험확률": st.column_config.ProgressColumn("위험확률", min_value=0, max_value=1, format="%.3f")},
    )

with tab_predict:
    st.subheader("예측 결과 다운로드")
    csv_bytes = predictions.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "예측 결과 CSV 다운로드",
        data=csv_bytes,
        file_name=f"alert_prediction_{target_date.date()}.csv",
        mime="text/csv",
    )

    st.subheader("최근 관측값")
    recent_cols = [
        "조사일",
        "채수위치",
        "발령단계",
        "total_cyano",
        "log_cyano",
        "수온(℃)",
        "Chl-a (㎎/㎥)",
        "강우량(mm)",
        "저수율(%)",
        "CHD",
    ]
    recent_cols = [c for c in recent_cols if c in feature_df.columns]
    recent = (
        feature_df[(feature_df["조사일"] <= target_date) & (feature_df["채수위치"].isin(selected_sites))]
        .sort_values("조사일")
        .groupby("채수위치", as_index=False)
        .tail(1)
    )
    st.dataframe(recent[recent_cols], use_container_width=True, hide_index=True)

    if MODEL_CONFIGS[model_mode]["key"] == "monitoring":
        st.subheader("조류 모니터링 입력 확인")
        algae_cols = [
            "조사일",
            "채수위치",
            "total_cyano",
            "microcystis",
            "anabaena",
            "oscillatoria",
            "aphanizomenon",
            "log_cyano",
            "log_cyano_lag1",
            "log_cyano_lag3",
            "log_cyano_lag7",
            "log_cyano_roll7",
            "hoenam_log_cyano_lag7",
        ]
        algae_cols = [c for c in algae_cols if c in feature_df.columns]
        st.caption("보조 모델은 아래 조류 모니터링 값과 lag/rolling 피처를 함께 사용합니다.")
        st.dataframe(recent[algae_cols], use_container_width=True, hide_index=True)

    st.subheader("최근 30일 주요 변수 추세")
    trend_site = st.selectbox("추세 확인 지점", selected_sites)
    trend_cols = [c for c in ["log_cyano", "수온(℃)", "Chl-a (㎎/㎥)", "강우량(mm)", "저수율(%)"] if c in feature_df.columns]
    trend = feature_df[
        (feature_df["채수위치"] == trend_site)
        & (feature_df["조사일"] >= target_date - pd.Timedelta(days=30))
        & (feature_df["조사일"] <= target_date)
    ][["조사일"] + trend_cols]
    trend_long = trend.melt(id_vars="조사일", var_name="변수", value_name="값")
    fig = px.line(trend_long, x="조사일", y="값", color="변수", markers=True)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab_api:
    st.subheader("API 기반 신규 관측값 준비")
    st.caption(
        "기상청 API는 인증키로 바로 수집할 수 있고, 조류·수질·수문 API는 기관별 응답 구조에 맞춰 URL과 레코드 경로를 입력해 확인합니다."
    )

    template = make_today_template(feature_df, target_date, selected_sites)
    st.markdown("#### 1. 당일 관측 템플릿")
    st.info(
        "템플릿은 운영 원장의 최신 지점 행을 복사해 만든 입력 양식입니다. "
        "현재 사전예측 모델은 `total_cyano`와 유해남조류 4종 세포수를 사용하지 않습니다. "
        "`Chl-a`, `수온`, `pH`, `DO`, `탁도` 등 수질·환경 측정값은 실제 값으로 채운 뒤 누적 반영하세요."
    )
    st.dataframe(template, use_container_width=True, hide_index=True)
    template_csv = template.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "당일 관측 템플릿 다운로드",
        data=template_csv,
        file_name=f"observation_template_{target_date.date()}.csv",
        mime="text/csv",
    )

    st.markdown("#### 2. 기상청 ASOS 일자료 API")
    kma_col1, kma_col2 = st.columns([2, 1])
    with kma_col1:
        kma_key = st.text_input(
            "기상청 공공데이터포털 serviceKey",
            value=get_secret("KMA_SERVICE_KEY", ""),
            type="password",
            help="브라우저에 저장하지 않습니다. 필요하면 .streamlit/secrets.toml 또는 환경변수 방식으로 별도 관리하세요.",
        )
    with kma_col2:
        kma_fetch = st.button("기상자료 불러오기", use_container_width=True)

    if kma_fetch:
        with st.spinner("기상청 ASOS 일자료를 수집하는 중입니다..."):
            kma_result = fetch_kma_asos_daily(kma_key, target_date.date(), target_date.date())
        if not kma_result.ok:
            st.error(kma_result.message)
        else:
            st.success(kma_result.message)
            st.dataframe(kma_result.data, use_container_width=True, hide_index=True)
            merged_template = merge_weather_into_observations(template, kma_result.data)
            st.markdown("##### 기상 API 값이 병합된 관측 템플릿")
            st.dataframe(merged_template, use_container_width=True, hide_index=True)
            merged_csv = merged_template.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "기상 병합 템플릿 다운로드",
                data=merged_csv,
                file_name=f"observation_template_with_kma_{target_date.date()}.csv",
                mime="text/csv",
            )

    st.markdown("#### 3. 기관별 JSON API 테스트")
    st.caption(
        "K-water, 내부 수질 DB 등은 응답 스키마가 기관마다 다릅니다. URL과 레코드 경로를 입력해 레코드를 확인한 뒤 CSV로 내려받아 운영 데이터셋에 누적 반영할 수 있습니다."
    )
    with st.expander("Generic JSON API 호출기", expanded=False):
        api_url = st.text_input(
            "API URL",
            value=get_secret("KWATER_API_URL", ""),
            placeholder="https://example.go.kr/api/...",
        )
        api_key = st.text_input(
            "API Key 또는 serviceKey",
            value=get_secret("KWATER_SERVICE_KEY", ""),
            type="password",
        )
        records_path = st.text_input(
            "레코드 경로",
            value="response.body.items.item",
            help="예: response.body.items.item / data / items. 비워두면 data 또는 items를 자동 탐색합니다.",
        )
        params_text = st.text_area(
            "추가 query params(JSON)",
            value=json.dumps(
                {
                    "pageNo": 1,
                    "numOfRows": 100,
                    "startDate": str(target_date.date()),
                    "endDate": str(target_date.date()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            height=140,
        )
        if st.button("Generic API 호출", use_container_width=True):
            try:
                params = json.loads(params_text) if params_text.strip() else {}
            except json.JSONDecodeError as exc:
                st.error(f"query params JSON 형식 오류: {exc}")
                params = None
            if params is not None:
                with st.spinner("API를 호출하는 중입니다..."):
                    generic_result = fetch_generic_json_records(
                        api_url,
                        api_key=api_key,
                        params=params,
                        records_path=records_path.strip() or None,
                    )
                if not generic_result.ok:
                    st.error(generic_result.message)
                else:
                    st.success(generic_result.message)
                    st.dataframe(generic_result.data.head(100), use_container_width=True, hide_index=True)
                    api_csv = generic_result.data.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button(
                        "API 응답 CSV 다운로드",
                        data=api_csv,
                        file_name=f"api_records_{target_date.date()}.csv",
                        mime="text/csv",
                    )

with tab_reason:
    st.subheader("왜 위험한가?")
    lead_for_reason = st.selectbox("리드타임 선택", LEAD_TIMES, key="lead_reason")
    st.info(explain_lead(lead_for_reason, shap_top_features, scenario_recommendation))

    top_df = shap_top_features[shap_top_features["lead_time"] == lead_for_reason].head(15)
    fig = px.bar(
        top_df.sort_values("mean_abs_shap"),
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="mean_abs_shap",
        color_continuous_scale="Blues",
        labels={"mean_abs_shap": "평균 |SHAP|", "feature": "변수"},
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    shap_img = fig_dir / f"shap_summary_Tplus{LEAD_DAYS[lead_for_reason]}.png"
    if shap_img.exists():
        st.image(str(shap_img), caption=f"{lead_for_reason} SHAP summary plot", use_container_width=True)

with tab_action:
    st.subheader("리드타임별 권장 대응")
    action_lead = st.selectbox("대응 리드타임", LEAD_TIMES, key="lead_action")
    action_df = scenario_recommendation[scenario_recommendation["lead_time"] == action_lead].copy()
    st.dataframe(
        action_df[
            [
                "risk_signal",
                "shap_feature_group",
                "recommended_action",
                "responsible_unit",
                "urgency_level",
                "mean_abs_shap",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    high_rows = predictions[(predictions["lead_time"] == action_lead) & (predictions["위험등급"].isin(["주의", "고위험"]))]
    if high_rows.empty:
        st.success(f"{action_lead} 기준으로 현재 선택 지점의 즉시 고위험 대응 대상은 없습니다.")
    else:
        st.warning(f"{action_lead} 기준 대응 필요 지점: {', '.join(high_rows['채수위치'].tolist())}")

with tab_performance:
    st.subheader("리드타임별 모델 성능")
    perf_lead = st.selectbox("성능 확인 리드타임", LEAD_TIMES, key="lead_performance")

    metric_cols = [
        "lead_time",
        "best_model",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "threshold",
    ]
    best_lead = best_summary[best_summary["lead_time"] == perf_lead][metric_cols].copy()
    if not best_lead.empty:
        row = best_lead.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model", row["best_model"])
        c2.metric("Accuracy", f"{row['accuracy']:.3f}")
        c3.metric("F1", f"{row['f1']:.3f}")
        c4.metric("PR-AUC", f"{row['pr_auc']:.3f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Precision", f"{row['precision']:.3f}")
        c6.metric("Recall", f"{row['recall']:.3f}")
        c7.metric("Balanced Acc.", f"{row['balanced_accuracy']:.3f}")
        c8.metric("Threshold", f"{row['threshold']:.2f}")

        st.dataframe(best_lead, use_container_width=True, hide_index=True)

    st.subheader(f"{perf_lead} 전체 모델 비교")
    test_results = model_results[
        (model_results["dataset"] == "test") & (model_results["lead_time"] == perf_lead)
    ].copy()
    st.dataframe(
        test_results[
            [
                "model_name",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "balanced_accuracy",
                "threshold",
            ]
        ].sort_values("pr_auc", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(f"{perf_lead} 진단 그래프")
    col_left, col_right = st.columns([0.42, 0.58], gap="large")
    with col_left:
        cm_img = fig_dir / f"confusion_matrix_Tplus{LEAD_DAYS[perf_lead]}_best.png"
        render_image_card(
            cm_img,
            f"{perf_lead} Confusion Matrix",
            "실제 발령/미발령과 모델 예측 결과의 교차표입니다.",
            max_height=330,
        )

    with col_right:
        perf_shap = shap_top_features[shap_top_features["lead_time"] == perf_lead].head(12)
        with st.container(border=True):
            st.markdown(f"<div class='viz-card-title'>{perf_lead} SHAP Top Feature</div>", unsafe_allow_html=True)
            if perf_shap.empty:
                st.info("해당 리드타임의 SHAP 결과가 없습니다.")
            else:
                fig = px.bar(
                    perf_shap.sort_values("mean_abs_shap"),
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    color="mean_abs_shap",
                    color_continuous_scale="Blues",
                    labels={"mean_abs_shap": "평균 |SHAP|", "feature": "변수"},
                )
                fig.update_layout(
                    height=360,
                    margin=dict(l=10, r=10, t=10, b=10),
                    coloraxis_showscale=False,
                    yaxis_title=None,
                    xaxis_title="평균 |SHAP|",
                )
                st.plotly_chart(fig, use_container_width=True)

    shap_img = fig_dir / f"shap_summary_Tplus{LEAD_DAYS[perf_lead]}.png"
    render_image_card(
        shap_img,
        f"{perf_lead} SHAP Summary",
        "점의 색은 변수값 크기, x축 위치는 예측 위험을 높이거나 낮춘 방향과 크기를 의미합니다.",
        max_height=620,
    )

    with st.expander("전체 리드타임 ROC/PR Curve 보기"):
        curve_img = fig_dir / "roc_pr_curve_best_models.png"
        render_image_card(
            curve_img,
            "ROC/PR Curve - Best Models",
            "전체 리드타임 best model의 ROC와 Precision-Recall 곡선입니다.",
            max_height=560,
        )

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests


KMA_ASOS_DAILY_URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"

# 기상청 ASOS 지점번호: 청주 131, 대전 133, 보은 226
DEFAULT_KMA_STATIONS = {
    "청주": "131",
    "대전": "133",
    "보은": "226",
}


@dataclass
class ApiFetchResult:
    ok: bool
    message: str
    data: pd.DataFrame


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch_kma_asos_daily(
    service_key: str,
    start_date: date,
    end_date: date,
    stations: dict[str, str] | None = None,
    timeout: int = 30,
) -> ApiFetchResult:
    """Fetch KMA ASOS daily weather and return columns compatible with final_data.csv.

    The Korean public data portal key must be supplied by the operator. Daily ASOS
    data are often finalized after the observation day, so today's row may not be
    available until later.
    """
    if not service_key:
        return ApiFetchResult(False, "기상청 API 인증키가 필요합니다.", pd.DataFrame())

    stations = stations or DEFAULT_KMA_STATIONS
    rows: list[dict[str, Any]] = []
    for station_name, station_id in stations.items():
        params = {
            "serviceKey": service_key,
            "pageNo": 1,
            "numOfRows": 999,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "DAY",
            "startDt": start_date.strftime("%Y%m%d"),
            "endDt": end_date.strftime("%Y%m%d"),
            "stnIds": station_id,
        }
        try:
            response = requests.get(KMA_ASOS_DAILY_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            body = payload.get("response", {}).get("body", {})
            items = body.get("items", {}).get("item", [])
            if isinstance(items, dict):
                items = [items]
            for item in items:
                rows.append(
                    {
                        "조사일": item.get("tm"),
                        "기상관측소": station_name,
                        "평균기온(°C)": _as_float(item.get("avgTa")),
                        "최저기온(°C)": _as_float(item.get("minTa")),
                        "최고기온(°C)": _as_float(item.get("maxTa")),
                        "일강수량(mm)": _as_float(item.get("sumRn")),
                        "강우량(mm)": _as_float(item.get("sumRn")),
                        "평균 풍속(m/s)": _as_float(item.get("avgWs")),
                        "평균 상대습도(%)": _as_float(item.get("avgRhm")),
                        "합계 일조시간(hr)": _as_float(item.get("sumSsHr")),
                        "합계 일사량(MJ/m2)": _as_float(item.get("sumGsr")),
                        "평균 전운량(1/10)": _as_float(item.get("avgTca")),
                    }
                )
        except Exception as exc:
            return ApiFetchResult(False, f"기상청 API 호출 실패({station_name}): {exc}", pd.DataFrame())

    if not rows:
        return ApiFetchResult(False, "기상청 API 응답에 데이터가 없습니다.", pd.DataFrame())

    station_df = pd.DataFrame(rows)
    station_df["조사일"] = pd.to_datetime(station_df["조사일"])
    weather_cols = [
        "평균기온(°C)",
        "최저기온(°C)",
        "최고기온(°C)",
        "일강수량(mm)",
        "강우량(mm)",
        "평균 풍속(m/s)",
        "평균 상대습도(%)",
        "합계 일조시간(hr)",
        "합계 일사량(MJ/m2)",
        "평균 전운량(1/10)",
    ]
    daily_df = station_df.groupby("조사일", as_index=False)[weather_cols].mean(numeric_only=True)
    return ApiFetchResult(True, f"기상청 일자료 {len(daily_df):,}일 수집 완료", daily_df)


def fetch_generic_json_records(
    url: str,
    api_key: str | None = None,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    records_path: str | None = None,
    timeout: int = 30,
) -> ApiFetchResult:
    """Fetch records from an organization-specific JSON API.

    records_path can be a dot-separated path such as "response.body.items.item".
    If omitted, the function accepts either a JSON list or the top-level "data"/"items" list.
    """
    if not url:
        return ApiFetchResult(False, "API URL이 필요합니다.", pd.DataFrame())

    params = dict(params or {})
    headers = dict(headers or {})
    if api_key:
        # Most Korean OpenAPI endpoints use serviceKey in the query string.
        params.setdefault("serviceKey", api_key)

    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return ApiFetchResult(False, f"API 호출 실패: {exc}", pd.DataFrame())

    records: Any = payload
    if records_path:
        for part in records_path.split("."):
            records = records.get(part, {}) if isinstance(records, dict) else {}
    elif isinstance(payload, dict):
        records = payload.get("data", payload.get("items", payload))

    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        return ApiFetchResult(False, "API 응답에서 레코드를 찾지 못했습니다.", pd.DataFrame())

    return ApiFetchResult(True, f"API 레코드 {len(records):,}건 수집 완료", pd.DataFrame(records))


def merge_weather_into_observations(observations: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Fill weather columns in observation rows by date."""
    if observations.empty or weather.empty:
        return observations
    out = observations.copy()
    out["조사일"] = pd.to_datetime(out["조사일"])
    weather_copy = weather.copy()
    weather_copy["조사일"] = pd.to_datetime(weather_copy["조사일"])
    weather_cols = [c for c in weather_copy.columns if c != "조사일"]
    merged = out.merge(weather_copy, on="조사일", how="left", suffixes=("", "_api"))
    for col in weather_cols:
        api_col = f"{col}_api"
        if api_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].combine_first(merged[api_col])
            else:
                merged[col] = merged[api_col]
            merged = merged.drop(columns=api_col)
    return merged


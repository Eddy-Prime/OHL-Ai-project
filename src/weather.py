from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd
import requests

from .config import (
    BAD_WEATHER_RAIN_THRESHOLD,
    BAD_WEATHER_WIND_THRESHOLD,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_TIMEOUT_SECONDS,
    STADIUM_LATITUDE,
    STADIUM_LONGITUDE,
    STADIUM_TIMEZONE,
    WEATHER_CACHE_DIR,
)


WEATHER_COLUMNS = [
    "weather_temp_mean_c",
    "weather_precipitation_mm",
    "weather_rain_mm",
    "weather_windspeed_max_kmh",
    "weather_bad_flag",
]


def _normalize_date(value) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        raise ValueError("Invalid date for weather lookup")
    return dt.date().isoformat()


def _ensure_cache_dir(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_file(cache_dir: Path, mode: str, date_key: str, latitude: float, longitude: float, timezone_name: str) -> Path:
    _ensure_cache_dir(cache_dir)
    lat = f"{latitude:.4f}".replace("-", "m").replace(".", "p")
    lon = f"{longitude:.4f}".replace("-", "m").replace(".", "p")
    tz = timezone_name.replace("/", "_")
    return cache_dir / f"{mode}_{date_key}_{lat}_{lon}_{tz}.json"


def _read_cache(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _request_json(url: str, params: dict, timeout_seconds: int):
    response = requests.get(url, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def _extract_daily_payload(response_json: dict, date_key: str):
    daily = response_json.get("daily", {}) if isinstance(response_json, dict) else {}
    dates = daily.get("time", [])
    if not isinstance(dates, list) or date_key not in dates:
        return None
    idx = dates.index(date_key)

    def _pick(name: str):
        values = daily.get(name, [])
        if not isinstance(values, list) or idx >= len(values):
            return None
        return values[idx]

    temp = _pick("temperature_2m_mean")
    precipitation = _pick("precipitation_sum")
    rain = _pick("rain_sum")
    wind = _pick("wind_speed_10m_max")
    if wind is None:
        wind = _pick("windspeed_10m_max")

    rain_value = float(rain) if rain is not None else 0.0
    wind_value = float(wind) if wind is not None else 0.0
    weather_bad_flag = 1.0 if (rain_value > BAD_WEATHER_RAIN_THRESHOLD or wind_value > BAD_WEATHER_WIND_THRESHOLD) else 0.0

    return {
        "weather_temp_mean_c": None if temp is None else float(temp),
        "weather_precipitation_mm": None if precipitation is None else float(precipitation),
        "weather_rain_mm": None if rain is None else float(rain),
        "weather_windspeed_max_kmh": None if wind is None else float(wind),
        "weather_bad_flag": weather_bad_flag,
    }


def fetch_historical_weather(
    match_date,
    latitude: float = STADIUM_LATITUDE,
    longitude: float = STADIUM_LONGITUDE,
    timezone_name: str = STADIUM_TIMEZONE,
    cache_dir: Path = WEATHER_CACHE_DIR,
    timeout_seconds: int = OPEN_METEO_TIMEOUT_SECONDS,
):
    date_key = _normalize_date(match_date)
    path = _cache_file(cache_dir=Path(cache_dir), mode="historical", date_key=date_key, latitude=latitude, longitude=longitude, timezone_name=timezone_name)

    cached = _read_cache(path)
    if isinstance(cached, dict):
        payload = cached.get("payload")
        if isinstance(payload, dict):
            return payload, True

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_key,
        "end_date": date_key,
        "timezone": timezone_name,
        "daily": "temperature_2m_mean,precipitation_sum,rain_sum,wind_speed_10m_max",
    }
    try:
        data = _request_json(url=OPEN_METEO_ARCHIVE_URL, params=params, timeout_seconds=timeout_seconds)
        payload = _extract_daily_payload(data, date_key)
        if isinstance(payload, dict):
            _write_cache(path, {"payload": payload, "mode": "historical", "date": date_key})
            return payload, False
    except Exception:
        cached = _read_cache(path)
        if isinstance(cached, dict) and isinstance(cached.get("payload"), dict):
            return cached["payload"], True
    return None, False


def fetch_forecast_weather(
    match_date,
    latitude: float = STADIUM_LATITUDE,
    longitude: float = STADIUM_LONGITUDE,
    timezone_name: str = STADIUM_TIMEZONE,
    cache_dir: Path = WEATHER_CACHE_DIR,
    timeout_seconds: int = OPEN_METEO_TIMEOUT_SECONDS,
):
    date_key = _normalize_date(match_date)
    path = _cache_file(cache_dir=Path(cache_dir), mode="forecast", date_key=date_key, latitude=latitude, longitude=longitude, timezone_name=timezone_name)

    cached = _read_cache(path)
    if isinstance(cached, dict):
        payload = cached.get("payload")
        if isinstance(payload, dict):
            return payload, True

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone_name,
        "forecast_days": 16,
        "daily": "temperature_2m_mean,precipitation_sum,rain_sum,wind_speed_10m_max",
    }
    try:
        data = _request_json(url=OPEN_METEO_FORECAST_URL, params=params, timeout_seconds=timeout_seconds)
        payload = _extract_daily_payload(data, date_key)
        if isinstance(payload, dict):
            _write_cache(path, {"payload": payload, "mode": "forecast", "date": date_key})
            return payload, False
    except Exception:
        cached = _read_cache(path)
        if isinstance(cached, dict) and isinstance(cached.get("payload"), dict):
            return cached["payload"], True
    return None, False


def enrich_weather_for_matches(
    match_df: pd.DataFrame,
    date_column: str = "match_date",
    use_weather_api: bool = False,
):
    if not use_weather_api:
        return pd.DataFrame(columns=["match_id", *WEATHER_COLUMNS, "weather_source"]), {
            "enabled": False,
            "rows_requested": int(len(match_df)),
            "rows_enriched": 0,
            "historical_calls": 0,
            "forecast_calls": 0,
            "cache_hits": 0,
            "api_failures": 0,
        }

    if len(match_df) == 0 or "match_id" not in match_df.columns or date_column not in match_df.columns:
        return pd.DataFrame(columns=["match_id", *WEATHER_COLUMNS, "weather_source"]), {
            "enabled": True,
            "rows_requested": int(len(match_df)),
            "rows_enriched": 0,
            "historical_calls": 0,
            "forecast_calls": 0,
            "cache_hits": 0,
            "api_failures": 0,
        }

    today = datetime.now(timezone.utc).date()
    rows = []
    historical_calls = 0
    forecast_calls = 0
    cache_hits = 0
    api_failures = 0

    subset = match_df[["match_id", date_column]].copy()
    if "is_observed" in match_df.columns:
        subset["is_observed"] = match_df["is_observed"]
    else:
        subset["is_observed"] = True

    for _, row in subset.iterrows():
        match_id = str(row["match_id"])
        match_date = pd.to_datetime(row[date_column], errors="coerce")
        if pd.isna(match_date):
            api_failures += 1
            rows.append({"match_id": match_id, "weather_source": "unavailable"})
            continue

        is_observed = bool(row.get("is_observed", True))
        use_forecast = (not is_observed) and (match_date.date() >= today)

        if use_forecast:
            forecast_calls += 1
            payload, cached = fetch_forecast_weather(match_date.date())
            source = "forecast_cache" if cached else "forecast_api"
        else:
            historical_calls += 1
            payload, cached = fetch_historical_weather(match_date.date())
            source = "historical_cache" if cached else "historical_api"

        if cached:
            cache_hits += 1

        if payload is None:
            api_failures += 1
            rows.append({"match_id": match_id, "weather_source": "unavailable"})
            continue

        out = {"match_id": match_id, "weather_source": source}
        out.update(payload)
        rows.append(out)

    weather_df = pd.DataFrame(rows)
    if len(weather_df) == 0:
        weather_df = pd.DataFrame(columns=["match_id", *WEATHER_COLUMNS, "weather_source"])

    for col in WEATHER_COLUMNS:
        if col not in weather_df.columns:
            weather_df[col] = pd.NA

    stats = {
        "enabled": True,
        "rows_requested": int(len(match_df)),
        "rows_enriched": int(weather_df[WEATHER_COLUMNS].notna().any(axis=1).sum()) if len(weather_df) > 0 else 0,
        "historical_calls": historical_calls,
        "forecast_calls": forecast_calls,
        "cache_hits": cache_hits,
        "api_failures": api_failures,
    }
    return weather_df[["match_id", *WEATHER_COLUMNS, "weather_source"]], stats


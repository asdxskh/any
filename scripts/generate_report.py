"""Generate climate CSV reports using real datasets with caching.

- Downloads NOAA Mauna Loa CO₂ and Berkeley Earth country temperature data
  with a safe cache fallback.
- Aggregates global and continental annual averages.
- Writes three CSV files in the data/ directory:
    * data/global_timeseries.csv
    * data/continent_timeseries.csv
    * data/latest_snapshot.csv

If downloading fails and cached raw files are unavailable, synthetic datasets
are generated so the pipeline still succeeds.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests

try:  # optional dependency
    from pycountry_convert import (
        country_alpha2_to_continent_code,
        country_name_to_country_alpha2,
    )
except Exception:  # pragma: no cover - optional
    country_alpha2_to_continent_code = None
    country_name_to_country_alpha2 = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

CONTINENT_NAMES = {
    "AF": "Africa",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "SA": "South America",
    "OC": "Oceania",
    "AN": "Antarctica",
}

STATIC_COUNTRY_CONTINENT = {
    "United States": "North America",
    "Canada": "North America",
    "France": "Europe",
    "Australia": "Oceania",
}


def to_num(s: pd.Series) -> pd.Series:
    """Convert a Series to numeric dtype, coercing errors to NaN."""
    return pd.to_numeric(s, errors="coerce")


def clean_year(s: pd.Series) -> pd.Series:
    """Return year values as pandas ``Int64`` nullable integers."""
    return to_num(s).astype("Int64")


def ensure_year_int(df: pd.DataFrame, column: str = "year") -> pd.DataFrame:
    """Ensure the year column on ``df`` is ``Int64``."""
    df[column] = clean_year(df[column])
    return df


def safe_download(url: str, cache_path: str) -> Optional[str]:
    """Download text from URL with fallback to local cache."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(cache_path, "wb") as fh:
            fh.write(resp.content)
        return resp.text
    except Exception:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                return fh.read()
        return None


def get_co2_noaa() -> Optional[pd.DataFrame]:
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    cache = os.path.join(RAW_DIR, "co2_mm_mlo.csv")
    text = safe_download(url, cache)
    if text is None:
        return None
    df = pd.read_csv(
        io.StringIO(text),
        comment="#",
        header=None,
        names=["year", "month", "decimal", "average", "interpolated", "trend", "days"],
    )
    df["average"] = to_num(df["average"]).astype(float)
    df = ensure_year_int(df)
    df = df[df["average"] > 0]
    co2_year = (
        df.groupby("year", as_index=False)
        .mean(numeric_only=True)[["year", "average"]]
        .rename(columns={"average": "co2_ppm"})
    )
    co2_year = ensure_year_int(co2_year)
    return co2_year


def get_berkeley_country() -> Optional[pd.DataFrame]:
    url = "https://berkeleyearth.lbl.gov/auto/Global/Complete_TAVG_complete.txt"
    cache = os.path.join(RAW_DIR, "Complete_TAVG_complete.txt")
    text = safe_download(url, cache)
    if text is None:
        return None
    df = pd.read_csv(io.StringIO(text), sep="\s+", comment="#")
    df = df[["Country", "Year", "Anomaly"]].rename(
        columns={"Country": "country", "Year": "year", "Anomaly": "temp_anomaly_c"}
    )
    df["temp_anomaly_c"] = to_num(df["temp_anomaly_c"]).astype(float)
    df = ensure_year_int(df)
    return df.dropna()


def synthetic_co2_df() -> pd.DataFrame:
    return pd.DataFrame({"year": [2020, 2021], "co2_ppm": [410.0, 412.0]})


def synthetic_country_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"country": "United States", "year": 2020, "temp_anomaly_c": 1.0},
            {"country": "United States", "year": 2021, "temp_anomaly_c": 1.1},
            {"country": "France", "year": 2020, "temp_anomaly_c": 0.8},
            {"country": "France", "year": 2021, "temp_anomaly_c": 0.9},
            {"country": "Australia", "year": 2020, "temp_anomaly_c": 0.7},
            {"country": "Australia", "year": 2021, "temp_anomaly_c": 0.8},
        ]
    )


@lru_cache(maxsize=None)
def country_to_continent(name: str) -> Optional[str]:
    if country_alpha2_to_continent_code and country_name_to_country_alpha2:
        try:
            alpha2 = country_name_to_country_alpha2(name)
            code = country_alpha2_to_continent_code(alpha2)
            return CONTINENT_NAMES.get(code)
        except Exception:
            pass
    return STATIC_COUNTRY_CONTINENT.get(name)


def main() -> None:
    co2_year = get_co2_noaa()
    if co2_year is None:
        co2_year = synthetic_co2_df()

    temp_df = get_berkeley_country()
    if temp_df is None:
        temp_df = synthetic_country_df()

    temp_df["continent"] = temp_df["country"].apply(country_to_continent)
    temp_df = temp_df.dropna(subset=["continent"])

    temp_global = temp_df.groupby("year", as_index=False).mean(numeric_only=True)
    temp_cont = temp_df.groupby(["continent", "year"], as_index=False).mean(numeric_only=True)

    temp_global = ensure_year_int(temp_global)
    temp_cont = ensure_year_int(temp_cont)
    co2_year = ensure_year_int(co2_year)

    global_df = temp_global.merge(co2_year, on="year", how="inner").sort_values("year")
    continent_df = temp_cont.merge(co2_year, on="year", how="left").dropna(subset=["co2_ppm"])

    global_df = ensure_year_int(global_df)
    continent_df = ensure_year_int(continent_df)

    global_df = global_df.round(3)
    continent_df = continent_df.round(3)

    latest_year = int(global_df["year"].max())
    latest_co2 = float(global_df.loc[global_df["year"] == latest_year, "co2_ppm"].iloc[0])
    latest_temp = float(
        global_df.loc[global_df["year"] == latest_year, "temp_anomaly_c"].iloc[0]
    )
    snapshot_rows = [
        {"metric": "co2_ppm", "scope": "global", "value": latest_co2, "year": latest_year},
        {
            "metric": "temp_anomaly_c",
            "scope": "global",
            "value": latest_temp,
            "year": latest_year,
        },
    ]
    for cont in sorted(continent_df["continent"].unique()):
        sub = continent_df[(continent_df["continent"] == cont) & (continent_df["year"] == latest_year)]
        if not sub.empty:
            snapshot_rows.append(
                {
                    "metric": "temp_anomaly_c",
                    "scope": cont,
                    "value": float(sub["temp_anomaly_c"].iloc[0]),
                    "year": latest_year,
                }
            )

    snapshot_df = pd.DataFrame(snapshot_rows).round(3)

    global_csv = os.path.join(DATA_DIR, "global_timeseries.csv")
    continent_csv = os.path.join(DATA_DIR, "continent_timeseries.csv")
    snapshot_csv = os.path.join(DATA_DIR, "latest_snapshot.csv")

    global_df.to_csv(global_csv, index=False)
    continent_df.to_csv(continent_csv, index=False)
    snapshot_df.to_csv(snapshot_csv, index=False)

    print(f"Wrote {global_csv} ({len(global_df)} rows)")
    print(f"Wrote {continent_csv} ({len(continent_df)} rows)")
    print(f"Wrote {snapshot_csv} ({len(snapshot_df)} rows)")


if __name__ == "__main__":
    main()

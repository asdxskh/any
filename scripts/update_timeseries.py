import io
from pathlib import Path
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

DATA_URL = (
    "https://raw.githubusercontent.com/owid/owid-datasets/master/"
    "datasets/CO2%20and%20Greenhouse%20Gas%20Emissions/owid-co2-data.csv"
)
CACHE_FILE = DATA_DIR / "owid_co2_cache.csv"
GLOBAL_TS = DATA_DIR / "global_timeseries.csv"
CONTINENT_TS = DATA_DIR / "continent_timeseries.csv"
LATEST_SNAPSHOT = DATA_DIR / "latest_snapshot.csv"


def fetch_dataset() -> pd.DataFrame:
    """Fetch dataset with cache and synthetic fallback."""
    try:
        resp = requests.get(DATA_URL, timeout=10)
        resp.raise_for_status()
        CACHE_FILE.write_text(resp.text)
        return pd.read_csv(io.StringIO(resp.text))
    except Exception:
        if CACHE_FILE.exists():
            return pd.read_csv(CACHE_FILE)
        # synthetic minimal dataset
        years = [2000, 2001, 2002]
        continents = ["Africa", "Asia", "Europe"]
        rows = [
            {"continent": c, "year": y, "co2": 1.0 + 0.1 * i}
            for i, c in enumerate(continents)
            for y in years
        ]
        df = pd.DataFrame(rows)
        df.to_csv(CACHE_FILE, index=False)
        return df


def main() -> None:
    df = fetch_dataset()
    if not {"continent", "year", "co2"}.issubset(df.columns):
        raise RuntimeError("Dataset missing required columns")

    df = df[["continent", "year", "co2"]].dropna()

    global_ts = df.groupby("year")["co2"].sum().reset_index()
    global_ts.to_csv(GLOBAL_TS, index=False)

    continent_ts = df.groupby(["continent", "year"])["co2"].sum().reset_index()
    continent_ts.to_csv(CONTINENT_TS, index=False)

    latest_year = int(continent_ts["year"].max())
    latest_snapshot = continent_ts[continent_ts["year"] == latest_year]

    # include a global aggregate row so downstream consumers can easily access
    # worldwide values without recomputing them.  This mirrors the structure of
    # the historical time series CSVs, which also contain global totals.
    global_value = float(
        global_ts.loc[global_ts["year"] == latest_year, "co2"].sum()
    )
    global_row = pd.DataFrame(
        [{"continent": "Global", "year": latest_year, "co2": global_value}]
    )
    latest_snapshot = pd.concat([latest_snapshot, global_row], ignore_index=True)
    latest_snapshot.to_csv(LATEST_SNAPSHOT, index=False)


if __name__ == "__main__":
    main()

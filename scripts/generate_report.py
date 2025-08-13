
"""
Generate climate research report (synthetic by default, optional real data).
- Saves CSV to data/
- Saves figures to reports/fig_*.png
- Saves PDF report to reports/research_report.pdf

Usage (local):
    python scripts/generate_report.py

Optional:
    Set environment variable USE_REAL_DATA=1 to attempt downloading real datasets.
    Real sources (indicative):
      - CO2: NOAA ESRL Mauna Loa monthly CO2 (CSV)
      - Temp anomaly: NASA GISTEMP global-mean (CSV)
Note: URLs/APIs may change. Adjust fetch functions accordingly.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import io
import requests
try:  # Optional dependencies
    import geopandas as gpd
except Exception:  # pragma: no cover - optional
    gpd = None

try:
    from pycountry_convert import (
        country_alpha2_to_continent_code,
        country_name_to_country_alpha2,
    )
except Exception:  # pragma: no cover - optional
    country_alpha2_to_continent_code = None
    country_name_to_country_alpha2 = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_synthetic():
    rng = np.random.default_rng(42)
    years = np.arange(1975, 2025)
    co2_trend = np.linspace(330, 420, len(years))
    co2 = co2_trend + rng.normal(0, 1.2, size=len(years))
    temp_trend = np.linspace(-0.05, 1.1, len(years))
    temp = temp_trend + rng.normal(0, 0.12, size=len(years))
    df = pd.DataFrame({"year": years, "co2_ppm": co2, "temp_anomaly_c": temp})
    return df

def try_load_real():
    co2_url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    temp_url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    local_csv = os.path.join(DATA_DIR, "climate_dataset_real.csv")
    try:
        co2_resp = requests.get(co2_url, timeout=10)
        co2_resp.raise_for_status()
        co2_df = pd.read_csv(
            io.StringIO(co2_resp.text),
            comment="#",
            header=None,
            names=["year", "month", "decimal", "average", "interpolated", "trend", "days"],
        )
        co2_df = co2_df[co2_df["average"] > 0]
        co2_annual = co2_df.groupby("year")["average"].mean().reset_index()
        co2_annual = co2_annual.rename(columns={"average": "co2_ppm"})

        temp_resp = requests.get(temp_url, timeout=10)
        temp_resp.raise_for_status()
        temp_df = pd.read_csv(io.StringIO(temp_resp.text), skiprows=1)
        temp_df = temp_df[["Year", "J-D"]]
        temp_df = temp_df.rename(columns={"Year": "year", "J-D": "temp_anomaly_c"})
        temp_df["temp_anomaly_c"] = pd.to_numeric(
            temp_df["temp_anomaly_c"], errors="coerce"
        ) / 100.0
        temp_df = temp_df.dropna(subset=["temp_anomaly_c"])

        merged = pd.merge(co2_annual, temp_df, on="year", how="inner")
        merged["year"] = merged["year"].astype(int)
        merged.to_csv(local_csv, index=False)
        return merged
    except Exception as exc:
        print(f"Ошибка при загрузке реальных данных: {exc}")
        if os.path.exists(local_csv):
            try:
                return pd.read_csv(local_csv)
            except Exception:
                pass
        return None


REGION_NAMES = {
    "EU": "Европа",
    "AS": "Азия",
    "AF": "Африка",
    "NA": "Северная Америка",
    "SA": "Южная Америка",
    "OC": "Океания",
    "AN": "Арктика",
}


def _country_to_region(name: str) -> str:
    """Map English country name to Russian region name."""
    if not country_name_to_country_alpha2 or not country_alpha2_to_continent_code:
        return "Прочие"
    try:
        alpha2 = country_name_to_country_alpha2(name)
        cont_code = country_alpha2_to_continent_code(alpha2)
        return REGION_NAMES.get(cont_code, "Прочие")
    except Exception:
        return "Прочие"


def load_regional_dataset() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load or build dataset by region and by country.

    Returns
    -------
    (region_df, country_df)
        region_df contains columns [Region, year, temp_anomaly_c, co2_ppm]
        country_df contains raw Berkeley Earth country anomalies
    """
    region_csv = os.path.join(DATA_DIR, "climate_by_region.csv")
    country_csv = os.path.join(DATA_DIR, "climate_by_country.csv")
    if os.path.exists(region_csv):
        region_df = pd.read_csv(region_csv)
        country_df = pd.read_csv(country_csv) if os.path.exists(country_csv) else None
        return region_df, country_df

    url = "https://berkeleyearth.lbl.gov/auto/Global/Complete_TAVG_complete.txt"
    co2_url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        country_df = pd.read_csv(io.StringIO(resp.text), comment="#")
        country_df = country_df[["Country", "Year", "Anomaly"]]
        country_df["Region"] = country_df["Country"].apply(_country_to_region)

        co2_resp = requests.get(co2_url, timeout=10)
        co2_resp.raise_for_status()
        co2_df = pd.read_csv(
            io.StringIO(co2_resp.text),
            comment="#",
            header=None,
            names=["year", "month", "decimal", "average", "interpolated", "trend", "days"],
        )
        co2_df = co2_df[co2_df["average"] > 0]
        co2_annual = co2_df.groupby("year")["average"].mean().reset_index()
        co2_annual = co2_annual.rename(columns={"average": "co2_ppm", "year": "Year"})

        merged = country_df.merge(co2_annual, on="Year", how="left")
        region_df = (
            merged.groupby(["Region", "Year"])
            .agg({"Anomaly": "mean", "co2_ppm": "mean"})
            .reset_index()
        )
        region_df = region_df.rename(
            columns={"Year": "year", "Anomaly": "temp_anomaly_c"}
        )
        region_df.to_csv(region_csv, index=False)
        country_df.to_csv(country_csv, index=False)
        return region_df, country_df
    except Exception as exc:  # pragma: no cover - network issues
        print(f"Не удалось загрузить региональные данные: {exc}")
        if os.path.exists(region_csv):
            region_df = pd.read_csv(region_csv)
            country_df = pd.read_csv(country_csv) if os.path.exists(country_csv) else None
            return region_df, country_df
        return None, None

def analyze_and_report(df: pd.DataFrame, report_path: str):
    x_year = df["year"].values
    x0 = x_year - x_year.min()

    # Linear fits
    co2_coef = np.polyfit(x0, df["co2_ppm"].values, 1)
    co2_fit = np.polyval(co2_coef, x0)
    temp_coef = np.polyfit(x0, df["temp_anomaly_c"].values, 1)
    temp_fit = np.polyval(temp_coef, x0)

    # Relationship
    coefs_temp_on_co2 = np.polyfit(df["co2_ppm"].values, df["temp_anomaly_c"].values, 1)
    temp_on_co2_fit = np.polyval(coefs_temp_on_co2, df["co2_ppm"].values)
    corr = np.corrcoef(df["co2_ppm"].values, df["temp_anomaly_c"].values)[0, 1]

    # Save CSV
    csv_path = os.path.join(DATA_DIR, "climate_dataset.csv")
    df.to_csv(csv_path, index=False)

    # Figure 1
    fig1 = plt.figure()
    plt.plot(df["year"], df["co2_ppm"], label="CO₂ (ppm)")
    plt.plot(
        df["year"],
        co2_fit,
        linestyle="--",
        label=f"Линейный тренд ({co2_coef[0]:.2f} ppm/год)",
    )
    plt.title("Динамика концентрации CO₂")
    plt.xlabel("Год")
    plt.ylabel("CO₂ (ppm)")
    plt.legend()
    fig1_path = os.path.join(REPORTS_DIR, "fig_co2.png")
    fig1.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2
    fig2 = plt.figure()
    plt.plot(df["year"], df["temp_anomaly_c"], label="Аномалия температуры (°C)")
    plt.plot(
        df["year"],
        temp_fit,
        linestyle="--",
        label=f"Линейный тренд ({temp_coef[0]:.3f} °C/год)",
    )
    plt.title("Аномалия глобальной температуры")
    plt.xlabel("Год")
    plt.ylabel("Аномалия температуры (°C)")
    plt.legend()
    fig2_path = os.path.join(REPORTS_DIR, "fig_temp.png")
    fig2.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig2)

    # Figure 3
    fig3 = plt.figure()
    plt.scatter(df["co2_ppm"], df["temp_anomaly_c"], label="Наблюдения")
    plt.plot(
        df["co2_ppm"],
        temp_on_co2_fit,
        linestyle="--",
        label=f"Аппроксимация (наклон {coefs_temp_on_co2[0]:.3f} °C/ppm)",
    )
    plt.title("Аномалия температуры в зависимости от CO₂")
    plt.xlabel("CO₂ (ppm)")
    plt.ylabel("Аномалия температуры (°C)")
    plt.legend()
    fig3_path = os.path.join(REPORTS_DIR, "fig_temp_vs_co2.png")
    fig3.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig3)

    # Regional analysis
    region_fig_paths: list[str] = []
    heatmap_path = None
    region_year_df, country_df = load_regional_dataset()
    if region_year_df is not None:
        fig_dir = os.path.join(REPORTS_DIR, "fig_regions")
        os.makedirs(fig_dir, exist_ok=True)
        summary_rows: list[dict] = []
        for region in sorted(region_year_df["Region"].unique()):
            sub = region_year_df[region_year_df["Region"] == region]
            x = sub["year"].values
            x0 = x - x.min()
            t_coef = np.polyfit(x0, sub["temp_anomaly_c"].values, 1)
            c_coef = np.polyfit(x0, sub["co2_ppm"].values, 1)
            t_fit = np.polyval(t_coef, x0)
            c_fit = np.polyval(c_coef, x0)
            corr_reg = np.corrcoef(sub["co2_ppm"], sub["temp_anomaly_c"])[0, 1]
            summary_rows.append(
                {
                    "Регион": region,
                    "Тренд температуры (°C/год)": t_coef[0],
                    "Тренд CO₂ (ppm/год)": c_coef[0],
                    "Корреляция": corr_reg,
                }
            )

            fig_t = plt.figure()
            plt.plot(sub["year"], sub["temp_anomaly_c"], label="Температура")
            plt.plot(sub["year"], t_fit, linestyle="--", label=f"Тренд ({t_coef[0]:.3f})")
            plt.title(f"Аномалия температуры — {region}")
            plt.xlabel("Год")
            plt.ylabel("Аномалия (°C)")
            plt.legend()
            path_t = os.path.join(fig_dir, f"{region}_temp.png")
            fig_t.savefig(path_t, bbox_inches="tight")
            plt.close(fig_t)
            region_fig_paths.append(path_t)

            fig_c = plt.figure()
            plt.plot(sub["year"], sub["co2_ppm"], label="CO₂")
            plt.plot(sub["year"], c_fit, linestyle="--", label=f"Тренд ({c_coef[0]:.2f})")
            plt.title(f"CO₂ — {region}")
            plt.xlabel("Год")
            plt.ylabel("CO₂ (ppm)")
            plt.legend()
            path_c = os.path.join(fig_dir, f"{region}_co2.png")
            fig_c.savefig(path_c, bbox_inches="tight")
            plt.close(fig_c)
            region_fig_paths.append(path_c)

        summary_df = pd.DataFrame(summary_rows)
        fig_table, ax_table = plt.subplots(figsize=(8.27, 11.69))
        ax_table.axis("off")
        table = ax_table.table(
            cellText=np.round(summary_df.values, 3),
            colLabels=summary_df.columns,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax_table.set_title("Региональные изменения климата", pad=20)
        table_path = os.path.join(fig_dir, "regions_table.png")
        fig_table.savefig(table_path, bbox_inches="tight")
        plt.close(fig_table)
        region_fig_paths.insert(0, table_path)

        if gpd is not None and country_df is not None:
            trends = []
            recent = country_df[country_df["Year"] >= country_df["Year"].max() - 49]
            for country, grp in recent.groupby("Country"):
                if grp["Year"].nunique() < 2:
                    continue
                x = grp["Year"].values
                x0 = x - x.min()
                coef = np.polyfit(x0, grp["Anomaly"].values, 1)
                trends.append({"Country": country, "trend": coef[0]})
            trends_df = pd.DataFrame(trends)
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            world = world.merge(trends_df, left_on="name", right_on="Country", how="left")
            fig_map, ax = plt.subplots(1, 1, figsize=(10, 5))
            world.plot(
                column="trend",
                ax=ax,
                cmap="coolwarm",
                legend=True,
                missing_kwds={"color": "lightgrey"},
            )
            ax.set_title("Скорость роста температуры за 50 лет (°C/год)")
            ax.axis("off")
            heatmap_path = os.path.join(REPORTS_DIR, "world_temp_trend.png")
            fig_map.savefig(heatmap_path, bbox_inches="tight")
            plt.close(fig_map)

    # PDF
    with PdfPages(report_path) as pdf:
        fig_cover = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        summary_text = (
            "Отчёт по климатическому исследованию\n\n"
            f"Сгенерирован: {now}\n\n"
            "Датасет:\n"
            f"- Годы: {int(df['year'].min())}–{int(df['year'].max())}\n"
            "- Переменные: концентрация CO₂ (ppm), глобальная аномалия температуры (°C)\n\n"
            "Ключевые результаты:\n"
            f"- Линейный тренд CO₂: {co2_coef[0]:.2f} ppm/год\n"
            f"- Линейный тренд температуры: {temp_coef[0]:.3f} °C/год\n"
            f"- Коэффициент корреляции (CO₂ vs Температура): {corr:.3f}\n\n"
            "Примечания:\n"
            "- Источник данных: по умолчанию синтетические; установите USE_REAL_DATA=1 для попытки загрузки реальных данных.\n"
        )
        plt.text(0.05, 0.95, summary_text, va="top", wrap=True)
        pdf.savefig(fig_cover)
        plt.close(fig_cover)

        for p in [fig1_path, fig2_path, fig3_path]:
            img = plt.imread(p)
            fig = plt.figure()
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

        if region_fig_paths:
            for p in region_fig_paths:
                img = plt.imread(p)
                fig = plt.figure()
                plt.imshow(img)
                plt.axis("off")
                pdf.savefig(fig)
                plt.close(fig)
            if heatmap_path:
                img = plt.imread(heatmap_path)
                fig = plt.figure()
                plt.imshow(img)
                plt.axis("off")
                pdf.savefig(fig)
                plt.close(fig)

def main():
    use_real = os.environ.get("USE_REAL_DATA", "0") == "1"
    if use_real:
        df = try_load_real()
        if df is None:
            print("Реальные данные недоступны; используем синтетические.")
            df = load_synthetic()
    else:
        df = load_synthetic()
    report_path = os.path.join(REPORTS_DIR, "research_report.pdf")
    analyze_and_report(df, report_path)
    print(f"Готово. Отчёт -> {report_path}")

if __name__ == "__main__":
    main()

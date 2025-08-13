
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
        temp_df["temp_anomaly_c"] = pd.to_numeric(temp_df["temp_anomaly_c"], errors="coerce") / 100.0
        temp_df = temp_df.dropna(subset=["temp_anomaly_c"])

        merged = pd.merge(co2_annual, temp_df, on="year", how="inner")
        merged["year"] = merged["year"].astype(int)
        return merged
    except Exception as exc:
        print(f"Ошибка при загрузке реальных данных: {exc}")
        return None

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

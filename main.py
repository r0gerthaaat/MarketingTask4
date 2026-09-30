import io
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

# ==========================================
# 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ
# ==========================================


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath, sep=";", decimal=",")
        if df.shape[1] == 1:
            df = pd.read_csv(filepath, sep=",", decimal=".")
    except Exception:
        df = pd.read_csv(filepath)

    numeric_cols = [
        "RealGDP",
        "real_0411_general",
        "real_0470_other",
        "real_0830_media",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(
                    r"\s+", "", regex=True
                )  # видаляє звичайні та нерозривні пробіли
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "IsWar" in df.columns:
        df["IsWar"] = (
            pd.to_numeric(df["IsWar"].astype(str).str.strip(), errors="coerce")
            .fillna(0)
            .astype(int)
        )
    if "Quarter" in df.columns:
        df["Quarter"] = (
            pd.to_numeric(
                df["Quarter"].astype(str).str.strip(), errors="coerce"
            )
            .fillna(1)
            .astype(int)
        )

    df["gdp_bln"] = df["RealGDP"] / 1e9
    df["exp_trade_mln"] = df["real_0411_general"] / 1e6
    df["exp_tourism_mln"] = df["real_0470_other"] / 1e6
    df["exp_media_mln"] = df["real_0830_media"] / 1e6

    df["exp_mkt_total_mln"] = df["exp_trade_mln"] + df["exp_tourism_mln"] + df["exp_media_mln"]

    df["exp_trade_lag1"] = df["exp_trade_mln"].shift(1)
    df["exp_tourism_lag1"] = df["exp_tourism_mln"].shift(1)
    df["exp_mkt_total_lag1"] = df["exp_mkt_total_mln"].shift(1)

    df["ln_gdp"] = np.log(df["gdp_bln"])
    df["ln_mkt_total"] = np.log(df["exp_mkt_total_mln"])

    quarter_dummies = pd.get_dummies(
        df["Quarter"], prefix="Q", drop_first=True, dtype=int
    )
    df = pd.concat([df, quarter_dummies], axis=1)

    return df

# ==========================================
# 2. МАТРИЦЯ КОРЕЛЯЦІЙ
# ==========================================


def run_correlation_analysis(df: pd.DataFrame):
    corr_cols = [
        "gdp_bln",
        "exp_trade_mln",
        "exp_tourism_mln",
        "exp_media_mln",
        "exp_mkt_total_mln",
        "IsWar",
        "Quarter",
    ]
    corr_matrix = df[corr_cols].corr()

    print("\n" + "=" * 60)
    print("1. МАТРИЦЯ ПАРНИХ КОРЕЛЯЦІЙ ПІРСОНА")
    print("=" * 60)
    print(corr_matrix.round(3).to_string())

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Кореляційна матриця змінних (2018–2025)", fontsize=13)
    plt.tight_layout()
    plt.savefig("1_correlation_matrix.png", dpi=300)
    plt.close()
    print("-> Графік матриці збережено у '1_correlation_matrix.png'")


# ==========================================
# 3. ТЕСТ ПРИЧИННОСТІ ЗА ГРЕЙНДЖЕРОМ
# ==========================================


def run_granger_causality(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("2. ТЕСТ ПРИЧИННОСТІ ЗА ГРЕЙНДЖЕРОМ (GRANGER CAUSALITY)")
    print("=" * 65)

    diff_df = pd.DataFrame(
        {
            "d_gdp": df["gdp_bln"].diff(),
            "d_mkt": df["exp_mkt_total_mln"].diff(),
            "d_tourism": df["exp_tourism_mln"].diff(),
        }
    ).dropna()

    diff_yoy = pd.DataFrame(
        {
            "dy_gdp": df["gdp_bln"].diff(4),
            "dy_mkt": df["exp_mkt_total_mln"].diff(4),
        }
    ).dropna()

    def print_table(test_res, name):
        print(f"\nТест: {name}")
        print(f"{'Лаг':<6} | {'F-stat':<10} | {'p-value':<10} | {'Висновок'}")
        print("-" * 55)
        for lag in [1, 2]:
            f_stat, p_val, _, _ = test_res[lag][0]["ssr_ftest"]
            status = (
                "Є причинність (p < 0.05)"
                if p_val < 0.05
                else "Немає причинності (p >= 0.05)"
            )
            print(f"{lag:<6} | {f_stat:<10.4f} | {p_val:<10.4f} | {status}")

    res_a = grangercausalitytests(
        diff_df[["d_gdp", "d_mkt"]], maxlag=2
    )
    print_table(res_a, "Маркетинг -> ВВП (квартальні різниці)")

    res_b = grangercausalitytests(
        diff_df[["d_mkt", "d_gdp"]], maxlag=2
    )
    print_table(res_b, "ВВП -> Маркетинг (квартальні різниці)")

    res_yoy_a = grangercausalitytests(
        diff_yoy[["dy_gdp", "dy_mkt"]], maxlag=2
    )
    print_table(res_yoy_a, "Маркетинг -> ВВП (сезонно очищені YoY різниці)")

    res_yoy_b = grangercausalitytests(
        diff_yoy[["dy_mkt", "dy_gdp"]], maxlag=2
    )
    print_table(res_yoy_b, "ВВП -> Маркетинг (сезонно очищені YoY різниці)")

# ==========================================
# 4. РЕГРЕСІЙНЕ МОДЕЛЮВАННЯ (4 МОДЕЛІ)
# ==========================================


def run_regression_models(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("3. ПОРІВНЯЛЬНИЙ ЕКОНОМЕТРИЧНИЙ АНАЛІЗ МОДЕЛЕЙ (OLS)")
    print("=" * 60)

    X1 = sm.add_constant(
        df[["exp_trade_mln", "exp_tourism_mln", "exp_media_mln"]]
    )
    y1 = df["gdp_bln"]
    m1 = sm.OLS(y1, X1).fit()

    X2 = sm.add_constant(
        df[
            [
                "exp_trade_mln",
                "exp_tourism_mln",
                "exp_media_mln",
                "IsWar",
                "Q_2",
                "Q_3",
                "Q_4",
            ]
        ]
    )
    y2 = df["gdp_bln"]
    m2 = sm.OLS(y2, X2).fit()

    lag_df = df.dropna(subset=["exp_trade_lag1", "exp_tourism_lag1"]).copy()
    X3 = sm.add_constant(
        lag_df[
            [
                "exp_trade_lag1",
                "exp_tourism_lag1",
                "IsWar",
                "Q_2",
                "Q_3",
                "Q_4",
            ]
        ]
    )
    y3 = lag_df["gdp_bln"]
    m3 = sm.OLS(y3, X3).fit()

    X4 = sm.add_constant(
        df[["ln_mkt_total", "IsWar", "Q_2", "Q_3", "Q_4"]]
    )
    y4 = df["ln_gdp"]
    m4 = sm.OLS(y4, X4).fit()

    summary_table = pd.DataFrame(
        {
            "Показник": ["R²", "Adj. R²", "F-statistic", "p-value (F)"],
            "Модель 1 (Наївна)": [
                f"{m1.rsquared:.3f}",
                f"{m1.rsquared_adj:.3f}",
                f"{m1.fvalue:.2f}",
                f"{m1.f_pvalue:.4e}",
            ],
            "Модель 2 (Контроль)": [
                f"{m2.rsquared:.3f}",
                f"{m2.rsquared_adj:.3f}",
                f"{m2.fvalue:.2f}",
                f"{m2.f_pvalue:.4e}",
            ],
            "Модель 3 (Лагова)": [
                f"{m3.rsquared:.3f}",
                f"{m3.rsquared_adj:.3f}",
                f"{m3.fvalue:.2f}",
                f"{m3.f_pvalue:.4e}",
            ],
            "Модель 4 (Log-Log)": [
                f"{m4.rsquared:.3f}",
                f"{m4.rsquared_adj:.3f}",
                f"{m4.fvalue:.2f}",
                f"{m4.f_pvalue:.4e}",
            ],
        }
    )
    print("\nЗведена таблиця якості моделей:")
    print(summary_table.to_string(index=False))

    print("\n--- Модель 2: Детальні коефіцієнти та p-values ---")
    m2_coefs = pd.DataFrame(
        {
            "Коефіцієнт (Beta)": m2.params,
            "Std. Error": m2.bse,
            "t-stat": m2.tvalues,
            "p-value": m2.pvalues,
        }
    )
    print(m2_coefs.round(4).to_string())

    plt.figure(figsize=(12, 5))
    plt.plot(
        df["Period"],
        df["gdp_bln"],
        label="Фактичний реальний ВВП",
        marker="o",
        color="#1f77b4",
        linewidth=2,
    )
    plt.plot(
        df["Period"],
        m2.fittedvalues,
        label="Підігнана модель OLS (Модель 2)",
        linestyle="--",
        color="#d62728",
        marker="x",
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.ylabel("Реальний ВВП, млрд грн (у цінах 2021)")
    plt.title(
        "Фактична динаміка ВВП та оцінка моделі з контролем факторів",
        fontsize=12,
    )
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("2_regression_fit.png", dpi=300)
    plt.close()
    print("-> Графік підгонки моделі збережено у '2_regression_fit.png'")


# ==========================================
# 5. КЛАСТЕРИЗАЦІЯ ПЕРІОДІВ ТА PCA
# ==========================================


def run_clustering_regimes(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("4. КЛАСТЕРИЗАЦІЯ ПЕРІОДІВ ТА АНАЛІЗ РЕЖИМІВ (K-MEANS + PCA)")
    print("=" * 60)

    features = [
        "gdp_bln",
        "exp_trade_mln",
        "exp_tourism_mln",
        "exp_media_mln",
        "IsWar",
    ]
    X_clust = df[features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)
    df["PCA1"] = pca_coords[:, 0]
    df["PCA2"] = pca_coords[:, 1]

    cluster_means = df.groupby("Cluster")[features].mean()
    print("\nСередні значення показників за кластерами:")
    print(cluster_means.round(1).to_string())

    plt.figure(figsize=(11, 7))
    palette = ["#2ca02c", "#ff7f0e", "#d62728"]
    sns.scatterplot(
        data=df,
        x="PCA1",
        y="PCA2",
        hue="Cluster",
        palette=palette,
        s=120,
        style="Cluster",
    )

    for _, row in df.iterrows():
        plt.text(
            row["PCA1"] + 0.08,
            row["PCA2"] + 0.08,
            row["Period"],
            fontsize=8,
            alpha=0.85,
        )

    plt.title(
        "Кластеризація кварталів (2018–2025): Економічні режими та маркетинг",
        fontsize=13,
    )
    plt.xlabel(
        f"Головна компонента 1 (Масштаб економіки / Війна) [{pca.explained_variance_ratio_[0]*100:.1f}%]"
    )
    plt.ylabel(
        f"Головна компонента 2 (Сезонна активність / Видатки) [{pca.explained_variance_ratio_[1]*100:.1f}%]"
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(
        title="Кластер режиму",
        labels=[
            "Кластер 2: Довоєнне зростання",
            "Кластер 0: Піки Q4 2018–2021",
            "Кластер 1: Воєнний режим",
        ],
    )
    plt.tight_layout()
    plt.savefig("3_clusters_pca.png", dpi=300)
    plt.close()
    print("-> Графік кластеризації збережено у '3_clusters_pca.png'")


# ==========================================
# ГОЛОВНИЙ ТОЧКА ВХОДУ
# ==========================================

if __name__ == "__main__":
    CSV_FILENAME = "marketing_gdp_data.csv"

    print("Завантаження даних із файлу:", CSV_FILENAME)
    data = load_and_preprocess_data(CSV_FILENAME)

    run_correlation_analysis(data)
    run_granger_causality(data)
    run_regression_models(data)
    run_clustering_regimes(data)

    print("\n" + "=" * 60)
    print("Економетричний аналіз завершено успішно.")
    print("Усі три графіки для презентації згенеровано у високій якості (300 DPI).")
    print("=" * 60)
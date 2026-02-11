import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def apply_correlation_to_df(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the correlation between GDP per capita and CO2 per capita for each country
    and return a Series that can be assigned to a column.

    Args:
        df (pd.DataFrame): DataFrame with columns 'country', 'gdp_per_capita', and 'co2_per_capita'

    Returns:
        pd.Series: A Series with correlation values for each row based on its country
    """

    # Calculate correlation per country using groupby + transform
    def calc_corr(group):
        subset = group[["gdp_per_capita", "co2_per_capita"]].dropna()
        if len(subset) < 5:
            return None
        corr_matrix = np.corrcoef(subset["gdp_per_capita"], subset["co2_per_capita"])
        return corr_matrix[0, 1]

    # Get unique country correlations
    country_correlations = df.groupby("country").apply(calc_corr)

    # Map back to original DataFrame index
    return df["country"].map(country_correlations)


def normalize_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Normalize the names in a column of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with the column to normalize
        column_name (str): Name of the column to normalize

    Returns:
        pd.DataFrame: DataFrame with the normalized column
    """
    return df[column_name].str.lower().str.strip()


def z_score_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Calculate the z-score of a column in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with the column to calculate the z-score for
        column_name (str): Name of the column to calculate the z-score for

    Returns:
        pd.DataFrame: DataFrame with the z-score of the column
    """
    return (df[column_name] - df[column_name].mean()) / df[column_name].std()


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Perform element-wise division, returning zero where the denominator is zero or negative.

    Args:
        numerator (np.ndarray): Numerator array (or pandas Series).
        denominator (np.ndarray): Denominator array (or pandas Series).

    Returns:
        np.ndarray: Result of element-wise division with zeros where denominator is invalid.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros_like(numerator)
    mask = denominator > 0
    return np.divide(numerator, denominator, out=out, where=mask)


def classify_income_group(series: pd.Series) -> pd.Categorical:
    """
    Classify GDP per capita values into World Bank income groups.

    Uses the World Bank FY2025 GNI thresholds:
        Low:          <= 1,145
        Lower-Middle: 1,146 - 4,515
        Upper-Middle: 4,516 - 14,005
        High:         > 14,005

    Args:
        series (pd.Series): A Series of GDP per capita values.

    Returns:
        pd.Categorical: Categorical Series with income group labels.
    """
    # classify countries using World Bank thresholds with numpy vectorized logic
    gdp_pc = series.values

    conditions = [
        gdp_pc < 1_145,
        (gdp_pc >= 1_145) & (gdp_pc < 4_516),
        (gdp_pc >= 4_516) & (gdp_pc < 14_005),
        gdp_pc >= 14_005,
    ]
    choices = ["Low", "Lower-Middle", "Upper-Middle", "High"]

    return np.select(conditions, choices, default="Unknown")




def get_top_bottom_n(
    df: pd.DataFrame, metric_col: str, n: int = 15, ascending: bool = True
) -> pd.DataFrame:
    """
    Return the top or bottom N rows of a DataFrame ranked by a metric column.

    Args:
        df (pd.DataFrame): DataFrame to rank.
        metric_col (str): Column name to rank by.
        n (int): Number of rows to return. Defaults to 15.
        ascending (bool): If True, return the smallest values (nsmallest).
            If False, return the largest values (nlargest). Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with the top or bottom N rows.
    """
    if ascending:
        return df.nsmallest(n, metric_col)
    return df.nlargest(n, metric_col)


def compute_energy_mix_shares(
    df: pd.DataFrame, green_cols: list, non_green_cols: list
) -> pd.DataFrame:
    """
    Compute green and non-green electricity share columns for each row.

    Args:
        df (pd.DataFrame): DataFrame with electricity generation columns.
        green_cols (list): Column names for green energy sources.
        non_green_cols (list): Column names for non-green energy sources.

    Returns:
        pd.DataFrame: Copy of the input with 'green_share' and 'non_green_share' columns added.
    """
    green_total = df[green_cols].sum(axis=1)
    non_green_total = df[non_green_cols].sum(axis=1)
    total = green_total + non_green_total

    df = df.copy()
    df["green_share"] = safe_divide(green_total.values, total.values)
    df["non_green_share"] = safe_divide(non_green_total.values, total.values)
    return df


def plot_dual_axis_timeseries(
    df: pd.DataFrame,
    x_col: str,
    y1_col: str,
    y2_col: str,
    y1_label: str,
    y2_label: str,
    title: str = None,
    correlation_value: float = None,
) -> None:
    """
    Plot a dual-axis time series chart with optional Pearson r annotation.

    Args:
        df (pd.DataFrame): DataFrame with the data to plot.
        x_col (str): Column name for the x-axis (typically 'year').
        y1_col (str): Column name for the left y-axis series.
        y2_col (str): Column name for the right y-axis series.
        y1_label (str): Label for the left y-axis.
        y2_label (str): Label for the right y-axis.
        title (str): Plot title. Defaults to None (auto-generated).
        correlation_value (float): Pearson r value to annotate. Defaults to None.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df[x_col], df[y1_col], label=y1_label)
    ax1.set_xlabel(x_col.capitalize())
    ax1.set_ylabel(y1_label)
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(df[x_col], df[y2_col], label=y2_label, color="orange")
    ax2.set_ylabel(y2_label)
    ax2.legend(loc="upper right")

    if correlation_value is not None:
        ax1.annotate(
            f"Pearson r: {correlation_value:.2f}",
            xy=(0.4, 0.93),
            xycoords="axes fraction",
            fontsize=12,
        )

    valid_rows = df.dropna(subset=[y1_col, y2_col])
    if not valid_rows.empty:
        min_x = valid_rows[x_col].min()
        max_x = valid_rows[x_col].max()
        ax1.set_xlim(min_x, max_x)
        if title is None:
            title = f"{y1_label} vs {y2_label}, {min_x}-{max_x}"

    if title:
        plt.title(title)

    plt.show()


def run_kmeans_elbow(
    df: pd.DataFrame,
    feature_cols: list,
    k_range: range = range(1, 8),
    n_clusters: int = None,
    random_state: int = 42,
    plot: bool = True,
) -> tuple:
    """
    Run KMeans clustering with StandardScaler preprocessing and optional elbow plot.

    Args:
        df (pd.DataFrame): DataFrame with the feature columns.
        feature_cols (list): Column names to use as clustering features.
        k_range (range): Range of k values for the elbow plot. Defaults to range(1, 8).
        n_clusters (int): Number of clusters for the final fit. Defaults to None.
        random_state (int): Random state for reproducibility. Defaults to 42.
        plot (bool): Whether to display the elbow plot. Defaults to True.

    Returns:
        tuple: (cluster_labels, scaled_data, kmeans_model) where any may be None.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    if plot:
        inertia = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            km.fit(scaled_data)
            inertia.append(km.inertia_)

        plt.plot(list(k_range), inertia, marker="o")
        plt.xlabel("Clusters (k)")
        plt.ylabel("Inertia")
        plt.show()

    cluster_labels = None
    kmeans_model = None
    if n_clusters is not None:
        kmeans_model = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        )
        cluster_labels = kmeans_model.fit_predict(scaled_data)

    return cluster_labels, scaled_data, kmeans_model

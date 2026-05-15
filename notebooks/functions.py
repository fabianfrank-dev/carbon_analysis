import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import re
import requests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET


def _clean_scraped_text(value: str | None) -> str:
    """
    Normalize scraped text by removing reference markers and extra whitespace.
    """
    if value is None:
        return ""

    # replace nbsp and strip citation markers like [1]
    cleaned = value.replace("\xa0", " ")
    cleaned = re.sub(r"\[\d+\]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _parse_optional_number(value: str | None) -> float:
    """
    Parse a scraped numeric string, returning NaN when no number is available.
    """
    cleaned = _clean_scraped_text(value)
    if cleaned in {"", "-", "—", "..", "No data"}:
        return np.nan

    # keep only the first numeric chunk so notes in the cell do not break parsing
    match = re.search(r"-?\d[\d,]*\.?\d*", cleaned.replace("$", ""))
    if match is None:
        return np.nan

    return float(match.group(0).replace(",", ""))


def _table_to_dataframe(table) -> pd.DataFrame:
    """
    Convert an HTML table element to a pandas DataFrame using BeautifulSoup only.
    """
    thead = table.find("thead")
    tbody = table.find("tbody")

    # prefer the explicit header row when the html provides one
    header_row = thead.find_all("tr")[-1] if thead and thead.find_all("tr") else table.find("tr")
    header_cells = header_row.find_all(["th", "td"]) if header_row else []
    headers = [_clean_scraped_text(cell.get_text(" ", strip=True)) for cell in header_cells]

    rows = []
    body_rows = tbody.find_all("tr") if tbody else table.find_all("tr")[1:]
    for row in body_rows:
        cells = row.find_all(["th", "td"])
        values = [_clean_scraped_text(cell.get_text(" ", strip=True)) for cell in cells]

        if not values:
            continue

        # pad or trim so odd wikipedia rows still fit the header width
        if len(values) < len(headers):
            values = values + [None] * (len(headers) - len(values))
        elif len(values) > len(headers):
            values = values[:len(headers)]

        rows.append(values)

    return pd.DataFrame(rows, columns=headers)


def fetch_wikipedia_gni_table(
    url: str,
    user_agent: str = "Mozilla/5.0",
) -> pd.DataFrame:
    """
    Fetch the Wikipedia GNI-per-capita table with requests + BeautifulSoup.
    """
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # wikipedia can have multiple tables, so find the one with the expected headers
    for table in soup.select("table.wikitable"):
        header_text = [
            _clean_scraped_text(cell.get_text(" ", strip=True)).lower()
            for cell in table.find_all("th")
        ]
        has_country = any(text == "country" for text in header_text)
        has_gni = any("gni per capita" in text for text in header_text)

        if not (has_country and has_gni):
            continue

        gni_df = _table_to_dataframe(table)

        # standardize the key columns so the notebooks can merge safely
        country_col = next((col for col in gni_df.columns if col.lower() == "country"), None)
        gni_col = next((col for col in gni_df.columns if "gni per capita" in col.lower()), None)
        if country_col is None or gni_col is None:
            continue

        gni_df = gni_df.rename(columns={country_col: "Country", gni_col: "gni_per_capita"})
        gni_df["Country"] = gni_df["Country"].map(_clean_scraped_text)
        gni_df["gni_per_capita"] = gni_df["gni_per_capita"].map(_parse_optional_number)

        return gni_df

    raise ValueError("Could not find the GNI per capita table on the page.")


def _strip_xml_namespace(tag: str) -> str:
    """
    Remove the namespace prefix from an XML tag.
    """
    return tag.split("}", 1)[-1]


def fetch_world_bank_xml_records(
    url: str,
    user_agent: str = "Mozilla/5.0",
) -> pd.DataFrame:
    """
    Fetch and flatten a World Bank XML response with ElementTree.
    """
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    rows = []

    # walk each record node and flatten its children into a dictionary
    for record in root.iter():
        if _strip_xml_namespace(record.tag) != "record":
            continue

        row = {}
        for child in record:
            key = _strip_xml_namespace(child.tag)
            row[key] = _clean_scraped_text(child.text)

            # keep ids too in case the text labels change in the future
            if child.attrib.get("id"):
                row[f"{key}_id"] = child.attrib["id"]

        rows.append(row)

    xml_df = pd.DataFrame(rows)
    if xml_df.empty:
        return xml_df

    # coerce the common numeric fields for easier downstream use
    if "date" in xml_df.columns:
        xml_df["date"] = pd.to_numeric(xml_df["date"], errors="coerce")
    if "value" in xml_df.columns:
        xml_df["value"] = xml_df["value"].map(_parse_optional_number)

    return xml_df


def apply_correlation_to_df(df: pl.DataFrame) -> pl.Series:
    """
    Calculate the correlation between GDP per capita and CO2 per capita for each country
    and return a Series that can be assigned to a column.

    Args:
        df (pd.DataFrame): DataFrame with columns 'country', 'gdp_per_capita', and 'co2_per_capita'

    Returns:
        pd.Series: A Series with correlation values for each row based on its country
    """
    # using try/except to alternate between pandas and polars
    # if polars is run: return polars/else: run pandas
    try:
        # Calculate correlation per country using groupby + transform
        country_corr = (
            df.filter(
                pl.col("gdp_per_capita").is_not_null() &
                pl.col("co2_per_capita").is_not_null())
            .group_by("country")
            .agg([
                pl.len().alias("n"),
                pl.corr("gdp_per_capita", "co2_per_capita").alias("correlation")])
            .with_columns(
                pl.when(pl.col("n") >= 5)
                .then(pl.col("correlation"))
                .otherwise(None)
                .alias("correlation"))
            .select(["country", "correlation"]))

        return df.join(country_corr, on="country", how="left")
    except:
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


def normalize_column(column_name: str, df: pd.DataFrame = None, ) -> pl.Expr:
    """
    Normalize the names in a column of a DataFrame.

    Args:
        column_name (str): Name of the column to normalize

    Returns:
        pl.Expr: Polars expression for the normalized column
    """
    if df is None:
        return pl.col(column_name).str.strip_chars().str.to_lowercase().alias(f"{column_name}_clean")
    else: 
        return df[column_name].str.lower().str.strip()


def z_score_column(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """
    Calculate the z-score of a column in a DataFrame.

    Args:
        df (pl.DataFrame): DataFrame with the column to calculate the z-score for
        column_name (str): Name of the column to calculate the z-score for

    Returns:
        pl.DataFrame: DataFrame with the z-score of the column
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
    # convert to numpy arrays
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros_like(numerator)
    mask = denominator > 0

    return np.divide(numerator, denominator, out=out, where=mask)


def classify_income_group(col: str) -> pl.Expr:
    """
    Classify GDP per capita values into World Bank income groups.

    Uses the World Bank FY2025 GNI thresholds:
        Low:          <= 1,145
        Lower-Middle: 1,146 - 4,515
        Upper-Middle: 4,516 - 14,005
        High:         > 14,005

    Args:
        col (str): Column name of the GDP per capita values.

    Returns:
        pl.Expr: Polars expression for the income group.
    """
    # classify countries using World Bank thresholds with numpy vectorized logic
    try:
        return (pl.when(pl.col(col) < 1145).then(pl.lit("Low"))
            .when(pl.col(col) < 4516).then(pl.lit("Lower-Middle"))
            .when(pl.col(col) < 14005).then(pl.lit("Upper-Middle"))
            .otherwise(pl.lit("High"))
            .alias("income_group"))
    except:
        gdp_pc = col
        conditions = [
            gdp_pc < 1_145,
            (gdp_pc >= 1_145) & (gdp_pc < 4_516),
            (gdp_pc >= 4_516) & (gdp_pc < 14_005),
            gdp_pc >= 14_005,
        ]
        choices = ["Low", "Lower-Middle", "Upper-Middle", "High"]

        return np.select(conditions, choices, default="Unknown")


def compute_energy_mix_shares(
    df: pl.DataFrame, green_cols: list, non_green_cols: list
) -> pl.DataFrame:
    """
    Compute green and non-green electricity share columns for each row.

    Args:
        df (pl.DataFrame): DataFrame with electricity generation columns.
        green_cols (list): Column names for green energy sources.
        non_green_cols (list): Column names for non-green energy sources.

    Returns:
        pl.DataFrame: Copy of the input with 'green_share' and 'non_green_share' columns added.
    """
    try:
        df = df.with_columns([
            (pl.sum_horizontal(green_cols) / pl.sum_horizontal(green_cols + non_green_cols)).alias("green_share"),
            (pl.sum_horizontal(non_green_cols) / pl.sum_horizontal(green_cols + non_green_cols)).alias("non_green_share")
        ])
        return df
    except:
        green_total = df[green_cols].sum(axis=1)
        non_green_total = df[non_green_cols].sum(axis=1)
        total = green_total + non_green_total

        df = df.copy()
        df["green_share"] = safe_divide(green_total.values, total.values)
        df["non_green_share"] = safe_divide(non_green_total.values, total.values)
        return df

def plot_dual_axis_timeseries(
    df: pl.DataFrame,
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

    # check if correlation value is provided and annotate it
    if correlation_value is not None:
        ax1.annotate(
            f"Pearson r: {correlation_value:.2f}",
            xy=(0.4, 0.93),
            xycoords="axes fraction",
            fontsize=12,
        ) 

    # check if there are valid rows and set the x-axis limits
    # also check for pandas or polars
    try:
        valid_rows = df.drop_nulls(subset=[y1_col, y2_col])
        if not valid_rows.is_empty():
            min_x = valid_rows[x_col].min()
            max_x = valid_rows[x_col].max()
            ax1.set_xlim(min_x, max_x)
            if title is None:
                title = f"{y1_label} vs {y2_label}, {min_x}-{max_x}"

    except:
        valid_rows = df.dropna(subset=[y1_col, y2_col])
        if not valid_rows.empty:
            min_x = valid_rows[x_col].min()
            max_x = valid_rows[x_col].max()
            ax1.set_xlim(min_x, max_x)
            if title is None:
                title = f"{y1_label} vs {y2_label}, {min_x}-{max_x}"

    # set the plot title
    if title:
        plt.title(title)

    plt.show()


def run_kmeans_elbow(
    X : np.ndarray,
    k_range,
    n_clusters : int | None,
    scale : bool,
    random_state : int,
    n_init : int,
    plot : bool,
    ax = None,
) -> tuple[np.ndarray | None, np.ndarray, KMeans | None, list[float], StandardScaler | None]:
    """
    K-means preparation, optional elbow plot and optional final clustering.

    Parameters
    ----------
    X : array-like
        2D-data matrix with shape (n_samples, n_features).
    k_range : iterable
        Values of k for the elbow plot.
    n_clusters : int | None
        If set, a final KMeans model with this k is trained.
        If None, only the elbow plot / inertia calculation is made.
    scale : bool
        Whether the data should be standardized before clustering.
    random_state : int
        Random state for reproducibility.
    n_init : int
        Number of KMeans initializations.
    plot : bool
        Whether to plot the elbow plot.
    ax : matplotlib.axes.Axes | None
        Optional existing axis for the plot.

    Returns
    -------
    cluster_labels : np.ndarray | None
        Cluster labels, if n_clusters is set, otherwise None.
    X_used : np.ndarray
        Used data matrix (scaled or unscaled).
    model : KMeans | None
        Trained KMeans model, if n_clusters is set, otherwise None.
    inertias : list[float]
        Inertia values for k_range.
    scaler : StandardScaler | None
        Fitted scaler, if scale=True, otherwise None.
    """
    X = np.asarray(X)

    # check input
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, but has shape {X.shape}.")

    # check for empty data
    if X.shape[0] == 0:
        raise ValueError("X is empty.")

    # check for NaN, inf or -inf
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN, inf or -inf. Please clean before.")

    # scale data if required
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_used = scaler.fit_transform(X)
    else:
        X_used = X.copy()

    # calculate inertias
    inertias = []
    k_values = list(k_range)

    # calculate inertias for each k
    for k in k_values:
        if k < 1:
            raise ValueError(f"Invalid k={k}. k must be >= 1.")
        if k > X_used.shape[0]:
            raise ValueError(
                f"Invalid k={k}. There are only {X_used.shape[0]} samples."
            )

        model_k = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
        )
        model_k.fit(X_used)
        inertias.append(model_k.inertia_)

    # plot inertias if required
    if plot:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        ax.plot(k_values, inertias, marker="o")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Plot")
        ax.set_xticks(k_values)

    # initialize variables  
    cluster_labels = None
    model = None

    # train final model if required
    if n_clusters is not None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if n_clusters > X_used.shape[0]:
            raise ValueError(
                f"n_clusters={n_clusters} is greater than the number of samples ({X_used.shape[0]})."
            )

        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
        )
        cluster_labels = model.fit_predict(X_used)

    return cluster_labels, X_used, model, inertias, scaler


def flatten_owid_panel_json(
    payload: dict,
    min_year: int | None = None,
) -> pd.DataFrame:
    """
    Flatten the OWID country -> yearly-record JSON structure into a panel DataFrame.

    Args:
        payload (dict): Raw OWID JSON payload.
        min_year (int | None): Optional lower year bound.

    Returns:
        pd.DataFrame: Country-year panel with country metadata repeated on each row.
    """
    rows = []

    for country, content in payload.items():
        if not isinstance(content, dict):
            continue

        # keep simple country-level metadata and ignore nested objects/lists
        metadata = {
            key: value
            for key, value in content.items()
            if key != "data" and not isinstance(value, (dict, list))
        }
        metadata["country"] = country

        for entry in content.get("data", []):
            if not isinstance(entry, dict):
                continue

            # repeat metadata on each yearly row so we get a flat panel
            row = metadata.copy()
            row.update(entry)
            rows.append(row)

    panel_df = pd.DataFrame.from_records(rows)
    if panel_df.empty:
        return panel_df

    if "year" in panel_df.columns:
        # coerce years safely before filtering
        panel_df["year"] = pd.to_numeric(panel_df["year"], errors="coerce")
        panel_df = panel_df.dropna(subset=["year"]).copy()
        panel_df["year"] = panel_df["year"].astype(int)

        if min_year is not None:
            panel_df = panel_df[panel_df["year"] >= min_year].copy()

    # bring the main identifier columns to the front for readability
    leading_cols = [col for col in ["country", "iso_code", "year"] if col in panel_df.columns]
    remaining_cols = [col for col in panel_df.columns if col not in leading_cols]
    return panel_df[leading_cols + remaining_cols]


def prepare_source_trend_table(
    df: pd.DataFrame,
    source_cols: list[str],
    time_col: str = "year",
    total_col: str = "co2",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Reshape wide emissions-source columns into a long trend table with source shares.

    Args:
        df (pd.DataFrame): Panel data containing total CO2 and source columns.
        source_cols (list[str]): Component emission columns such as coal/oil/gas.
        time_col (str): Time column to aggregate over.
        total_col (str): Total emissions column used as denominator for shares.
        group_cols (list[str] | None): Optional extra grouping columns.

    Returns:
        pd.DataFrame: Long-format source table with aggregated source totals and shares.
    """
    group_cols = group_cols or []
    available_source_cols = [col for col in source_cols if col in df.columns]

    if not available_source_cols:
        raise ValueError("None of the requested source columns were found in the DataFrame.")

    base_cols = [time_col, total_col] + group_cols
    source_df = df[base_cols + available_source_cols].copy()

    # convert the source columns to numeric before reshaping
    numeric_cols = [total_col] + available_source_cols
    for column in numeric_cols:
        source_df[column] = pd.to_numeric(source_df[column], errors="coerce")

    # reshape from wide source columns to a tidy long table
    melted = source_df.melt(
        id_vars=base_cols,
        value_vars=available_source_cols,
        var_name="source",
        value_name="source_co2",
    ).dropna(subset=["source_co2", total_col])

    aggregation_cols = group_cols + [time_col, "source"]
    # aggregate source totals and compare them to total CO2 in the same period
    trend_df = melted.groupby(aggregation_cols, as_index=False).agg(
        source_co2=("source_co2", "sum"),
        total_co2=(total_col, "sum"),
    )
    trend_df["source_share"] = safe_divide(
        trend_df["source_co2"].to_numpy(),
        trend_df["total_co2"].to_numpy(),
    )

    source_labels = {
        col: col.removesuffix("_co2").replace("_", " ").title()
        for col in available_source_cols
    }
    # keep the original source ordering for nicer plots later
    trend_df["source_label"] = pd.Categorical(
        trend_df["source"].map(source_labels),
        categories=[source_labels[col] for col in available_source_cols],
        ordered=True,
    )

    return trend_df.sort_values(aggregation_cols).reset_index(drop=True)


def compute_group_concentration(
    df: pd.DataFrame,
    period_col: str,
    entity_col: str,
    value_col: str,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Compute concentration metrics such as HHI and top-N share within each period.

    Args:
        df (pd.DataFrame): Input panel data.
        period_col (str): Column defining the time period.
        entity_col (str): Column identifying entities within each period.
        value_col (str): Numeric column used to compute concentration.
        top_n (int): Number of leading entities to include in the top-share metric.

    Returns:
        pd.DataFrame: Period-level totals, HHI, effective entity count, and top-N share.
    """
    concentration_df = df[[period_col, entity_col, value_col]].copy()
    # make sure the concentration metric only uses valid numeric rows
    concentration_df[value_col] = pd.to_numeric(concentration_df[value_col], errors="coerce")
    concentration_df = concentration_df.dropna(subset=[period_col, entity_col, value_col])
    concentration_df = concentration_df[concentration_df[value_col] > 0].copy()

    # compute each entity's share inside its year/group
    concentration_df["group_total"] = concentration_df.groupby(period_col)[value_col].transform("sum")
    concentration_df["share"] = safe_divide(
        concentration_df[value_col].to_numpy(),
        concentration_df["group_total"].to_numpy(),
    )

    # HHI captures overall concentration, while top_n_share keeps it intuitive
    summary_df = concentration_df.groupby(period_col, as_index=False).agg(
        total_value=(value_col, "sum"),
        entity_count=(entity_col, "nunique"),
        hhi=("share", lambda shares: np.square(shares).sum()),
        top_n_share=("share", lambda shares: shares.nlargest(top_n).sum()),
    )
    # effective entity count is the inverse HHI interpretation
    summary_df["effective_entity_count"] = safe_divide(
        np.ones(len(summary_df)),
        summary_df["hhi"].to_numpy(),
    )

    return summary_df.sort_values(period_col).reset_index(drop=True)


def get_top_bottom_n(
    df: pd.DataFrame,
    metric_col: str,
    n: int = 15,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Return the top or bottom N rows of a DataFrame ranked by a metric column.
    """
    if ascending:
        return df.nsmallest(n, metric_col)
    return df.nlargest(n, metric_col)

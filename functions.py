import pandas as pd
import numpy as np

def get_clean_correlation(df: pd.DataFrame, country_name: str) -> float | None:
    """
    Calculate the correlation between GDP per capita and CO2 per capita for a given country.

    Args:
        df (pd.DataFrame): DataFrame with columns 'country', 'gdp_per_capita', and 'co2_per_capita'
        country_name (str): Name of the country to calculate the correlation for

    Returns:
        float | None: Correlation coefficient if there are enough data points, None otherwise
    """

    # Select the data for the country 
    data = df[df['country'] == country_name].copy()

    subset = data[['gdp_per_capita', 'co2_per_capita']].dropna()

    # Check if there are enough data points
    if len(subset) < 5:
        # return None if there are not enough data points
        return None

    # Calculate the correlation
    corr_matrix = np.corrcoef(subset['gdp_per_capita'], subset['co2_per_capita'])

    return corr_matrix[0, 1]


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

def get_income_group(df: pd.DataFrame, country_name: str) -> str:
    """
    Get the income group of a country.

    Args:
        df (pd.DataFrame): DataFrame with columns 'country' and 'income_group'
        country_name (str): Name of the country to get the income group for

    Returns:
        str: Income group of the country
    """

    
    

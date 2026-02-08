import pandas as pd
import numpy as np


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
        subset = group[['gdp_per_capita', 'co2_per_capita']].dropna()
        if len(subset) < 5:
            return None
        corr_matrix = np.corrcoef(subset['gdp_per_capita'], subset['co2_per_capita'])
        return corr_matrix[0, 1]
    
    # Get unique country correlations
    country_correlations = df.groupby('country').apply(calc_corr)
    
    # Map back to original DataFrame index
    return df['country'].map(country_correlations)


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

    
    

import pandas as pd
import numpy as np

def get_clean_correlation(df, country_name):
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

# Decoupling Economic Growth from CO2 Emissions: A Data Analytics Study

### Overview

This repository contains a comprehensive data analytics project investigating the historical and contemporary relationship between economic development and environmental impact. Developed as a university-level technical demonstration, the project explores whether nations can achieve "absolute decoupling"—increasing Gross Domestic Product (GDP) while simultaneously reducing territorial CO2 emissions.

The analysis utilizes a high-performance Polars-led workflow to process large-scale environmental and economic datasets, transitioning to Pandas for specific ingestion tasks and integration with the Python visualization and machine learning ecosystem.
Research Question

## To what extent has economic growth decoupled from CO2 emissions across different income groups, and what role do energy intensity and electricity mix play in this transition?

This study does not seek to prove definitive causality; rather, it identifies correlations, clusters country archetypes based on developmental profiles, and uses interpretability tools to highlight which features most strongly associate with high carbon intensity.

## Key Analyses & Features

- **Data Orchestration**: Reshaping wide-format GDP data into long-format tidy data and merging heterogeneous sources (CSV, XML, HTML) by country-year identifiers.

- **Mixed Processing Workflow**: Leveraging Polars for memory-efficient joins and transformations, while using Pandas for XML/HTML parsing and matplotlib compatibility.

- **Feature Engineering**: Calculation of per-capita metrics, carbon intensity (CO2 per unit of GDP), and Z-score based outlier detection to identify data anomalies or unique "green growth" stories.

- **Income Group Classification**: Categorizing nations based on GNI per capita sourced from Wikipedia to analyze if decoupling is a luxury of high-income economies.

- **Live Data Ingestion**: Parsing World Bank XML API responses to integrate real-time energy use data.

- **Dimensionality Reduction**: Using UMAP and PCA to visualize how countries cluster geographically and economically in a high-dimensional feature space.

## Data Sources

- **Our World in Data (OWID)**: Primary CO2 emissions and electricity production mix datasets.

- **World Bank**: Historical GDP data and Energy Use per capita (via XML API).

- **Wikipedia**: Current GNI per capita tables for income group mapping.

## Tech Stack

- **Data Manipulation**: polars, pandas, numpy, (pyarrow, dask)

- **Visualization**: seaborn, matplotlib

- **Machine Learning & Stats**: scikit-learn, umap-learn, (shaply)

- **Data Ingestion**: lxml, requests

## Project Structure

├── data/
│   ├── co2_data.csv            # CO2 emissions by country and year
│   └── gdp_data.csv            # GDP (PPP) metrics in wide format
├── data_analytics_project.ipynb # Main analysis and visualization notebook
├── functions.py                # Helper scripts for API calls and cleaning
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation

## Installation & Running the Notebook

- Clone the repository:

    ```bash
    git clone https://github.com/your-username/carbon_analysis.git
    cd carbon_analysis
    ```

- Install dependencies:
    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

    Launch the analysis:

    ```bash
    jupyter notebook data_analytics_project.ipynb
    ```

- Note on Reproducibility: Several sections of the notebook fetch data live from the World Bank API and Wikipedia. An active internet connection is required to execute the notebook in its entirety. If the APIs are unreachable, the notebook is structured to use cached versions where available.

## Future Improvements

- Lag-Time Analysis: Implementing time-series cross-correlation to see if GDP growth leads to emissions reductions with a specific year-delay.

- Policy Indicators: Integrating qualitative data on national carbon taxes or renewable energy subsidies to see if they correlate with the observed clusters.

- Dashboarding: Converting the static visualizations into an interactive Streamlit dashboard for real-time country comparisons.

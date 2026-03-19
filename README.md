# Carbon Emissions and Economic Development

> Work in progress. This repository is being developed as a university data analytics project focused on clear data handling, thoughtful visualization, and methodical analysis.

## Overview

This project examines whether economic growth can occur without a proportional increase in carbon emissions. More specifically, it studies the relationship between **GDP development**, **CO2 emissions**, and **energy structure** across countries over time.

The notebook combines multiple public datasets, reshapes and merges them at the country-year level, engineers per-capita and efficiency metrics, and then explores whether countries appear to remain tightly coupled to emissions growth or show signs of relative decoupling.

The project is exploratory rather than causal: the goal is to identify patterns, contrasts, and analytically useful country groupings, not to claim that any one factor alone explains national emission outcomes.

## Research Question

The central question is:

**Can countries grow economically while reducing their carbon intensity or weakening the link between GDP growth and CO2 emissions?**

This question is explored through:

- country-level GDP-CO2 correlation analysis
- per-capita normalization
- income-group comparisons
- carbon-intensity trends
- electricity-mix comparisons
- selected clustering and dimensionality-reduction analyses

## Current Scope

The notebook currently demonstrates the following workflow:

1. **Data loading and cleaning**
   CO2 and GDP data are loaded, reduced to relevant columns, and cleaned to remove aggregate entities and unusable rows.

2. **Reshaping and merging**
   GDP data is transformed from wide to long format, then merged with emissions data using ISO country codes and years.

3. **Feature engineering**
   The analysis constructs `co2_per_capita`, `gdp_per_capita`, and `co2_per_gdp` to make countries more comparable across population size and economic scale.

4. **Missing-data inspection**
   Missing values are visualized and summarized after merging to make data quality issues explicit rather than implicit.

5. **Correlation analysis**
   GDP per capita and CO2 per capita are compared over time at the country level using Pearson correlation.

6. **Income-group classification**
   Countries are assigned to income groups using available GNI-per-capita data and then compared across key climate-economic metrics.

7. **Comparative visualization**
   The project includes violin plots, faceted time series, dual-axis plots, bar charts, and scatter-style comparative views.

8. **Carbon-intensity analysis**
   CO2 per GDP is used as an emissions-efficiency measure, both across time and for ranking countries in recent years.

9. **Energy-structure analysis**
   Electricity-source data is used to compare green and non-green energy shares with carbon intensity.

10. **Extended exploratory analysis**
    Additional sections include z-score style outlier analysis, clustering of country profiles, and dimensionality reduction to inspect broader country archetypes.

## Key Findings So Far

The current notebook suggests several recurring patterns:

- **High-income countries** show the widest spread of GDP-CO2 correlations, including more cases consistent with decoupling.
- **Lower-income countries** are more concentrated around positive correlations, indicating that growth and emissions often still move together.
- **Carbon intensity** tends to decline over time across income groups, although the speed and level of improvement differ substantially.
- **Electricity mix matters**, but it does not fully explain efficiency differences on its own.
- Some countries behave as clear **outliers**, which makes anomaly detection and clustering analytically useful.

These findings should be read as exploratory patterns rather than final conclusions.

## Data Sources

| Dataset | Source | Coverage / Use |
|---|---|---|
| CO2 emissions | [Our World in Data](https://ourworldindata.org/co2-emissions) | Main emissions data, filtered to modern country-year observations |
| GDP | [World Bank](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD) | National GDP, reshaped from wide to long format |
| Electricity production / mix | [Our World in Data](https://ourworldindata.org/electricity-mix) | Used to derive green and non-green electricity shares |
| GNI per capita | [Wikipedia / World Bank thresholds](https://en.wikipedia.org/wiki/List_of_countries_by_GNI_(nominal)_per_capita) | Used for income-group assignment |
| Energy use per capita | World Bank XML API | Used in the extended exploratory energy-use section |

## Tech Stack

This project is designed to showcase practical data-analysis skills in Python. Depending on the current notebook version, the workflow may combine:

- **Pandas / Polars** for tabular wrangling and reshaping
- **NumPy** for numeric operations and derived metrics
- **Matplotlib / Seaborn** for visual analysis
- **scikit-learn** for scaling, clustering, and exploratory ML-style preprocessing
- **lxml / pandas XML or HTML readers** for external data ingestion

The focus is less on model performance and more on analytical clarity, data preparation, and interpretable structure in the data.

## Project Structure

```text
carbon_analysis/
├── data/
│   ├── co2_data.csv
│   └── gdp_data.csv
├── data_analytics_project.ipynb
├── functions.py
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+

### Setup

```bash
git clone https://github.com/fabianfrank-ai/carbon_analysis
cd carbon_analysis

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Running the Notebook

```bash
jupyter notebook data_analytics_project.ipynb
```

## Reproducibility Notes

- Some notebook sections rely on **live external data fetching** such as Wikipedia tables, Our World in Data CSV endpoints, or World Bank API responses.
- Because of that, **internet access may be required** to run the full notebook end to end.
- External tables or APIs may change structure over time, so some ingestion steps may need small adjustments in the future.
- Income groups are assigned using the available GNI-per-capita source used in the notebook; this is useful for comparison, but should not be interpreted as a perfect historical classification for every year in the panel.

## Why This Project Fits Data Analytics

This repository is intended to demonstrate:

- multi-source dataset handling
- reshaping and merging on composite keys
- groupby-based transformation and aggregation
- feature engineering for comparability
- visual communication of quantitative patterns
- exploratory clustering and preprocessing workflows
- careful interpretation of real-world, imperfect data

## Next Steps

Planned improvements include:

- strengthening the missing-data and preprocessing section
- improving documentation of the more advanced exploratory analyses
- refining the clustering / dimensionality-reduction interpretation
- expanding reproducibility and dependency documentation
- polishing the notebook narrative for professor review

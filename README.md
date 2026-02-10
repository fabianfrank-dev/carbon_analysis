# Carbon Emissions and Economic Development: A Visual Analysis

> **Work in progress** -- this project is actively being developed. New datasets, analyses, and visualizations will be added over time.

## Overview

Can economic growth happen without increasing carbon emissions? This project investigates the **coupling and decoupling** of GDP growth and CO2 emissions across countries, using publicly available data from 1960 to 2024.

By computing per-capita metrics and Pearson correlations over time, countries are classified along a spectrum -- from those where growth and emissions move in lockstep (strong coupling) to those where GDP rises while emissions fall (decoupling). The analysis then examines whether this pattern is systematic across income groups.

## Key Findings (so far)

- **High-income countries** show the widest spread of GDP-CO2 correlations, with many achieving meaningful decoupling
- **Lower-income countries** cluster around positive correlations, indicating that growth and emissions still move together
- Historical emission trajectories follow an **inverted-U pattern** for wealthy nations, consistent with the Environmental Kuznets Curve
- **Carbon intensity** (CO2 per $1M GDP) has declined across all income groups, but large disparities remain
- Countries with higher shares of **green electricity** tend to have lower carbon intensity, though notable exceptions exist (e.g., Bermuda, South Sudan)

## Data Sources

| Dataset | Source | Coverage |
|---|---|---|
| CO2 Emissions | [Our World in Data](https://ourworldindata.org/co2-emissions) | 1750--2024 (filtered to post-1960) |
| GDP | [World Bank](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD) | 1960--2024 |
| Electricity Production | [Our World in Data](https://ourworldindata.org/electricity-mix) | 2000--2024 |
| GNI per Capita | [Wikipedia / World Bank](https://en.wikipedia.org/wiki/List_of_countries_by_GNI_(nominal)_per_capita) | Latest available year |

## Project Structure

```
carbon_analysis/
├── data/
│   ├── co2_data.csv              # Our World in Data CO2 dataset
│   └── gdp_data.csv              # World Bank GDP dataset
├── data_analytics_project.ipynb  # Main analysis notebook
├── functions.py                  # Helper functions (correlation, normalization)
├── requirements.txt              # Python dependencies
└── README.md
```

## Analysis Pipeline

1. **Data loading & cleaning** -- Select relevant columns, filter out aggregate entities (continents, "World"), align time ranges
2. **Merging** -- Left join CO2 and GDP data on ISO country code + year; visualize missing data
3. **Per-capita normalization** -- Compute CO2 per capita and GDP per capita for fair cross-country comparison
4. **Correlation analysis** -- Pearson r between GDP/capita and CO2/capita per country over time
5. **Income group classification** -- Assign World Bank income brackets (Low / Lower-Middle / Upper-Middle / High) based on GNI thresholds
6. **Comparative visualizations** -- Violin plots, bar charts, faceted time series, and dual-axis plots
7. **Carbon intensity** -- CO2 per $1M GDP to measure economic efficiency
8. **Electricity mix** -- Green vs. non-green electricity share for the most and least carbon-efficient economies

## Setup

### Prerequisites

- Python 3.10+

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd carbon_analysis

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

Open the Jupyter notebook:

```bash
jupyter notebook data_analytics_project.ipynb
```

## Dependencies

- pandas
- numpy
- seaborn
- matplotlib
- lxml

## Roadmap

This project is in its early stages. Planned additions include:

- [ ] Deeper analysis of electricity production and its impact on emissions
- [ ] Time-series forecasting of emission trends
- [ ] Regional breakdowns and geographic visualizations
- [ ] Policy impact analysis (e.g., carbon pricing, Paris Agreement effects)

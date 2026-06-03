# Carbon Emissions and Economic Development

Data Analytics course project analyzing whether countries can grow economically while reducing carbon intensity or weakening the link between GDP growth and CO2 emissions.

## Submission Guide

The recommended grading path is the pandas version:

1. Open `notebooks/pandas/05_summary_and_key_findings.ipynb` for the consolidated story, plots, and final interpretation.
2. Open the four supporting pandas notebooks for the full workflow:
   - `notebooks/pandas/01_data_prep_and_core_relationships.ipynb`
   - `notebooks/pandas/02_income_groups_and_carbon_efficiency.ipynb`
   - `notebooks/pandas/03_electricity_mix_and_energy_use.ipynb`
   - `notebooks/pandas/04_clustering_and_cluster_dynamics.ipynb`
3. Treat the polars notebooks as an alternative implementation / extension of the same analysis.

The summary notebook is intentionally the main submission artifact because it can be reviewed quickly while still showing the main data preparation, feature engineering, visualizations, clustering, and conclusions.

## Research Question

**Can countries grow economically while reducing their carbon intensity or weakening the link between GDP growth and CO2 emissions?**

The project answers this through exploratory analysis rather than causal inference. It identifies patterns across countries and income groups, but it does not claim that any single variable fully explains national emissions outcomes.

## Analytical Workflow

The project demonstrates a full data analytics workflow:

1. **Data loading and cleaning**: CO2 and GDP data are loaded from local CSV files and filtered to country-year observations.
2. **Reshaping and merging**: World Bank GDP data is reshaped from wide to long format and merged with emissions data using ISO country codes and years.
3. **Feature engineering**: Per-capita and efficiency metrics are created, including `co2_per_capita`, `gdp_per_capita`, and `co2_per_gdp`.
4. **Missing-data inspection**: Missingness is visualized and summarized after the merge.
5. **Relationship analysis**: GDP per capita and CO2 per capita are compared over time using country-level Pearson correlations.
6. **Income-group comparison**: Countries are grouped by income level to compare emissions trajectories and carbon efficiency.
7. **Energy-structure analysis**: Electricity mix and energy-use measures are integrated to inspect whether cleaner energy shares align with lower carbon intensity.
8. **Unsupervised learning**: KMeans clustering, PCA, and UMAP-style dimensionality reduction are used to explore country archetypes.
9. **Interpretation**: Results are summarized with limitations and policy-relevant insights.

## Key Findings

- High-income countries show the widest range of GDP-CO2 relationships, including more cases consistent with relative decoupling.
- Lower-income countries more often show positive GDP-CO2 relationships, suggesting that growth and emissions remain more tightly linked.
- Carbon intensity generally declines over time, but the speed and starting point differ substantially across income groups.
- Electricity mix matters, but it does not fully explain differences in carbon efficiency by itself.
- Clustering reveals country archetypes that combine economic scale, emissions intensity, energy structure, and outlier behavior.

## Data Sources

| Dataset | Source | Use |
|---|---|---|
| CO2 emissions | [Our World in Data](https://ourworldindata.org/co2-emissions) | Main emissions panel and population values |
| GDP | [World Bank](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD) | National GDP values reshaped from wide to long format |
| Electricity production / mix | [Our World in Data](https://ourworldindata.org/electricity-mix) | Green and non-green electricity share analysis |
| GNI per capita | Wikipedia table based on World Bank-style reporting | Income-group assignment for comparison |
| Energy use per capita | World Bank XML API | Energy-use extension section |

## Project Structure

```text
carbon_analysis/
├── data/
│   ├── co2_data.csv
│   └── gdp_data.csv
├── notebooks/
│   ├── functions.py
│   ├── pandas/
│   │   ├── 01_data_prep_and_core_relationships.ipynb
│   │   ├── 02_income_groups_and_carbon_efficiency.ipynb
│   │   ├── 03_electricity_mix_and_energy_use.ipynb
│   │   ├── 04_clustering_and_cluster_dynamics.ipynb
│   │   ├── 05_summary_and_key_findings.ipynb
│   │   └── README.md
│   └── polars/
│       ├── 01_data_prep_and_core_relationships.ipynb
│       ├── 02_income_groups_and_carbon_efficiency.ipynb
│       ├── 03_electricity_mix_and_energy_use.ipynb
│       ├── 04_clustering_and_dimensionality_reduction.ipynb
│       └── README.md
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/pandas/05_summary_and_key_findings.ipynb
```

On Windows, activate the environment with:

```bash
.venv\Scripts\activate
```

## Reproducibility Notes

- The summary notebook and early data-preparation sections run from the local files in `data/`.
- Some supporting notebooks fetch live external data from Wikipedia, Our World in Data, and the World Bank API.
- Internet access is therefore required for the full extended workflow.
- External web tables and APIs can change structure over time; if that happens, the local CSV-based summary remains the most stable submission artifact.
- Income groups are assigned from currently available country-level income information, so they should be interpreted as a practical comparison framework rather than a perfect historical classification.

## Why This Fits Data Analytics

This project demonstrates:

- multi-source data handling
- wide-to-long reshaping
- country-year panel merging
- missing-data inspection
- feature engineering for comparability
- visual analysis and interpretation
- grouped aggregation and ranking
- unsupervised clustering and dimensionality reduction
- careful limitations around exploratory analysis

## AI Use Disclosure

AI assistance was used for brainstorming, debugging, code-quality feedback, library usage questions, and improving explanatory wording. The analysis choices, implementation review, interpretation, and final submission decisions were checked by the author, and the work is presented as the author's own course project.

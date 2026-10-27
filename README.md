# Credit Risk PD Model — Home Credit Default Risk

This project develops a probability of default (PD) model on the Home
Credit Default Risk dataset, framed around credit-risk industry practice
and regulatory context.

The Weight of Evidence (WoE) scorecard is the industry-standard PD
modeling technique in retail credit risk, favored for its interpretability
and acceptance under Basel IRB and IFRS 9 requirements. It serves as the
primary model here. A LightGBM gradient-boosting challenger is added
alongside, with SHAP (SHapley Additive exPlanations) for global and local
explainability. An IFRS 9 Expected Credit Loss (ECL) appendix illustrates
how PD estimates translate into accounting-relevant risk indicators.

The project is structured around the CRISP-DM (Cross-Industry Standard
Process for Data Mining) framework, covering Business Understanding, Data
Understanding, Data Preparation, Modeling, and Evaluation. The Deployment
phase is out of scope: this is a demonstration model, not a
production-ready system.

**Status:** Work in progress.

## Structure

- `notebooks/01_eda.ipynb`: Exploratory data analysis
- `notebooks/02_data_preparation.ipynb`: Cleaning, feature engineering and train/test split
- `notebooks/03_scorecard_woe_lr.ipynb`: WoE binning and logistic regression scorecard *(planned)*
- `notebooks/04_challenger_lightgbm.ipynb`: LightGBM challenger and SHAP *(planned)*
- `notebooks/05_ifrs9_ecl.ipynb`: IFRS 9 Expected Credit Loss appendix *(planned)*
- `notebooks/06_validation.ipynb`: Model validation and fairness testing *(planned)*
- `configs/`: Fitted preparation parameters and modelling settings
- `src/`: Reusable feature engineering, modeling and evaluation code
- `reports/`: Model validation report and executive summary

## Ethical and regulatory considerations

Credit scoring is classified as a **high-risk** application under the
EU AI Act (Annex III), and automated credit decisions fall under
GDPR Article 22, which grants data subjects the right to meaningful
information about the logic involved. These considerations shape the
modeling choices in this project:

- **Interpretability first.** The WoE scorecard is chosen as the primary
  model because point-scaled logistic regression is auditable at the
  feature level, which is a regulatory expectation rather than a
  preference.
- **Protected attributes excluded.** `CODE_GENDER` is not used as a
  predictor, since EU Directive 2004/113/EC prohibits sex discrimination in
  access to goods and services. It is retained only to test the model's
  outputs for gender disparities.
- **Explainability for challengers.** SHAP is applied to the LightGBM
  model to make its predictions defensible at both the population and
  individual level.
- **Proxy discrimination.** Features that correlate with protected
  attributes (gender, age, national origin) can encode indirect bias
  even when protected attributes are excluded. A brief fairness
  discussion is included in the validation notebook (06).
- **Scope of this project.** This is a demonstration model built on a
  public dataset. It is not intended for deployment and would require
  formal model validation, independent review, and fairness testing
  before any real credit decision.

## Data

Source: [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk).

Raw data is not committed. 

See "Reproducibility" below.

## Reproducibility

Requires Python 3.12+ and a Kaggle API token.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
kaggle competitions download -c home-credit-default-risk -p data/raw
cd data/raw && tar -xf home-credit-default-risk.zip
```

Run the notebooks in numerical order: each one reads the output of the
previous notebook from `data/processed/`.
# Cross-Sectional Asset Pricing with Fama–MacBeth (Streamlit)

## Overview
This project implements a **cross-sectional asset pricing framework** using the **Fama–MacBeth (1973) methodology** to study why some assets earn higher returns than others. The application is built in **Streamlit** and uses **Fama–French 25 Size × Book-to-Market portfolios** together with standard factor models (FF3, FF5, FF5 + Momentum).

The goal of the project is **not trading or alpha discovery**, but to **replicate and demonstrate academic asset-pricing methods** used in quantitative research, portfolio management, and risk modeling.

---

## Why this project is relevant
A central question in finance is whether differences in expected returns across assets can be explained by **systematic risk exposure**. Cross-sectional asset pricing provides a structured way to test this idea by asking:

> Do assets that are more exposed to certain risk factors earn higher returns?

The methods implemented here are widely used in:
- Academic finance research  
- Quantitative asset management  
- Factor-based portfolio construction  
- Financial risk and performance attribution  

Understanding these techniques is foundational for **quantitative research, asset pricing, and risk-focused roles**.

---

## How it works (high level)

The app follows the standard **two-step Fama–MacBeth procedure**:

### 1. Estimate risk exposure (First Pass)
For each asset (portfolio), the app estimates **factor betas** using rolling time-series regressions of excess returns on factor returns. This captures how sensitive each asset is to common sources of risk.

### 2. Estimate prices of risk (Second Pass)
Each month, the app runs a **cross-sectional regression** of asset returns on the estimated betas. This produces a time series of **factor risk premia**, showing how much the market rewards each type of risk.

### Statistical inference
To ensure valid inference, **Newey–West standard errors** are used when computing average risk premia, accounting for autocorrelation and heteroskedasticity.

### Portfolio-based validation
To give the results an economic interpretation, the app builds a **long–short portfolio** based on predicted returns:
- Long assets with the highest predicted returns
- Short assets with the lowest predicted returns

Performance metrics such as Sharpe ratio and drawdown are reported.

---

## Data used
- **Assets:** Fama–French 25 Size × Book-to-Market portfolios  
- **Frequency:** Monthly  
- **Factors:** FF3, FF5, and Momentum (when available)  
- **Risk-free rate:** Included in factor data  

Using benchmark portfolios avoids survivorship and delisting biases and aligns the project with academic standards.

---

## Robustness checks

### Why robustness matters
Asset pricing results can be sensitive to modeling choices such as estimation windows or sample periods. Robustness checks help ensure that results are **not driven by a single assumption or market episode**.

If a factor is genuinely priced, its estimated effect should be **reasonably stable across different specifications**.

### What robustness checks are included

#### 1. Rolling window sensitivity
The Fama–MacBeth procedure is repeated using different rolling windows (e.g., 36, 60, 120 months) to estimate betas.

This checks whether:
- Results depend heavily on a specific window length
- Risk premia are driven by short-term noise

Stable estimates across windows indicate more reliable results.

#### 2. Subsample analysis (Pre/Post split)
The sample is split into two periods around a user-defined year (e.g., before and after 2008).

This tests whether:
- Risk premia change across market regimes
- Factor pricing is consistent before and after major events

Consistency across subsamples suggests structural stability.

### How to interpret robustness results
- **Similar signs and magnitudes across windows/subsamples** → strong evidence of pricing  
- **Large swings or sign changes** → possible instability or regime dependence  
- **Consistently insignificant results** → limited pricing power  

Robustness results are not expected to be identical, but they should be **directionally consistent** if the model is reliable.

---

## How to use the app

### 1. Run the application
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 2. Select model options
- Choose a factor model: FF3, FF5, or FF5 + Momentum  
- Set the rolling window length  
- Choose Newey–West lag length  
- Set the long–short portfolio cutoff  

### 3. View results
The app displays:
- Estimated factor risk premia and t-statistics  
- Time-series plots of prices of risk  
- Long–short portfolio performance  
- Robustness results across windows and subsamples  

### 4. Download outputs
You can download:
- Fama–MacBeth regression results  
- Newey–West summary tables  
- Long–short portfolio returns  
- Robustness check results  

---

## Assumptions and limitations
- Uses portfolios instead of individual stocks, limiting firm-level interpretation  
- No transaction costs are applied  
- Results depend on the chosen factor model  
- Focuses on explanatory power, not return prediction  

These limitations are consistent with standard academic asset-pricing exercises.

---

## Repository structure
```
.
├── app.py
├── requirements.txt
└── README.md
```

---

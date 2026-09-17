
"""
Statify v2.0 — AI-Era Statistical Analysis Platform
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single-file version. Copy this entire file to your GitHub repo as app.py.
No other files needed.
"""
from __future__ import annotations
import io, uuid, json, zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as sp_stats


from theme import inject_global_css
from io_layer import load_file
from profiler import get_column_types, summarize_dataframe
from cleaning import handle_missing, convert_dtype, rename_column, delete_row, delete_column, update_cell
from tool_registry import STAT_TOOLS, get_tool, list_tools_by_category
from recommender import compatible_tools_for
from validator import _validate
from stats_engine import run_analysis
from graph_engine import GRAPH_CATALOG, list_graphs_for_tool, build_graph




APP_CONFIG = {
    "name": "Statify", "version": "2.0.0", "icon": "📊",
    "tagline": "AI-Era Statistical Analysis Platform",
    "max_analyses": 8,
}
PAGES = {
    1: {"title": "Upload",    "icon": "📤"},
    2: {"title": "Editor",    "icon": "✏️"},
    3: {"title": "Builder",   "icon": "🧪"},
    4: {"title": "Dashboard", "icon": "🗂"},
    5: {"title": "Report",    "icon": "📑"},
}

# ═══════════════════════════════════════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════════
# FILE LOADER
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN TYPES
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# STATS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

"""
Statistical-tool registry.

Each entry declares:
  - category
  - description
  - variable role contracts (which roles, which semantic types are accepted, cardinality)
  - parameters (with defaults & types)
  - assumptions (for reporting)
  - recommended graphs (for Page 4 graph picker)
"""



# ═══════════════════════════════════════════════════════════════════════════════
# STATS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

"""
Statistical execution engine.

Each tool dispatches to a small isolated function returning a dict:
{
    "summary": dict[str, scalar],
    "tables":  dict[str, pandas.DataFrame],
    "extras":  dict (anything graph engine might need),
    "p_value": float | None,
    "alpha":   float,
    "ok": bool,
    "error": str (optional),
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════


        
# ═══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
Dynamic interpretation engine.
Generates plain-language explanations from actual numbers.
"""


def _sig(p, alpha):
    return p is not None and p < alpha


def _strength_r(r):
    a = abs(r)
    if a < 0.1:  return "negligible"
    if a < 0.3:  return "weak"
    if a < 0.5:  return "moderate"
    if a < 0.7:  return "strong"
    return "very strong"


def _direction(r):
    return "positive" if r >= 0 else "negative"


def _wrap(headline, explanation, statistical, assumptions, conclusion):
    return {"headline": headline, "explanation": explanation,
            "statistical": statistical, "assumptions": assumptions,
            "conclusion": conclusion}


def _d_size(d):
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


def _v_size(v):
    if v < 0.1: return "weak"
    if v < 0.3: return "moderate"
    return "strong"


def interpret_result(tool: str, result: dict) -> dict:
    """Returns {'headline','explanation','statistical','assumptions','conclusion'}."""
    if not result.get("ok", True):
        return {"headline": "Analysis failed",
                "explanation": result.get("error","Unknown error"),
                "statistical": "", "assumptions": "", "conclusion": ""}

    s = result.get("summary", {})
    p = result.get("p_value"); a = result.get("alpha", 0.05)

    # ---------- DESCRIPTIVE ----------
    if tool == "Descriptive Statistics":
        return _wrap(
            f"Summarized {s.get('variables',0)} numeric variables across {s.get('n',0):,} observations.",
            "Descriptive statistics show central tendency (mean/median), spread (std), and distribution shape (skew/kurtosis) for every selected variable.",
            "No hypothesis is tested — these are summary measures only.",
            "—",
            "Use these summaries to understand each variable's range, typical value, and asymmetry before deeper analysis.")

    if tool == "Frequency Table":
        return _wrap(
            f"`{s.get('variable','')}` has {s.get('categories',0)} unique categories across {s.get('n',0):,} cases.",
            "Frequency counts and percentages reveal how observations distribute across categorical levels.",
            "Descriptive only — no significance test.",
            "—",
            "Identify dominant categories and detect rare classes that may need merging.")

    if tool == "Cross Tabulation":
        return _wrap(
            f"Cross-tabulation of {s.get('rows',0)}×{s.get('cols',0)}.",
            "Each cell shows the joint frequency of the two categorical variables.",
            "Use Chi-Square to test independence formally.",
            "—",
            "Compare row/column distributions to spot association patterns.")

    if tool == "Outlier Detection (IQR)":
        return _wrap(
            f"Found {s.get('outliers',0)} outliers in {s.get('n',0):,} observations.",
            f"Values outside [{s.get('lower')}, {s.get('upper')}] are flagged using the 1.5 × IQR rule.",
            "—", "—",
            "Investigate flagged points: data-entry errors, true extremes, or natural variation.")

    if tool == "Skewness & Kurtosis":
        return _wrap(
            "Distribution shape diagnostics computed for each numeric variable.",
            "Skewness measures asymmetry (0 = symmetric, >0 = right tail, <0 = left tail). Kurtosis measures tail heaviness (3 = normal, >3 = heavy tails).",
            "Use these to decide whether to apply transformations or non-parametric tests.",
            "—", "Variables with |skew| > 1 or |kurtosis| > 3 warrant attention.")

    # ---------- HYPOTHESIS ----------
    if tool == "One-Sample t-Test":
        sig = _sig(p, a)
        return _wrap(
            f"Mean of {s.get('mean',0):.3f} {'differs significantly' if sig else 'does not differ significantly'} from μ₀ = {s.get('mu0',0):.3f} (t = {s.get('t',0):.3f}, p = {p:.4f}).",
            "The one-sample t-test compares the observed sample mean against a fixed reference value.",
            f"With α = {a}, p {'<' if sig else '≥'} α, so we {'reject' if sig else 'fail to reject'} H₀.",
            "Approximate normality and independent observations.",
            ("Strong evidence the population mean differs from the reference." if sig
             else "Insufficient evidence to claim a difference from the reference value."))

    if tool == "Independent t-Test":
        sig = _sig(p, a); d = s.get("cohens_d", 0)
        return _wrap(
            f"{'Significant' if sig else 'No significant'} mean difference between groups "
            f"({s.get('mean1',0):.2f} vs {s.get('mean2',0):.2f}, t = {s.get('t',0):.3f}, p = {p:.4f}, d = {d:.3f}).",
            "Compares means of two independent groups; Cohen's d quantifies practical effect size.",
            f"At α = {a}, the result is {'statistically significant' if sig else 'not significant'}. "
            f"Effect size is {_d_size(d)}.",
            "Independent observations · Approximately normal data · Equal variance (if Student's t).",
            ("The two groups likely come from populations with different means." if sig
             else "No reliable evidence that the two groups differ."))

    if tool == "Paired t-Test":
        sig = _sig(p, a); d = s.get("cohens_d", 0)
        return _wrap(
            f"Paired difference is {'significant' if sig else 'not significant'} (t = {s.get('t',0):.3f}, p = {p:.4f}, d = {d:.3f}).",
            "Tests whether the mean change between paired measurements differs from zero.",
            f"At α = {a}, {'reject' if sig else 'fail to reject'} H₀. Effect size: {_d_size(d)}.",
            "Differences are approximately normal · Observations are paired.",
            ("There is a meaningful change between the two measurements." if sig
             else "No reliable change detected between the two measurements."))

    if tool == "Z-Test (One Sample)":
        sig = _sig(p, a)
        return _wrap(
            f"Z = {s.get('z',0):.3f}, p = {p:.4f} — {'significant' if sig else 'not significant'}.",
            "Tests sample mean against a hypothesized value when population σ is known.",
            f"At α = {a}, {'reject' if sig else 'fail to reject'} H₀.",
            "Population variance known · Large sample.",
            "Mean differs from reference." if sig else "No evidence mean differs.")

    if tool == "Proportion Z-Test":
        sig = _sig(p, a)
        return _wrap(
            f"Sample proportion {s.get('p_hat',0):.3f} vs reference {s.get('p0',0):.3f} (z = {s.get('z',0):.3f}, p = {p:.4f}).",
            "Tests whether observed proportion differs from a hypothesized value.",
            f"At α = {a}, {'reject' if sig else 'fail to reject'} H₀.",
            "Independent binary trials.",
            "Proportion is significantly different." if sig else "Proportion not significantly different.")

    if tool == "Chi-Square Test of Independence":
        sig = _sig(p, a); v = s.get("cramers_v",0)
        return _wrap(
            f"χ² = {s.get('chi2',0):.3f}, df = {s.get('df',0)}, p = {p:.4f}, Cramér's V = {v:.3f}.",
            "Tests whether two categorical variables are statistically independent.",
            f"At α = {a}, the variables are {'significantly associated' if sig else 'not significantly associated'}. "
            f"Effect size (Cramér's V) is {_v_size(v)}.",
            "Independent observations · Expected counts ≥ 5 in most cells.",
            "The two variables are related." if sig else "No evidence of association.")

    if tool == "Chi-Square Goodness of Fit":
        sig = _sig(p, a)
        return _wrap(
            f"χ² = {s.get('chi2',0):.3f}, p = {p:.4f}.",
            "Tests whether observed counts deviate from an expected (uniform) distribution.",
            f"At α = {a}, distribution {'differs' if sig else 'matches'} expected.",
            "Independent obs · Expected counts ≥ 5.",
            "Categories are not uniformly distributed." if sig else "No deviation from uniform distribution.")

    if tool == "Fisher's Exact Test":
        sig = _sig(p, a); odds = s.get("odds_ratio",1)
        return _wrap(
            f"Odds ratio = {odds:.3f}, p = {p:.4f}.",
            "Exact test for 2×2 tables; preferred when expected counts are small.",
            f"At α = {a}, {'reject' if sig else 'fail to reject'} H₀. Odds ratio of {odds:.2f} indicates "
            f"{'higher' if odds>1 else 'lower'} odds in the first category.",
            "Binary categorical variables.",
            "Significant association detected." if sig else "No significant association.")

    # ---------- CORRELATION ----------
    if tool in ("Pearson Correlation","Spearman Correlation","Kendall's Tau"):
        r = s.get("r",0); sig = _sig(p, a)
        return _wrap(
            f"A {_strength_r(r)} {_direction(r)} relationship (r = {r:.3f}) "
            f"{'is statistically significant' if sig else 'was observed but is not significant'} (p = {p:.4f}, n = {s.get('n',0):,}).",
            f"{tool} measures association strength between two numeric variables on a [-1, +1] scale.",
            f"At α = {a}, the correlation is {'significant' if sig else 'not significant'}.",
            "Approximately monotonic relationship (Spearman) or linear & bivariate-normal (Pearson).",
            ("As one variable increases, the other tends to "
             f"{'increase' if r>=0 else 'decrease'} — this association is reliable."
             if sig else "Insufficient evidence of a true association in the population."))

    if tool == "Correlation Matrix":
        return _wrap(
            f"Pairwise {s.get('method','pearson')} correlations across {s.get('variables',0)} variables (n = {s.get('n',0):,}).",
            "The matrix shows linear (Pearson) or rank-based (Spearman/Kendall) associations among all pairs.",
            "Inspect cells with |r| > 0.5 for substantive relationships; |r| > 0.8 may indicate multicollinearity.",
            "Same as the chosen method's pairwise assumptions.",
            "Identifies clusters of related variables and potential redundancy.")

    if tool == "Point-Biserial Correlation":
        r = s.get("r_pb",0); sig = _sig(p, a)
        return _wrap(
            f"r_pb = {r:.3f} ({_strength_r(r)} {_direction(r)}), p = {p:.4f}.",
            "Correlation between a binary and a continuous variable.",
            f"At α = {a}, {'significant' if sig else 'not significant'}.",
            "—", "Means likely differ across groups." if sig else "Means do not differ meaningfully.")
# ---------- REGRESSION ----------
    if tool == "Simple Linear Regression":
        sig = _sig(p, a); r2 = s.get("r2",0)
        return _wrap(
            f"The model explains {r2*100:.1f}% of variance in Y (R² = {r2:.3f}, slope = {s.get('slope',0):.3f}, p = {p:.4f}).",
            "Simple linear regression models Y as a straight-line function of X.",
            f"Slope is {'significantly different from zero' if sig else 'not significant'} at α = {a}. "
            f"Each unit increase in X is associated with a {s.get('slope',0):.3f}-unit "
            f"{'increase' if s.get('slope',0)>=0 else 'decrease'} in Y.",
            "Linearity · Normal residuals · Constant variance · Independence.",
            (f"X is a useful predictor of Y, accounting for {r2*100:.0f}% of its variability."
             if sig else "X does not reliably predict Y."))

    if tool == "Multiple Linear Regression":
        sig = _sig(p, a); r2 = s.get("r2",0); ar = s.get("adj_r2",0)
        return _wrap(
            f"Model R² = {r2:.3f} (Adjusted R² = {ar:.3f}), F-test p = {p:.4f}.",
            "Multiple regression models Y as a linear combination of several predictors.",
            f"Overall model is {'significant' if sig else 'not significant'} at α = {a}. "
            f"Inspect each coefficient's p-value to identify reliable predictors.",
            "Linearity · No multicollinearity · Normal residuals · Constant variance.",
            (f"Predictors jointly explain {r2*100:.0f}% of Y. Use coefficient signs and p-values for variable importance."
             if sig else "The predictors collectively do not explain Y."))

    if tool == "Polynomial Regression":
        r2 = s.get("r2",0)
        return _wrap(
            f"Polynomial of degree {s.get('degree',2)} fits with R² = {r2:.3f}.",
            "Polynomial regression captures non-linear (curved) relationships.",
            f"Higher R² than linear regression suggests curvature in the relationship.",
            "Polynomial form is appropriate.",
            f"The relationship is non-linear; polynomial captures {r2*100:.0f}% of variance.")

    if tool == "Logistic Regression":
        acc = s.get("accuracy",0); auc = s.get("auc", float("nan"))
        return _wrap(
            f"Classification accuracy = {acc*100:.1f}%; AUC = {auc:.3f}.",
            "Logistic regression models the probability of a binary outcome as a logistic function of predictors.",
            f"AUC > 0.7 indicates acceptable discrimination; > 0.8 is good; > 0.9 is excellent.",
            "Binary outcome · Independence · No multicollinearity.",
            ("Predictors successfully discriminate between the two classes."
             if (auc==auc and auc>0.7) else "Discrimination is weak — consider feature engineering."))

    if tool in ("Ridge Regression","Lasso Regression"):
        r2 = s.get("r2",0)
        return _wrap(
            f"{tool} achieves R² = {r2:.3f} on training data.",
            f"{'Ridge (L2) penalizes large coefficients to reduce variance' if 'Ridge' in tool else 'Lasso (L1) shrinks some coefficients to exactly zero, performing feature selection'}.",
            "Lower training R² than OLS is expected — gains come from better generalization.",
            "Standardized features recommended.",
            "Use these regularized models when predictors are correlated or when you want sparser models.")

    # ---------- ANOVA ----------
    if tool == "One-Way ANOVA":
        sig = _sig(p, a)
        return _wrap(
            f"F = {s.get('F',0):.3f}, p = {p:.4f} across {s.get('groups',0)} groups (n = {s.get('n',0):,}).",
            "One-way ANOVA tests whether group means differ.",
            f"At α = {a}, group means {'differ significantly' if sig else 'do not differ significantly'}.",
            "Normality · Equal variance · Independence.",
            ("At least one group mean differs from the others — run Tukey HSD for pairwise comparisons."
             if sig else "No reliable difference detected across the groups."))

    if tool == "Two-Way ANOVA":
        return _wrap(
            f"Model R² = {s.get('r2',0):.3f}, n = {s.get('n',0):,}.",
            "Two-way ANOVA assesses main effects of two categorical factors and their interaction.",
            "Inspect each row's p-value: significant interaction implies the effect of one factor depends on the other.",
            "Normality · Equal variance.", "Examine main effects and interaction term separately.")

    if tool == "Repeated Measures ANOVA":
        sig = _sig(p, a)
        return _wrap(
            f"F = {s.get('F',0):.3f}, p = {p:.4f}, k = {s.get('k',0)} conditions.",
            "Tests whether repeated measurements differ across time points or conditions within subjects.",
            f"At α = {a}, conditions {'differ' if sig else 'do not differ'} significantly.",
            "Sphericity · Normality.", "Run pairwise contrasts to identify which conditions differ." if sig else "No condition effects detected.")

    if tool == "ANCOVA":
        return _wrap(
            f"Model R² = {s.get('r2',0):.3f}.",
            "ANCOVA tests group differences while adjusting for a continuous covariate.",
            "Check the C(group) row's p-value for the adjusted group effect.",
            "Linearity of covariate · Homogeneity of regression slopes.",
            "Group differences after controlling for the covariate.")

    if tool == "Tukey HSD Post-Hoc":
        return _wrap(
            f"Pairwise comparisons across groups at α = {s.get('alpha',0.05)}.",
            "Tukey HSD identifies which specific group pairs differ after a significant ANOVA.",
            "Rows marked 'reject = True' indicate significant pairwise differences.",
            "Same as ANOVA.", "Use this to pinpoint where the ANOVA effect lies.")

    # ---------- NON-PARAMETRIC ----------
    if tool == "Mann-Whitney U Test":
        sig = _sig(p, a)
        return _wrap(
            f"U = {s.get('U',0):.1f}, p = {p:.4f}.",
            "Non-parametric alternative to the independent t-test based on ranks.",
            f"At α = {a}, distributions {'differ' if sig else 'do not differ'}.",
            "Independent observations.",
            "The two groups have different distributions." if sig else "No distributional difference detected.")

    if tool == "Wilcoxon Signed-Rank Test":
        sig = _sig(p, a)
        return _wrap(
            f"W = {s.get('W',0):.1f}, p = {p:.4f}, n = {s.get('n',0):,}.",
            "Non-parametric paired test on signed differences.",
            f"At α = {a}, paired differences are {'significant' if sig else 'not significant'}.",
            "Symmetric differences · Paired observations.",
            "Significant change between paired measures." if sig else "No reliable change detected.")

    if tool == "Kruskal-Wallis Test":
        sig = _sig(p, a)
        return _wrap(
            f"H = {s.get('H',0):.3f}, p = {p:.4f} across {s.get('groups',0)} groups.",
            "Non-parametric alternative to one-way ANOVA.",
            f"At α = {a}, group distributions {'differ' if sig else 'do not differ'}.",
            "Independent observations.",
            "At least one group differs — follow up with pairwise tests." if sig else "No group differences detected.")

    if tool == "Friedman Test":
        sig = _sig(p, a)
        return _wrap(
            f"χ² = {s.get('chi2',0):.3f}, p = {p:.4f}, k = {s.get('k',0)} conditions.",
            "Non-parametric alternative to repeated-measures ANOVA.",
            f"At α = {a}, conditions {'differ' if sig else 'do not differ'}.",
            "Related observations.",
            "Conditions differ — run pairwise contrasts." if sig else "No reliable differences across conditions.")

    if tool == "Shapiro-Wilk Normality":
        sig = _sig(p, a)
        return _wrap(
            f"W = {s.get('W',0):.4f}, p = {p:.4f}.",
            "Tests whether data come from a normal distribution.",
            f"At α = {a}, the data are {'NOT normally' if sig else 'plausibly normally'} distributed.",
            "—",
            ("Use non-parametric methods or transform the data." if sig
             else "Parametric methods (t-tests, ANOVA) are appropriate."))

    if tool == "Levene's Test":
        sig = _sig(p, a)
        return _wrap(
            f"W = {s.get('W',0):.3f}, p = {p:.4f}.",
            "Tests equality of variances across groups.",
            f"At α = {a}, group variances {'differ' if sig else 'are roughly equal'}.",
            "—",
            ("Use Welch's t-test or non-parametric alternatives." if sig
             else "Standard parametric tests are appropriate."))

    if tool == "Kolmogorov-Smirnov Test":
        sig = _sig(p, a)
        return _wrap(
            f"D = {s.get('D',0):.4f}, p = {p:.4f}.",
            "Compares the empirical CDF to a normal distribution.",
            f"At α = {a}, distribution {'differs from' if sig else 'is consistent with'} normal.",
            "—", "Consider transformations if non-normal." if sig else "Distribution looks normal.")

    # ---------- TIME SERIES ----------
    if tool == "Time Series Decomposition":
        return _wrap(
            f"Decomposed series with seasonal period = {s.get('period',12)}.",
            "Splits the series into trend, seasonal, and residual components for visual inspection.",
            "Strong seasonal component suggests seasonal models (SARIMA, Holt-Winters).",
            "—", "Use components to choose appropriate forecasting method.")

    if tool == "Stationarity (ADF Test)":
        sig = _sig(p, a)
        return _wrap(
            f"ADF stat = {s.get('adf_stat',0):.3f}, p = {p:.4f}.",
            "Augmented Dickey-Fuller tests for a unit root.",
            f"At α = {a}, series is {'stationary' if sig else 'NON-stationary'}.",
            "—",
            ("Series is stationary — proceed with ARMA modeling." if sig
             else "Difference the series before modeling."))

    if tool == "ARIMA Forecast":
        return _wrap(
            f"ARIMA{s.get('order',(1,1,1))} fitted (AIC = {s.get('aic',0):.2f}).",
            "ARIMA models combine autoregression, differencing, and moving averages for forecasting.",
            "Lower AIC indicates a better fit. Cross-validate before deploying forecasts.",
            "Stationary residuals.",
            "Use the forecast table for upcoming periods; widen intervals reflect uncertainty.")

    if tool == "Exponential Smoothing":
        return _wrap(
            f"Holt-Winters fit (n = {s.get('n',0):,}).",
            "Exponential smoothing weights recent observations more heavily.",
            "Useful when level and seasonality drift slowly over time.",
            "—", "Suitable for short-horizon forecasting of smooth series.")

    if tool == "Autocorrelation (ACF/PACF)":
        return _wrap(
            f"ACF/PACF computed up to lag {s.get('lags',20)}.",
            "ACF spikes suggest MA terms; PACF spikes suggest AR terms.",
            "Use these plots to choose ARIMA(p, d, q) orders.",
            "—", "Significant lags indicate temporal structure to exploit.")

    # ---------- CLUSTERING ----------
    if tool == "K-Means Clustering":
        return _wrap(
            f"K = {s.get('k',0)} clusters · inertia = {s.get('inertia',0):.2f} · n = {s.get('n',0):,}.",
            "K-means partitions observations into K groups by minimizing within-cluster variance.",
            "Compare cluster centers to characterize each group.",
            "Roughly spherical clusters · Standardized features.",
            "Use the cluster assignments for segmentation or downstream supervised modeling.")

    if tool == "Hierarchical Clustering":
        return _wrap(
            f"{s.get('clusters',0)} clusters via {s.get('method','ward')} linkage.",
            "Builds a hierarchy of merges; cut the dendrogram at the desired number of clusters.",
            "Inspect the dendrogram to choose the natural cluster count.",
            "—", "Useful when cluster count is unknown a priori.")

    if tool == "DBSCAN":
        return _wrap(
            f"{s.get('clusters',0)} clusters identified, {s.get('noise',0)} noise points.",
            "DBSCAN finds dense regions and labels outliers as noise.",
            "Tune ε and min_samples for cleaner clusters.",
            "—", "Best when clusters have arbitrary shapes and outliers are present.")
# ---------- DIMENSIONALITY ----------
    if tool == "PCA":
        tv = s.get("total_variance",0)
        return _wrap(
            f"{s.get('n_components',2)} components capture {tv*100:.1f}% of total variance (n = {s.get('n',0):,}).",
            "PCA projects correlated variables onto orthogonal axes ordered by variance explained.",
            "Components with low variance can usually be discarded.",
            "Linear relationships · Standardized features.",
            f"Reduce dimensionality while preserving {tv*100:.0f}% of the information.")

    if tool == "Factor Analysis":
        return _wrap(
            f"{s.get('n_factors',2)} latent factors extracted (n = {s.get('n',0):,}).",
            "Factor analysis recovers latent constructs that explain shared variance.",
            "Inspect loadings: |loading| > 0.3-0.4 marks variables that load on a factor.",
            "—", "Use factor scores as compact representations.")

    # ---------- SURVIVAL ----------
    if tool == "Kaplan-Meier Survival":
        return _wrap(
            f"{s.get('events',0)} events in n = {s.get('n',0):,} observations · median time = {s.get('median_time',0):.2f}.",
            "Estimates survival probability over time, accounting for censoring.",
            "Use log-rank test to compare survival across groups.",
            "Right-censored data · Non-informative censoring.",
            "Inspect the survival curve for drop-off points.")

    if tool == "Cox Proportional Hazards":
        return _wrap(
            f"Concordance = {s.get('concordance',0):.3f} (n = {s.get('n',0):,}).",
            "Cox regression estimates hazard ratios for covariates without specifying baseline hazard.",
            "Hazard ratio > 1 means higher risk; < 1 means protective effect.",
            "Proportional hazards · Independent observations.",
            "Use HRs and their confidence intervals to identify risk factors.")

    # ---------- ML ----------
    if "Classifier" in tool or tool in ("K-Nearest Neighbors","Naive Bayes Classifier","Linear Discriminant Analysis"):
        acc = s.get("accuracy",0)
        return _wrap(
            f"Test accuracy = {acc*100:.1f}% across {len(s.get('classes',[]))} classes "
            f"(train n = {s.get('n_train',0):,}, test n = {s.get('n_test',0):,}).",
            f"{tool} learns class boundaries from labelled features and is evaluated on a held-out test set.",
            "Compare to a baseline (e.g., majority-class accuracy). Inspect confusion matrix for class-specific errors.",
            "Sufficient training samples per class.",
            ("Model performance is acceptable for many tasks." if acc > 0.7
             else "Performance is modest — consider more features, tuning, or alternative algorithms."))

    # ---------- EFFECT SIZES ----------
    if tool == "Cohen's d (Effect Size)":
        d = s.get("cohens_d",0)
        return _wrap(
            f"Cohen's d = {d:.3f} ({_d_size(d)}).",
            "Cohen's d expresses the standardized mean difference between two groups.",
            "0.2 = small · 0.5 = medium · 0.8 = large.",
            "—", f"Effect size is {_d_size(d)} — consider practical importance alongside p-values.")

    if tool == "Cramér's V":
        v = s.get("cramers_v",0)
        return _wrap(
            f"Cramér's V = {v:.3f} ({_v_size(v)}).",
            "Cramér's V is a chi-square-based effect size for nominal associations.",
            "0.1 = weak · 0.3 = moderate · 0.5+ = strong.",
            "—", f"Association strength is {_v_size(v)}.")

    # ---------- DEFAULT ----------
    return _wrap(
        f"{tool} completed (n = {s.get('n',0):,}).",
        "See result tables and graphs for detailed output.",
        f"p-value = {p}, α = {a}." if p is not None else "—",
        "—", "Review the output and visualizations for substantive insights.")


# ============================================================
# Graph-level interpretation
# ============================================================
GRAPH_INTERP = {
    "histogram":       "The histogram shows the frequency distribution. Look for skewness, multiple peaks, and outliers.",
    "boxplot":         "The box shows the interquartile range; the line is the median; whiskers extend to ~1.5×IQR. Points beyond are outliers.",
    "violin":          "Width represents density. Wider sections mean more observations at that value.",
    "kde":             "A smoothed density estimate. Useful for comparing shapes across groups.",
    "qq":              "Points falling on the diagonal indicate normality. Systematic deviation suggests non-normal distribution.",
    "scatter":         "Each point is one observation. Look for trends, clusters, and outliers.",
    "regression":      "The fitted line shows the average relationship. Vertical scatter reflects residual variability.",
    "residual":        "Residuals should hover randomly around zero. Patterns indicate model misspecification.",
    "bar":             "Bar heights reflect category magnitudes. Larger differences are visually obvious.",
    "stacked_bar":     "Stacked segments show the composition within each category.",
    "pie":             "Slices show proportional shares of the whole. Best for ≤6 categories.",
    "line":            "Lines connect time-ordered points. Look for trend, seasonality, and breaks.",
    "heatmap":         "Color intensity encodes the value. Identify hotspots and patterns at a glance.",
    "pair_plot":       "All pairwise scatterplots in a grid — useful for spotting cross-variable relationships.",
    "confusion_matrix":"Diagonal cells = correct predictions. Off-diagonal cells reveal which classes are confused.",
    "roc":             "Higher curve = better classifier. AUC=1 is perfect; AUC=0.5 is random.",
    "dendrogram":      "Vertical height = dissimilarity at merge. Cut horizontally to define clusters.",
}


def interpret_graph(graph_id: str) -> str:
    return GRAPH_INTERP.get(graph_id, "Visual representation of the analysis output.")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

"""
Multi-format report exporter.
Generates PDF (reportlab), Excel (openpyxl), HTML (string), PNG zip.
"""



# ------------------------------------------------------------
# PDF
# ------------------------------------------------------------
def export_pdf(analysis, result, interpretation, figures: list) -> bytes:
    """figures: list of (title, plotly Figure or {'_img': data-uri})"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                              topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=colors.HexColor("#7C5CFF"), fontSize=20, spaceAfter=12)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#4FD1C5"), fontSize=14, spaceAfter=8)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, textColor=colors.grey)

    story = []
    story.append(Paragraph(analysis.get("title","Statify Report"), h1))
    story.append(Paragraph(f"Tool: <b>{analysis.get('tool','')}</b> &nbsp;·&nbsp; Generated: {datetime.now():%Y-%m-%d %H:%M}", small))
    story.append(Spacer(1, 0.4*cm))

    # Headline
    story.append(Paragraph("Headline", h2))
    story.append(Paragraph(interpretation.get("headline",""), body))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Explanation", h2))
    story.append(Paragraph(interpretation.get("explanation",""), body))
    story.append(Spacer(1, 0.3*cm))

    if interpretation.get("statistical"):
        story.append(Paragraph("Statistical Meaning", h2))
        story.append(Paragraph(interpretation.get("statistical",""), body))
        story.append(Spacer(1, 0.3*cm))

    if interpretation.get("assumptions") and interpretation["assumptions"] != "—":
        story.append(Paragraph("Assumptions", h2))
        story.append(Paragraph(interpretation.get("assumptions",""), body))
        story.append(Spacer(1, 0.3*cm))

    if interpretation.get("conclusion"):
        story.append(Paragraph("Conclusion", h2))
        story.append(Paragraph(interpretation.get("conclusion",""), body))
        story.append(Spacer(1, 0.3*cm))

    # Tables
    for tname, tdf in result.get("tables", {}).items():
        if not isinstance(tdf, pd.DataFrame): continue
        story.append(Paragraph(tname, h2))
        data = [list(tdf.columns)] + tdf.astype(str).values.tolist()
        if len(data) > 30:
            data = data[:30] + [["…"]*len(data[0])]
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#7C5CFF")),
            ("TEXTCOLOR",(0,0),(-1,0), colors.white),
            ("FONTNAME",(0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1), 8),
            ("GRID",(0,0),(-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f5f5fa")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))

    # Figures
    for title, fig in figures:
        try:
            img_bytes = _fig_to_png_bytes(fig)
            if img_bytes is None: continue
            story.append(PageBreak())
            story.append(Paragraph(title, h2))
            story.append(Image(io.BytesIO(img_bytes), width=16*cm, height=10*cm))
        except Exception:
            continue

    doc.build(story)
    return buf.getvalue()


def _fig_to_png_bytes(fig) -> bytes | None:
    if isinstance(fig, dict) and "_img" in fig:
        import base64
        return base64.b64decode(fig["_img"].split(",")[1])
    try:
        return fig.to_image(format="png", width=1100, height=600, scale=2)
    except Exception:
        # kaleido might be missing — skip silently
        return None
# ------------------------------------------------------------
# Excel
# ------------------------------------------------------------
def export_excel(analysis, result, interpretation) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Summary sheet
        summary_rows = [
            ("Title", analysis.get("title","")),
            ("Tool", analysis.get("tool","")),
            ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
            ("Headline", interpretation.get("headline","")),
            ("Explanation", interpretation.get("explanation","")),
            ("Statistical", interpretation.get("statistical","")),
            ("Assumptions", interpretation.get("assumptions","")),
            ("Conclusion", interpretation.get("conclusion","")),
        ]
        pd.DataFrame(summary_rows, columns=["Field","Value"]).to_excel(writer, sheet_name="Summary", index=False)
        for tname, tdf in result.get("tables", {}).items():
            if isinstance(tdf, pd.DataFrame):
                safe = tname[:30].replace("/","_")
                tdf.to_excel(writer, sheet_name=safe, index=False)
    return buf.getvalue()


# ------------------------------------------------------------
# HTML
# ------------------------------------------------------------
def export_html(analysis, result, interpretation, figures: list) -> str:
    parts = ["""<!doctype html><html><head><meta charset="utf-8">
<title>Statify Report</title>
<style>
  body{font-family:Inter,system-ui,sans-serif;background:#0a0b10;color:#ECEEF6;margin:0;padding:40px;max-width:1100px;margin:auto;}
  h1{background:linear-gradient(135deg,#7C5CFF,#4FD1C5);-webkit-background-clip:text;color:transparent;font-size:2.2rem;}
  h2{color:#4FD1C5;margin-top:2rem;border-bottom:1px solid #232636;padding-bottom:6px;}
  table{width:100%;border-collapse:collapse;margin:1rem 0;background:#11131b;}
  th{background:#7C5CFF;color:#0a0b10;padding:8px;text-align:left;}
  td{border:1px solid #232636;padding:6px 8px;}
  tr:nth-child(even){background:#161924;}
  .meta{color:#9aa0b4;font-size:0.85rem;}
  .card{background:#11131b;border:1px solid #232636;border-radius:14px;padding:20px;margin:14px 0;}
</style></head><body>"""]
    parts.append(f"<h1>{analysis.get('title','Statify Report')}</h1>")
    parts.append(f"<div class='meta'>Tool: <b>{analysis.get('tool','')}</b> · Generated: {datetime.now():%Y-%m-%d %H:%M}</div>")

    parts.append("<div class='card'>")
    parts.append(f"<h2>Headline</h2><p>{interpretation.get('headline','')}</p>")
    parts.append(f"<h2>Explanation</h2><p>{interpretation.get('explanation','')}</p>")
    if interpretation.get("statistical"):
        parts.append(f"<h2>Statistical Meaning</h2><p>{interpretation.get('statistical')}</p>")
    if interpretation.get("assumptions") and interpretation['assumptions'] != "—":
        parts.append(f"<h2>Assumptions</h2><p>{interpretation.get('assumptions')}</p>")
    if interpretation.get("conclusion"):
        parts.append(f"<h2>Conclusion</h2><p>{interpretation.get('conclusion')}</p>")
    parts.append("</div>")

    for tname, tdf in result.get("tables", {}).items():
        if isinstance(tdf, pd.DataFrame):
            parts.append(f"<h2>{tname}</h2>")
            parts.append(tdf.to_html(index=False, border=0))

    for title, fig in figures:
        try:
            if isinstance(fig, dict) and "_img" in fig:
                parts.append(f"<h2>{title}</h2><img src='{fig['_img']}' style='max-width:100%;border-radius:12px;'>")
            else:
                parts.append(f"<h2>{title}</h2>")
                parts.append(fig.to_html(include_plotlyjs="cdn", full_html=False))
        except Exception:
            continue

    parts.append("</body></html>")
    return "".join(parts)


# ------------------------------------------------------------
# PNG ZIP
# ------------------------------------------------------------
def export_png_zip(figures: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, (title, fig) in enumerate(figures, 1):
            png = _fig_to_png_bytes(fig)
            if png is None: continue
            safe = "".join(c if c.isalnum() else "_" for c in title)[:40]
            zf.writestr(f"{i:02d}_{safe}.png", png)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

"""Page 1 — Upload"""


def page1_render():
    st.markdown(
        '<div class="hero-title">Upload your <span class="grad">dataset</span></div>'
        '<div class="hero-sub">CSV · XLSX · XLS · TSV · JSON · SPSS — Statify auto-detects format and types.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col_up, col_meta = st.columns([1.2, 1])

    with col_up:
        st.markdown('<div class="sec-title">📤 Drop your file</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose dataset",
            type=["csv","xlsx","xls","tsv","json","sav"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            df, status, msg = load_file(uploaded)
            if status == "success":
                st.session_state.df = df
                st.session_state.df_original = df.copy()
                st.session_state.filename = uploaded.name
                st.session_state.file_meta = summarize_dataframe(df)
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")

    with col_meta:
        if st.session_state.df is not None:
            meta = st.session_state.file_meta
            st.markdown('<div class="sec-title">📋 Dataset Snapshot</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.markdown(f'<div class="metric"><div class="label">Rows</div><div class="value">{meta["rows"]:,}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric"><div class="label">Columns</div><div class="value">{meta["cols"]}</div></div>', unsafe_allow_html=True)
            c3, c4 = st.columns(2)
            c3.markdown(f'<div class="metric"><div class="label">Missing</div><div class="value">{meta["missing"]:,}</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric"><div class="label">Memory KB</div><div class="value">{meta["memory_kb"]}</div></div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title">👁️ Preview · first 5 rows</div>', unsafe_allow_html=True)
        prev = st.session_state.df.head(5).copy()
        prev.index = range(1, len(prev)+1)
        st.dataframe(prev, use_container_width=True)

        # Type pills
        st.markdown('<div class="sec-title" style="margin-top:1.2rem;">🧩 Variable Types</div>', unsafe_allow_html=True)
        b = st.session_state.file_meta["buckets"]
        type_html = ""
        for kind, color in [("numeric","accent"),("categorical","good"),
                            ("datetime","warn"),("boolean","accent"),("text","")]:
            for col in b.get(kind, []):
                cls = f"pill pill-{color}" if color else "pill"
                type_html += f"<span class='{cls}' title='{kind}'>{col}</span>"
        st.markdown(type_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            if st.button("⚡ Statify It →", type="primary", use_container_width=True):
                goto(2)
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EDITOR
# ═══════════════════════════════════════════════════════════════════════════════



def _refresh_meta():
    st.session_state.file_meta = summarize_dataframe(st.session_state.df)


def page2_render():
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        return

    st.markdown(
        '<div class="hero-title">Edit & <span class="grad">clean</span> your data</div>'
        '<div class="hero-sub">Spreadsheet-like editing with type conversion, missing-value handling, and instant preview.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab_edit, tab_clean, tab_struct = st.tabs(["✏️ Cells", "🧼 Missing & Types", "🏗 Structure"])

    df = st.session_state.df

    # ============ CELLS ============
    with tab_edit:
        st.markdown('<div class="sec-title">Inline editor — changes saved instantly</div>', unsafe_allow_html=True)
        edited = st.data_editor(
            df,
            use_container_width=True,
            height=480,
            num_rows="dynamic",
            key="data_editor",
        )
        col_save, col_reset = st.columns([1,1])
        with col_save:
            changed = not edited.equals(df)
            if st.button("💾 Save Changes", type="primary", disabled=not changed, use_container_width=True):
                st.session_state.df = edited.reset_index(drop=True)
                _refresh_meta()
                st.success("Changes saved.")
                st.rerun()
        with col_reset:
            if st.button("↺ Reset to Original", use_container_width=True):
                st.session_state.df = st.session_state.df_original.copy()
                _refresh_meta()
                st.rerun()

    # ============ MISSING & TYPES ============
    with tab_clean:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="sec-title">Missing-value handling</div>', unsafe_allow_html=True)
            col = st.selectbox("Column", df.columns, key="miss_col")
            strategy = st.selectbox(
                "Strategy",
                ["drop","mean","median","mode","zero","ffill","bfill","custom"],
                key="miss_strategy",
            )
            custom_val = st.text_input("Custom value", key="miss_val") if strategy == "custom" else None
            n_missing = int(df[col].isna().sum()) if col in df.columns else 0
            st.caption(f"{n_missing} missing in `{col}`")
            if st.button("Apply", type="primary", key="apply_miss", use_container_width=True):
                st.session_state.df = handle_missing(df, col, strategy, custom_val)
                _refresh_meta()
                st.success(f"Applied {strategy} to {col}.")
                st.rerun()

        with c2:
            st.markdown('<div class="sec-title">Type conversion</div>', unsafe_allow_html=True)
            col2 = st.selectbox("Column ", df.columns, key="dtype_col")
            target = st.selectbox("Target type", ["int","float","str","bool","datetime","category"], key="dtype_t")
            st.caption(f"Current: `{df[col2].dtype}`")
            if st.button("Convert", type="primary", key="apply_dtype", use_container_width=True):
                st.session_state.df = convert_dtype(df, col2, target)
                _refresh_meta()
                st.success(f"`{col2}` → {target}")
                st.rerun()

    # ============ STRUCTURE ============
    with tab_struct:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="sec-title">Rename column</div>', unsafe_allow_html=True)
            old = st.selectbox("Column to rename", df.columns, key="ren_old")
            new = st.text_input("New name", key="ren_new")
            if st.button("Rename", type="primary", key="apply_ren", use_container_width=True):
                if new.strip() and new != old:
                    st.session_state.df = rename_column(df, old, new.strip())
                    _refresh_meta()
                    st.success(f"`{old}` → `{new}`")
                    st.rerun()

        with c2:
            st.markdown('<div class="sec-title">Delete column</div>', unsafe_allow_html=True)
            dc = st.selectbox("Column", df.columns, key="del_col")
            if st.button("🗑 Delete column", key="apply_dc", use_container_width=True):
                st.session_state.df = delete_column(df, dc)
                _refresh_meta()
                st.warning(f"Deleted `{dc}`")
                st.rerun()

        with c3:
            st.markdown('<div class="sec-title">Delete row</div>', unsafe_allow_html=True)
            ri = st.number_input("Row index (0-based)", min_value=0, max_value=max(len(df)-1,0), step=1, key="del_row")
            if st.button("🗑 Delete row", key="apply_dr", use_container_width=True):
                st.session_state.df = delete_row(df, int(ri))
                _refresh_meta()
                st.warning(f"Row {ri} removed")
                st.rerun()

    # Quick stats
    st.markdown("<br>", unsafe_allow_html=True)
    meta = st.session_state.file_meta
    cs = st.columns(4)
    cs[0].markdown(f'<div class="metric"><div class="label">Rows</div><div class="value">{meta["rows"]:,}</div></div>', unsafe_allow_html=True)
    cs[1].markdown(f'<div class="metric"><div class="label">Columns</div><div class="value">{meta["cols"]}</div></div>', unsafe_allow_html=True)
    cs[2].markdown(f'<div class="metric"><div class="label">Missing</div><div class="value">{meta["missing"]:,}</div></div>', unsafe_allow_html=True)
    cs[3].markdown(f'<div class="metric"><div class="label">Duplicates</div><div class="value">{meta["duplicates"]:,}</div></div>', unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

"""
Page 3 — Statistical Analysis Builder (REDESIGNED).

Fixes from previous version:
  • Dynamic role-based variable filtering (numeric/cat/datetime/bool)
  • Tool dropdown filtered by dataset compatibility
  • Per-role multi-select with cardinality validation
  • Parameter forms generated dynamically from registry
  • Up to 8 saved analyses with proper edit/delete
  • Validation messages, AI placeholder slots
"""


def _new_analysis_template():
    return {
        "id": uuid.uuid4().hex[:8],
        "title": "",
        "tool": None,
        "variables": {},
        "params": {},
    }




def _render_role_selector(role_name, role_def, buckets, current_value, key_prefix):
    allowed = []
    for t in role_def["types"]:
        allowed += buckets.get(t, [])
    label = role_def.get("label") or role_name.replace("_", " ").title()
    type_hint = "/".join(sorted(role_def["types"]))
    help_text = f"Allowed types: {type_hint}"

    if role_def.get("multi"):
        default = current_value if isinstance(current_value, list) else []
        return st.multiselect(label, options=allowed, default=[v for v in default if v in allowed],
                                help=help_text, key=f"{key_prefix}_{role_name}")
    else:
        idx = 0
        if current_value in allowed:
            idx = allowed.index(current_value) + 1
        opts = ["— select —"] + allowed
        choice = st.selectbox(label, options=opts, index=idx,
                                help=help_text, key=f"{key_prefix}_{role_name}")
        return None if choice == "— select —" else choice


def _render_param_form(spec, current_params, key_prefix):
    out = {}
    if not spec["params"]:
        st.caption("This tool has no extra parameters.")
        return out
    cols = st.columns(min(3, len(spec["params"])))
    for i, (pname, pdef) in enumerate(spec["params"].items()):
        with cols[i % len(cols)]:
            label = pdef.get("label") or pname.replace("_", " ").title()
            cur = current_params.get(pname, pdef.get("default"))
            ptype = pdef.get("type", "text")
            if ptype == "number":
                out[pname] = st.number_input(label, value=float(cur), key=f"{key_prefix}_p_{pname}", format="%.4f")
            elif ptype == "select":
                opts = pdef["options"]
                out[pname] = st.selectbox(label, options=opts, index=opts.index(cur) if cur in opts else 0, key=f"{key_prefix}_p_{pname}")
            elif ptype == "boolean":
                out[pname] = st.checkbox(label, value=bool(cur), key=f"{key_prefix}_p_{pname}")
            else:
                out[pname] = st.text_input(label, value=str(cur), key=f"{key_prefix}_p_{pname}")
    return out


def page3_render():
    if st.session_state.df is None:
        st.warning("Upload a dataset first.")
        return

    st.markdown(
        '<div class="hero-title">Build your <span class="grad">analysis</span></div>'
        '<div class="hero-sub">Choose a statistical tool, assign variables to roles, and set parameters. Save up to 8 analyses.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    df = st.session_state.df
    buckets = get_column_types(df)

    # AI placeholder
    with st.expander("🤖 AI Recommendations (Coming Soon)", expanded=False):
        st.info(
            "Statify's AI engine will soon recommend the best test based on your dataset, "
            "auto-select compatible variables, and explain trade-offs. "
            "Slot is reserved for `core.ai.recommendations` integration."
        )

    # Layout: builder on left, saved cards on right
    col_build, col_saved = st.columns([1.4, 1])

    # =================== BUILDER ===================
    with col_build:
        st.markdown('<div class="sec-title">🧪 New Analysis</div>', unsafe_allow_html=True)

        if "p3_draft" not in st.session_state:
            st.session_state.p3_draft = _new_analysis_template()
        draft = st.session_state.p3_draft

        # Title
        draft["title"] = st.text_input("Analysis Title", value=draft.get("title",""),
                                          placeholder="e.g. Income vs Spending Score",
                                          key=f"title_{draft['id']}")

        # Tool picker — categorised + compatibility-filtered
        compat = set(compatible_tools_for(buckets))
        cats = list_tools_by_category()
        cat_choice = st.selectbox("Category", ["All"] + sorted(cats.keys()), key=f"cat_{draft['id']}")
        if cat_choice == "All":
            tool_list = sorted(STAT_TOOLS.keys())
        else:
            tool_list = sorted(cats[cat_choice])
        # Annotate compatibility
        annotated = [(t, "✓" if t in compat else "✕") for t in tool_list]
        labels = [f"{ok}  {t}" for t, ok in annotated]
        cur_tool = draft.get("tool")
        cur_idx = 0
        if cur_tool:
            for i, t in enumerate(tool_list):
                if t == cur_tool: cur_idx = i; break
        chosen_label = st.selectbox("Statistical Tool", labels, index=cur_idx, key=f"tool_{draft['id']}")
        chosen_tool = tool_list[labels.index(chosen_label)]

        if chosen_tool != draft.get("tool"):
            draft["tool"] = chosen_tool
            draft["variables"] = {}
            draft["params"] = {pn: pd.get("default") for pn, pd in get_tool(chosen_tool)["params"].items()}

        spec = get_tool(chosen_tool)
        if chosen_tool not in compat:
            st.warning(f"⚠ This tool's variable requirements aren't fully met by your dataset. Missing types: " +
                        ", ".join(sorted({t for r in spec["roles"].values() for t in r["types"]})))
        st.markdown(f"<div class='card'><b>{chosen_tool}</b><div style='color:#9aa0b4;font-size:.85rem;margin-top:6px;'>{spec['desc']}</div></div>", unsafe_allow_html=True)

        # Roles
        st.markdown('<div class="sec-title" style="margin-top:1rem;">📐 Variables</div>', unsafe_allow_html=True)
        for rname, rdef in spec["roles"].items():
            cur = draft["variables"].get(rname)
            sel = _render_role_selector(rname, rdef, buckets, cur, key_prefix=f"v_{draft['id']}")
            draft["variables"][rname] = sel

        # Params
        st.markdown('<div class="sec-title" style="margin-top:1rem;">⚙️ Parameters</div>', unsafe_allow_html=True)
        draft["params"] = _render_param_form(spec, draft["params"], key_prefix=f"par_{draft['id']}")

        # Assumptions
        if spec["assumptions"]:
            st.markdown('<div class="sec-title" style="margin-top:1rem;">📋 Assumptions</div>', unsafe_allow_html=True)
            html = "".join(f"<span class='pill'>{a}</span>" for a in spec["assumptions"])
            st.markdown(html, unsafe_allow_html=True)

        # Recommended graphs
        if spec.get("graphs"):
            names = [GRAPH_CATALOG[g][0] for g in spec["graphs"] if g in GRAPH_CATALOG]
            st.markdown(
                '<div class="sec-title" style="margin-top:1rem;">📊 Recommended Graphs</div>'
                + "".join(f"<span class='pill pill-accent'>{n}</span>" for n in names),
                unsafe_allow_html=True,
            )

        # Validate + save
        ok, errs = _validate(draft, buckets)
        st.markdown("<br>", unsafe_allow_html=True)
        if errs:
            for e in errs:
                st.error(e)

        cap = len(st.session_state.analyses) >= APP_CONFIG["max_analyses"]
        c_save, c_clear = st.columns([2,1])
        with c_save:
            if st.button("💾 Save Analysis", type="primary", disabled=not ok or cap, use_container_width=True):
                st.session_state.analyses.append(dict(draft))
                st.session_state.p3_draft = _new_analysis_template()
                st.success("Analysis saved.")
                st.rerun()
        with c_clear:
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.p3_draft = _new_analysis_template()
                st.rerun()

        if cap:
            st.warning(f"Maximum of {APP_CONFIG['max_analyses']} analyses reached. Delete one to add more.")

    # =================== SAVED ===================
    with col_saved:
        st.markdown(f'<div class="sec-title">📂 Saved ({len(st.session_state.analyses)}/{APP_CONFIG["max_analyses"]})</div>', unsafe_allow_html=True)

        if not st.session_state.analyses:
            st.markdown("<div class='card' style='text-align:center;color:#9aa0b4;'>No analyses saved yet.</div>", unsafe_allow_html=True)
        else:
            for i, a in enumerate(st.session_state.analyses):
                vars_str = " · ".join(
                    f"{k}: {', '.join(v) if isinstance(v,list) else v}"
                    for k, v in a["variables"].items() if v
                )
                st.markdown(
                    f"""
                    <div class='analysis-card'>
                        <h4>{i+1}. {a.get('title','Untitled')}</h4>
                        <div class='meta'><b>Tool:</b> {a.get('tool')}</div>
                        <div class='meta'>{vars_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("✏️ Edit", key=f"edit_{a['id']}", use_container_width=True):
                        st.session_state.p3_draft = dict(a)
                        st.session_state.analyses.pop(i)
                        st.rerun()
                with cc2:
                    if st.button("🗑", key=f"del_{a['id']}", use_container_width=True):
                        st.session_state.analyses.pop(i)
                        st.rerun()
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

"""Page 4 — Analysis Selection Dashboard"""


def page4_render():
    if not st.session_state.analyses:
        st.warning("Save at least one analysis on Page 3 first.")
        return

    st.markdown(
        '<div class="hero-title">Run your <span class="grad">analyses</span></div>'
        '<div class="hero-sub">Pick graphs and launch the dashboard for any saved analysis.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, a in enumerate(st.session_state.analyses):
        with cols[idx % 2]:
            spec = get_tool(a["tool"])
            vars_str = " · ".join(
                f"<b>{k}</b>: {', '.join(v) if isinstance(v,list) else v}"
                for k, v in a["variables"].items() if v
            )
            params_str = " · ".join(f"{k}={v}" for k, v in a["params"].items()) or "default"

            st.markdown(
                f"""
                <div class='analysis-card'>
                    <h4>📌 {a.get('title','Untitled')}</h4>
                    <div class='meta'>🧪 <b>{a.get('tool')}</b></div>
                    <div class='meta'>{vars_str}</div>
                    <div class='meta'>⚙️ {params_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Graph picker
            graph_opts = list_graphs_for_tool(spec)
            graph_id_map = {label: gid for gid, label in graph_opts}
            cur = st.session_state.graph_selections.get(a["id"], [label for _, label in graph_opts])
            chosen_labels = st.multiselect(
                "Graphs to include",
                options=list(graph_id_map.keys()),
                default=[c for c in cur if c in graph_id_map],
                key=f"graphs_{a['id']}",
            )
            st.session_state.graph_selections[a["id"]] = chosen_labels

            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("▶ Run Analysis", key=f"run_{a['id']}", type="primary", use_container_width=True):
                    st.session_state.active_analysis = a["id"]
                    goto(5)
            with cc2:
                if st.button("📊 Graphs Only", key=f"graphs_only_{a['id']}", use_container_width=True):
                    st.session_state.active_analysis = a["id"]
                    goto(5)
            st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — REPORT
# ═══════════════════════════════════════════════════════════════════════════════

"""Page 5 — Professional Dashboard & Report"""


def _find_analysis(aid):
    for a in st.session_state.analyses:
        if a["id"] == aid:
            return a
    return None


def page5_render():
    aid = st.session_state.active_analysis
    if aid is None:
        st.warning("Select an analysis from Page 4 first.")
        return
    analysis = _find_analysis(aid)
    if analysis is None:
        st.error("Analysis not found.")
        return

    # Compute (cached)
    if aid not in st.session_state.analysis_results:
        with st.spinner("Computing..."):
            st.session_state.analysis_results[aid] = run_analysis(
                analysis["tool"], st.session_state.df,
                analysis["variables"], analysis["params"],
            )
    result = st.session_state.analysis_results[aid]
    interp = interpret_result(analysis["tool"], result)

    # ---------- TOP NAV ----------
    top1, top2, top3 = st.columns([1, 4, 2])
    with top1:
        if st.button("← Back", key="report_back", use_container_width=True):
            goto(4)
    with top2:
        st.markdown(
            f"<div style='font-weight:700;font-size:1.4rem;'>📑 {analysis.get('title','Report')}</div>"
            f"<div class='meta' style='color:#9aa0b4;font-size:.85rem;'>{analysis['tool']}  ·  generated {datetime.now():%Y-%m-%d %H:%M}</div>",
            unsafe_allow_html=True,
        )
    with top3:
        if st.button("🔄 Recompute", use_container_width=True):
            st.session_state.analysis_results.pop(aid, None)
            st.rerun()

    st.markdown("<hr class='div'>", unsafe_allow_html=True)

    if not result.get("ok", True):
        st.error(f"❌ {result.get('error','Analysis failed')}")
        return

    # ---------- HEADLINE CARD ----------
    st.markdown(
        f"""
        <div class='glass'>
            <div class='sec-title'>🎯 Headline</div>
            <div style='font-size:1.15rem;font-weight:600;color:#ECEEF6;'>{interp['headline']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- KEY METRICS ----------
    s = result.get("summary", {})
    if s:
        keys = list(s.keys())[:4]
        cs = st.columns(max(len(keys), 1))
        for i, k in enumerate(keys):
            v = s[k]
            try:
                v_str = f"{float(v):.4f}" if isinstance(v, (int, float)) else str(v)
            except Exception:
                v_str = str(v)
            cs[i].markdown(
                f"<div class='metric'><div class='label'>{k.replace('_',' ')}</div><div class='value'>{v_str}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)

    # ---------- TABS ----------
    tab_overview, tab_tables, tab_graphs, tab_interpret, tab_export = st.tabs(
        ["🎯 Overview", "📋 Tables", "📊 Graphs", "🧠 Interpretation", "📤 Export"]
    )

    # === Overview ===
    with tab_overview:
        st.markdown(f"<div class='card'><b>Explanation</b><br>{interp['explanation']}</div>", unsafe_allow_html=True)
        if interp.get("statistical"):
            st.markdown(f"<div class='card' style='margin-top:.8rem;'><b>Statistical Meaning</b><br>{interp['statistical']}</div>", unsafe_allow_html=True)
        if interp.get("conclusion"):
            st.markdown(f"<div class='card' style='margin-top:.8rem;'><b>Conclusion</b><br>{interp['conclusion']}</div>", unsafe_allow_html=True)
        if interp.get("assumptions") and interp["assumptions"] != "—":
            st.markdown(f"<div class='card' style='margin-top:.8rem;'><b>Assumptions</b><br>{interp['assumptions']}</div>", unsafe_allow_html=True)

        # Significance pill
        p = result.get("p_value"); a = result.get("alpha", 0.05)
        if p is not None:
            sig = p < a
            cls = "pill-good" if sig else "pill-bad"
            txt = "Significant" if sig else "Not Significant"
            st.markdown(f"<div style='margin-top:1rem;'><span class='pill {cls}'>{txt} · p = {p:.4f} · α = {a}</span></div>", unsafe_allow_html=True)

    # === Tables ===
    with tab_tables:
        for tname, tdf in result.get("tables", {}).items():
            st.markdown(f"<div class='sec-title'>{tname}</div>", unsafe_allow_html=True)
            st.dataframe(tdf, use_container_width=True)

    # === Graphs ===
    figs_for_export = []
    with tab_graphs:
        spec = get_tool(analysis["tool"])
        graph_opts = list_graphs_for_tool(spec)
        chosen_labels = st.session_state.graph_selections.get(aid)
        if chosen_labels is None:
            chosen_labels = [label for _, label in graph_opts]
        chosen_ids = [gid for gid, label in graph_opts if label in chosen_labels]

        if not chosen_ids:
            st.info("No graphs selected. Go to Page 4 to choose visualizations.")
        for gid in chosen_ids:
            label = GRAPH_CATALOG[gid][0]
            try:
                fig = build_graph(gid, result)
            except Exception as e:
                st.warning(f"Could not render {label}: {e}")
                continue
            if fig is None:
                continue
            # Could be list (e.g. histograms per variable)
            figs = fig if isinstance(fig, list) else [fig]
            for j, f in enumerate(figs):
                title = f"{label}" + (f" #{j+1}" if len(figs) > 1 else "")
                if isinstance(f, dict) and "_img" in f:
                    st.markdown(f"<div class='sec-title'>{title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<img src='{f['_img']}' style='width:100%;border-radius:12px;'>", unsafe_allow_html=True)
                else:
                    st.plotly_chart(f, use_container_width=True)
                with st.expander(f"💡 What does this {label} mean?"):
                    st.write(interpret_graph(gid))
                figs_for_export.append((title, f))

    # === Interpretation tab ===
    with tab_interpret:
        st.markdown(f"### {interp['headline']}")
        st.write(f"**Plain-language summary**\n\n{interp['explanation']}")
        if interp.get("statistical"):
            st.write(f"**Statistical reasoning**\n\n{interp['statistical']}")
        if interp.get("assumptions") and interp["assumptions"] != "—":
            st.write(f"**Assumptions**\n\n{interp['assumptions']}")
        if interp.get("conclusion"):
            st.write(f"**Bottom line**\n\n{interp['conclusion']}")

    # === Export ===
    with tab_export:
        st.markdown('<div class="sec-title">Download your report</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        safe_title = "".join(c if c.isalnum() else "_" for c in analysis.get("title","report"))[:40] or "report"

        with c1:
            try:
                pdf_bytes = export_pdf(analysis, result, interp, figs_for_export)
                st.download_button("📕 PDF", pdf_bytes, file_name=f"{safe_title}.pdf",
                                     mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error(f"PDF: {e}")

        with c2:
            try:
                xls = export_excel(analysis, result, interp)
                st.download_button("📗 Excel", xls, file_name=f"{safe_title}.xlsx",
                                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                     use_container_width=True)
            except Exception as e:
                st.error(f"Excel: {e}")

        with c3:
            try:
                html = export_html(analysis, result, interp, figs_for_export)
                st.download_button("🌐 HTML", html, file_name=f"{safe_title}.html",
                                     mime="text/html", use_container_width=True)
            except Exception as e:
                st.error(f"HTML: {e}")

        with c4:
            try:
                if figs_for_export:
                    z = export_png_zip(figs_for_export)
                    st.download_button("🖼 PNG ZIP", z, file_name=f"{safe_title}_charts.zip",
                                         mime="application/zip", use_container_width=True)
                else:
                    st.button("🖼 PNG ZIP", disabled=True, use_container_width=True)
            except Exception as e:
                st.error(f"PNG: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════

def goto(page: int):
    st.session_state.page = max(1, min(5, page))
    st.rerun()


def render_progress(current: int):
    parts = ['<div class="progress-wrap">']
    for i in range(1, 6):
        cls = "progress-step"
        if i == current:  cls += " active"
        elif i < current: cls += " done"
        meta = PAGES[i]
        parts.append(f'<div class="{cls}">{meta["icon"]} &nbsp; {i}. {meta["title"]}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_footer_nav():
    page         = st.session_state.page
    df_loaded    = st.session_state.df is not None
    has_analysis = len(st.session_state.analyses) > 0
    next_enabled = {
        1: df_loaded, 2: df_loaded, 3: has_analysis,
        4: st.session_state.active_analysis is not None, 5: False,
    }[page]
    st.markdown('<hr class="div">', unsafe_allow_html=True)
    cols = st.columns([1, 4, 1])
    with cols[0]:
        if page > 1:
            if st.button("← Back", key=f"back_{page}", use_container_width=True):
                goto(page - 1)
    with cols[2]:
        if page < 5:
            if st.button("Next →", key=f"next_{page}", type="primary",
                         disabled=not next_enabled, use_container_width=True):
                goto(page + 1)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            f'<div class="brand">{APP_CONFIG["name"]}</div>'
            f'<div class="brand-sub">v{APP_CONFIG["version"]} · {APP_CONFIG["tagline"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<hr class='div' style='margin:1rem 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Pages</div>', unsafe_allow_html=True)
        for i in range(1, 6):
            label    = f"{PAGES[i]['icon']}  {i}. {PAGES[i]['title']}"
            disabled = (i > 1 and st.session_state.df is None) or \
                       (i >= 4 and not st.session_state.analyses)
            btn_type = "primary" if st.session_state.page == i else "secondary"
            if st.button(label, key=f"nav_{i}", disabled=disabled,
                         use_container_width=True, type=btn_type):
                goto(i)
        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        if st.session_state.df is not None:
            df = st.session_state.df
            st.markdown('<div class="sec-title">Dataset</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="card" style="padding:0.8rem;">'
                f'<div style="font-weight:600;font-size:0.9rem;">{st.session_state.filename or "data"}</div>'
                f'<div style="color:#9aa0b4;font-size:0.78rem;margin-top:4px;">'
                f'{df.shape[0]:,} rows × {df.shape[1]} cols</div></div>',
                unsafe_allow_html=True,
            )
        if st.session_state.analyses:
            st.markdown('<div class="sec-title" style="margin-top:1rem;">Analyses</div>', unsafe_allow_html=True)
            for a in st.session_state.analyses:
                st.markdown(f"<div class='pill pill-accent'>{a.get('title','Untitled')}</div>", unsafe_allow_html=True)
        st.markdown("<hr class='div'>", unsafe_allow_html=True)
        if st.button("🔄 Reset Session", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def init_session_state():
    defaults = {
        "page": 1, "df": None, "df_original": None,
        "filename": "", "file_meta": {}, "edit_history": [],
        "analyses": [], "active_analysis": None,
        "analysis_results": {}, "graph_selections": {},
        "ai_suggestions": [], "theme": "dark",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.set_page_config(
        page_title=APP_CONFIG["name"],
        page_icon=APP_CONFIG["icon"],
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    inject_global_css()
    render_sidebar()
    render_progress(st.session_state.page)

    {
        1: page1_render,
        2: page2_render,
        3: page3_render,
        4: page4_render,
        5: page5_render,
    }[st.session_state.page]()

    render_footer_nav()


if __name__ == "__main__":
    main()



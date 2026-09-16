
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

"""Pure-function data cleaning operations (return new DataFrame)."""


def handle_missing(df: pd.DataFrame, column: str, strategy: str, fill_value=None) -> pd.DataFrame:
    """strategy ∈ {'drop', 'mean', 'median', 'mode', 'zero', 'custom', 'ffill', 'bfill'}"""
    out = df.copy()
    if column not in out.columns:
        return out
    if strategy == "drop":
        out = out.dropna(subset=[column]).reset_index(drop=True)
    elif strategy == "mean" and pd.api.types.is_numeric_dtype(out[column]):
        out[column] = out[column].fillna(out[column].mean())
    elif strategy == "median" and pd.api.types.is_numeric_dtype(out[column]):
        out[column] = out[column].fillna(out[column].median())
    elif strategy == "mode":
        m = out[column].mode(dropna=True)
        if len(m): out[column] = out[column].fillna(m.iloc[0])
    elif strategy == "zero":
        out[column] = out[column].fillna(0)
    elif strategy == "custom":
        out[column] = out[column].fillna(fill_value)
    elif strategy == "ffill":
        out[column] = out[column].ffill()
    elif strategy == "bfill":
        out[column] = out[column].bfill()
    return out


def convert_dtype(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    """target ∈ {'int','float','str','bool','datetime','category'}"""
    out = df.copy()
    if column not in out.columns:
        return out
    try:
        if target == "int":      out[column] = pd.to_numeric(out[column], errors="coerce").astype("Int64")
        elif target == "float":  out[column] = pd.to_numeric(out[column], errors="coerce")
        elif target == "str":    out[column] = out[column].astype(str)
        elif target == "bool":   out[column] = out[column].astype(bool)
        elif target == "datetime": out[column] = pd.to_datetime(out[column], errors="coerce")
        elif target == "category": out[column] = out[column].astype("category")
    except Exception:
        pass
    return out


def rename_column(df, old, new):
    if old in df.columns and new and old != new:
        return df.rename(columns={old: new})
    return df


def delete_row(df, idx):
    if 0 <= idx < len(df):
        return df.drop(df.index[idx]).reset_index(drop=True)
    return df


def delete_column(df, col):
    if col in df.columns:
        return df.drop(columns=[col])
    return df


def update_cell(df, row_idx, col, value):
    out = df.copy()
    if 0 <= row_idx < len(out) and col in out.columns:
        try:
            dtype = out[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                value = int(float(value)) if str(value).strip() != "" else pd.NA
            elif pd.api.types.is_float_dtype(dtype):
                value = float(value) if str(value).strip() != "" else np.nan
            elif pd.api.types.is_bool_dtype(dtype):
                value = str(value).lower() in ("true", "1", "yes", "y")
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                value = pd.to_datetime(value, errors="coerce")
            out.at[out.index[row_idx], col] = value
        except Exception:
            out.at[out.index[row_idx], col] = value
    return out


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

NUM   = {"numeric"}
CAT   = {"categorical", "boolean"}
TIME  = {"datetime"}


def role(types, multi=False, min_n=1, max_n=None, optional=False, label=None):
    return {
        "types": set(types),
        "multi": multi,
        "min": min_n,
        "max": max_n,
        "optional": optional,
        "label": label,
    }


STAT_TOOLS: dict = {

    # ===== DESCRIPTIVE =====
    "Descriptive Statistics": {
        "category": "Descriptive",
        "desc": "Mean, median, std, quartiles, skewness, kurtosis.",
        "roles": {"variables": role(NUM, multi=True, min_n=1, max_n=20, label="Numeric Variables")},
        "params": {},
        "assumptions": [],
        "graphs": ["histogram", "boxplot", "violin", "kde"],
    },
    "Frequency Table": {
        "category": "Descriptive",
        "desc": "Counts and percentages for categorical variables.",
        "roles": {"variable": role(CAT, multi=False, label="Categorical Variable")},
        "params": {},
        "assumptions": [],
        "graphs": ["bar", "pie"],
    },
    "Cross Tabulation": {
        "category": "Descriptive",
        "desc": "Contingency table between two categorical variables.",
        "roles": {
            "row": role(CAT, label="Row Variable"),
            "col": role(CAT, label="Column Variable"),
        },
        "params": {"normalize": {"type": "select", "options": ["none","row","col","all"], "default": "none"}},
        "assumptions": [],
        "graphs": ["heatmap", "stacked_bar"],
    },
    "Skewness & Kurtosis": {
        "category": "Descriptive",
        "desc": "Distribution shape diagnostics.",
        "roles": {"variables": role(NUM, multi=True, label="Numeric Variables")},
        "params": {},
        "assumptions": [],
        "graphs": ["histogram", "kde", "qq"],
    },
    "Outlier Detection (IQR)": {
        "category": "Descriptive",
        "desc": "Identifies outliers via 1.5 × IQR rule.",
        "roles": {"variable": role(NUM, label="Numeric Variable")},
        "params": {"k": {"type": "number", "default": 1.5}},
        "assumptions": [],
        "graphs": ["boxplot", "scatter"],
    },

    # ===== HYPOTHESIS TESTING =====
    "One-Sample t-Test": {
        "category": "Hypothesis Testing",
        "desc": "Compares sample mean against a hypothesized value.",
        "roles": {"variable": role(NUM, label="Numeric Variable")},
        "params": {
            "mu0":   {"type": "number", "default": 0.0, "label": "Hypothesized Mean (μ₀)"},
            "alpha": {"type": "number", "default": 0.05},
            "tail":  {"type": "select", "options": ["two-sided","greater","less"], "default": "two-sided"},
        },
        "assumptions": ["Approximately normal distribution","Independent observations"],
        "graphs": ["histogram","qq","boxplot"],
    },
    "Independent t-Test": {
        "category": "Hypothesis Testing",
        "desc": "Compares means of two independent groups.",
        "roles": {
            "value": role(NUM, label="Numeric Outcome"),
            "group": role(CAT, label="Grouping Variable (2 levels)"),
        },
        "params": {
            "alpha":     {"type": "number", "default": 0.05},
            "equal_var": {"type": "boolean", "default": False, "label": "Assume Equal Variance"},
            "tail":      {"type": "select", "options": ["two-sided","greater","less"], "default": "two-sided"},
        },
        "assumptions": ["Normality within groups","Independence","Equal variance (if selected)"],
        "graphs": ["boxplot","violin","histogram"],
    },
    "Paired t-Test": {
        "category": "Hypothesis Testing",
        "desc": "Compares two paired numeric measures.",
        "roles": {
            "var1": role(NUM, label="Measurement 1"),
            "var2": role(NUM, label="Measurement 2"),
        },
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Normality of differences","Paired observations"],
        "graphs": ["boxplot","scatter","histogram"],
    },
    "Z-Test (One Sample)": {
        "category": "Hypothesis Testing",
        "desc": "Compares mean to known value when population σ is known.",
        "roles": {"variable": role(NUM)},
        "params": {
            "mu0":   {"type": "number", "default": 0.0},
            "sigma": {"type": "number", "default": 1.0, "label": "Population σ"},
            "alpha": {"type": "number", "default": 0.05},
        },
        "assumptions": ["Population variance known","Large sample or normal data"],
        "graphs": ["histogram","qq"],
    },
    "Proportion Z-Test": {
        "category": "Hypothesis Testing",
        "desc": "Compares observed proportion to expected.",
        "roles": {"variable": role(CAT, label="Binary Variable")},
        "params": {
            "p0":    {"type": "number", "default": 0.5, "label": "Hypothesized Proportion"},
            "alpha": {"type": "number", "default": 0.05},
        },
        "assumptions": ["Binary outcome","Independent trials"],
        "graphs": ["bar","pie"],
    },
    "Chi-Square Test of Independence": {
        "category": "Hypothesis Testing",
        "desc": "Tests independence between two categorical variables.",
        "roles": {
            "var1": role(CAT, label="Variable 1"),
            "var2": role(CAT, label="Variable 2"),
        },
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Independence","Expected counts ≥ 5 in most cells"],
        "graphs": ["heatmap","stacked_bar"],
    },
    "Chi-Square Goodness of Fit": {
        "category": "Hypothesis Testing",
        "desc": "Compares observed counts to expected distribution.",
        "roles": {"variable": role(CAT)},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Independent observations","Expected counts ≥ 5"],
        "graphs": ["bar"],
    },
    "Fisher's Exact Test": {
        "category": "Hypothesis Testing",
        "desc": "Exact test for 2×2 contingency tables.",
        "roles": {
            "var1": role(CAT, label="Binary Variable 1"),
            "var2": role(CAT, label="Binary Variable 2"),
        },
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Binary categorical variables"],
        "graphs": ["heatmap","stacked_bar"],
    },

    # ===== CORRELATION =====
    "Pearson Correlation": {
        "category": "Correlation",
        "desc": "Linear correlation between two numeric variables.",
        "roles": {"x": role(NUM, label="Variable X"), "y": role(NUM, label="Variable Y")},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Linearity","Bivariate normality"],
        "graphs": ["scatter","regression","heatmap"],
    },
    "Spearman Correlation": {
        "category": "Correlation",
        "desc": "Rank-based monotonic correlation.",
        "roles": {"x": role(NUM, label="Variable X"), "y": role(NUM, label="Variable Y")},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Monotonic relationship"],
        "graphs": ["scatter","regression"],
    },
    "Kendall's Tau": {
        "category": "Correlation",
        "desc": "Rank correlation for small samples.",
        "roles": {"x": role(NUM), "y": role(NUM)},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": [],
        "graphs": ["scatter"],
    },
    "Correlation Matrix": {
        "category": "Correlation",
        "desc": "Pairwise correlations across multiple numeric variables.",
        "roles": {"variables": role(NUM, multi=True, min_n=2, max_n=20)},
        "params": {"method": {"type": "select", "options": ["pearson","spearman","kendall"], "default":"pearson"}},
        "assumptions": [],
        "graphs": ["heatmap","pair_plot"],
    },
    "Point-Biserial Correlation": {
        "category": "Correlation",
        "desc": "Correlation between binary and continuous variable.",
        "roles": {"binary": role(CAT, label="Binary Variable"), "numeric": role(NUM, label="Numeric Variable")},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": [],
        "graphs": ["boxplot","violin"],
    },

    # ===== REGRESSION =====
    "Simple Linear Regression": {
        "category": "Regression",
        "desc": "Predicts numeric Y from one numeric X.",
        "roles": {
            "y": role(NUM, label="Dependent (Y)"),
            "x": role(NUM, label="Independent (X)"),
        },
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Linearity","Normality of residuals","Homoscedasticity","Independence"],
        "graphs": ["regression","residual","scatter"],
    },
    "Multiple Linear Regression": {
        "category": "Regression",
        "desc": "Predicts numeric Y from multiple numeric Xs.",
        "roles": {
            "y": role(NUM, label="Dependent (Y)"),
            "x": role(NUM, multi=True, min_n=2, max_n=15, label="Independent Variables"),
        },
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Linearity","No perfect multicollinearity","Normality of residuals"],
        "graphs": ["regression","residual","pair_plot"],
    },
    "Polynomial Regression": {
        "category": "Regression",
        "desc": "Fits Y as polynomial of X.",
        "roles": {"y": role(NUM), "x": role(NUM)},
        "params": {"degree": {"type": "number", "default": 2}},
        "assumptions": ["Polynomial relationship"],
        "graphs": ["regression","residual"],
    },
    "Logistic Regression": {
        "category": "Regression",
        "desc": "Predicts binary outcome from numeric/categorical predictors.",
        "roles": {
            "y": role(CAT, label="Binary Outcome (Y)"),
            "x": role(NUM, multi=True, min_n=1, max_n=15, label="Predictors (Numeric)"),
        },
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Binary outcome","Independence","No multicollinearity"],
        "graphs": ["roc","confusion_matrix","scatter"],
    },
    "Ridge Regression": {
        "category": "Regression",
        "desc": "Linear regression with L2 regularization.",
        "roles": {"y": role(NUM), "x": role(NUM, multi=True, min_n=1, max_n=20)},
        "params": {"alpha": {"type": "number", "default": 1.0}},
        "assumptions": [],
        "graphs": ["regression","residual"],
    },
    "Lasso Regression": {
        "category": "Regression",
        "desc": "Linear regression with L1 regularization for feature selection.",
        "roles": {"y": role(NUM), "x": role(NUM, multi=True, min_n=1, max_n=20)},
        "params": {"alpha": {"type": "number", "default": 0.1}},
        "assumptions": [],
        "graphs": ["regression","residual","bar"],
    },

    # ===== ANOVA =====
    "One-Way ANOVA": {
        "category": "ANOVA",
        "desc": "Compares means across 3+ groups.",
        "roles": {
            "value": role(NUM, label="Numeric Outcome"),
            "group": role(CAT, label="Grouping Variable (3+ levels)"),
        },
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Normality within groups","Equal variance","Independence"],
        "graphs": ["boxplot","violin","bar"],
    },
    "Two-Way ANOVA": {
        "category": "ANOVA",
        "desc": "Two categorical factors influencing numeric outcome.",
        "roles": {
            "value":   role(NUM, label="Numeric Outcome"),
            "factor1": role(CAT, label="Factor 1"),
            "factor2": role(CAT, label="Factor 2"),
        },
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Normality","Equal variance"],
        "graphs": ["boxplot","violin","heatmap"],
    },
    "Repeated Measures ANOVA": {
        "category": "ANOVA",
        "desc": "Within-subject ANOVA across timepoints/conditions.",
        "roles": {"variables": role(NUM, multi=True, min_n=3, max_n=10, label="Repeated Measurements")},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Sphericity","Normality"],
        "graphs": ["line","boxplot"],
    },
    "ANCOVA": {
        "category": "ANOVA",
        "desc": "ANOVA controlling for a continuous covariate.",
        "roles": {
            "value":     role(NUM, label="Outcome"),
            "group":     role(CAT, label="Group"),
            "covariate": role(NUM, label="Covariate"),
        },
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Linearity of covariate","Homogeneity of regression slopes"],
        "graphs": ["regression","boxplot"],
    },
    "Tukey HSD Post-Hoc": {
        "category": "ANOVA",
        "desc": "Pairwise comparisons after ANOVA.",
        "roles": {"value": role(NUM), "group": role(CAT)},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": [],
        "graphs": ["boxplot"],
    },

    # ===== NON-PARAMETRIC =====
    "Mann-Whitney U Test": {
        "category": "Non-Parametric",
        "desc": "Non-parametric alternative to independent t-test.",
        "roles": {
            "value": role(NUM, label="Numeric Outcome"),
            "group": role(CAT, label="Group (2 levels)"),
        },
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Independent observations"],
        "graphs": ["boxplot","violin"],
    },
    "Wilcoxon Signed-Rank Test": {
        "category": "Non-Parametric",
        "desc": "Non-parametric paired test.",
        "roles": {"var1": role(NUM), "var2": role(NUM)},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Paired observations","Symmetric differences"],
        "graphs": ["boxplot","histogram"],
    },
    "Kruskal-Wallis Test": {
        "category": "Non-Parametric",
        "desc": "Non-parametric alternative to one-way ANOVA.",
        "roles": {
            "value": role(NUM, label="Numeric Outcome"),
            "group": role(CAT, label="Group (3+ levels)"),
        },
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": ["Independent observations"],
        "graphs": ["boxplot","violin"],
    },
    "Friedman Test": {
        "category": "Non-Parametric",
        "desc": "Non-parametric repeated-measures test.",
        "roles": {"variables": role(NUM, multi=True, min_n=3, max_n=10)},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": [],
        "graphs": ["line","boxplot"],
    },
    "Shapiro-Wilk Normality": {
        "category": "Non-Parametric",
        "desc": "Tests if data is normally distributed.",
        "roles": {"variable": role(NUM)},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": [],
        "graphs": ["histogram","qq","kde"],
    },
    "Levene's Test": {
        "category": "Non-Parametric",
        "desc": "Tests equality of variances across groups.",
        "roles": {"value": role(NUM), "group": role(CAT)},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": [],
        "graphs": ["boxplot"],
    },
    "Kolmogorov-Smirnov Test": {
        "category": "Non-Parametric",
        "desc": "Compares sample distribution to a reference.",
        "roles": {"variable": role(NUM)},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": [],
        "graphs": ["histogram","kde"],
    },
# ===== TIME SERIES =====
    "Time Series Decomposition": {
        "category": "Time Series",
        "desc": "Trend / seasonal / residual decomposition.",
        "roles": {"time": role(TIME, label="Date/Time"), "value": role(NUM, label="Series Value")},
        "params": {"period": {"type": "number", "default": 12}},
        "assumptions": [],
        "graphs": ["line"],
    },
    "Stationarity (ADF Test)": {
        "category": "Time Series",
        "desc": "Augmented Dickey-Fuller stationarity test.",
        "roles": {"value": role(NUM, label="Series")},
        "params": {"alpha": {"type": "number", "default": 0.05}},
        "assumptions": [],
        "graphs": ["line"],
    },
    "ARIMA Forecast": {
        "category": "Forecasting",
        "desc": "ARIMA(p,d,q) forecast.",
        "roles": {"value": role(NUM, label="Series")},
        "params": {
            "p": {"type": "number", "default": 1},
            "d": {"type": "number", "default": 1},
            "q": {"type": "number", "default": 1},
            "horizon": {"type": "number", "default": 10},
        },
        "assumptions": ["Stationarity (after differencing)"],
        "graphs": ["line"],
    },
    "Exponential Smoothing": {
        "category": "Forecasting",
        "desc": "Holt-Winters forecasting.",
        "roles": {"value": role(NUM)},
        "params": {"horizon": {"type": "number", "default": 10}},
        "assumptions": [],
        "graphs": ["line"],
    },
    "Autocorrelation (ACF/PACF)": {
        "category": "Time Series",
        "desc": "Auto- and partial-autocorrelation.",
        "roles": {"value": role(NUM)},
        "params": {"lags": {"type": "number", "default": 20}},
        "assumptions": [],
        "graphs": ["bar"],
    },

    # ===== CLUSTERING =====
    "K-Means Clustering": {
        "category": "Clustering",
        "desc": "Unsupervised clustering with k centroids.",
        "roles": {"variables": role(NUM, multi=True, min_n=2, max_n=20)},
        "params": {"k": {"type": "number", "default": 3}},
        "assumptions": ["Roughly spherical clusters"],
        "graphs": ["scatter","pair_plot"],
    },
    "Hierarchical Clustering": {
        "category": "Clustering",
        "desc": "Agglomerative hierarchical clustering.",
        "roles": {"variables": role(NUM, multi=True, min_n=2, max_n=20)},
        "params": {
            "method":   {"type": "select", "options": ["ward","complete","average","single"], "default": "ward"},
            "clusters": {"type": "number", "default": 3},
        },
        "assumptions": [],
        "graphs": ["dendrogram","heatmap"],
    },
    "DBSCAN": {
        "category": "Clustering",
        "desc": "Density-based clustering, finds arbitrary shapes.",
        "roles": {"variables": role(NUM, multi=True, min_n=2)},
        "params": {"eps": {"type": "number", "default": 0.5}, "min_samples": {"type": "number", "default": 5}},
        "assumptions": [],
        "graphs": ["scatter"],
    },

    # ===== DIMENSIONALITY =====
    "PCA": {
        "category": "Dimensionality",
        "desc": "Principal Component Analysis.",
        "roles": {"variables": role(NUM, multi=True, min_n=2, max_n=50)},
        "params": {"n_components": {"type": "number", "default": 2}},
        "assumptions": ["Linear relationships"],
        "graphs": ["scatter","bar"],
    },
    "Factor Analysis": {
        "category": "Dimensionality",
        "desc": "Latent factor extraction.",
        "roles": {"variables": role(NUM, multi=True, min_n=3, max_n=50)},
        "params": {"n_factors": {"type": "number", "default": 2}},
        "assumptions": [],
        "graphs": ["heatmap","bar"],
    },

    # ===== SURVIVAL =====
    "Kaplan-Meier Survival": {
        "category": "Survival",
        "desc": "Survival probability over time.",
        "roles": {
            "duration": role(NUM, label="Duration"),
            "event":    role(CAT, label="Event Indicator (0/1)"),
        },
        "params": {},
        "assumptions": ["Right-censored data"],
        "graphs": ["line"],
    },
    "Cox Proportional Hazards": {
        "category": "Survival",
        "desc": "Cox regression for time-to-event with covariates.",
        "roles": {
            "duration":   role(NUM, label="Duration"),
            "event":      role(CAT, label="Event"),
            "covariates": role(NUM, multi=True, min_n=1, max_n=10),
        },
        "params": {},
        "assumptions": ["Proportional hazards"],
        "graphs": ["line","bar"],
    },

    # ===== ML BASICS =====
    "Decision Tree Classifier": {
        "category": "Machine Learning",
        "desc": "Tree-based classification.",
        "roles": {
            "y": role(CAT, label="Target Class"),
            "x": role(NUM, multi=True, min_n=1, max_n=20),
        },
        "params": {"max_depth": {"type": "number", "default": 5}},
        "assumptions": [],
        "graphs": ["confusion_matrix","bar"],
    },
    "Random Forest Classifier": {
        "category": "Machine Learning",
        "desc": "Ensemble of decision trees.",
        "roles": {
            "y": role(CAT, label="Target Class"),
            "x": role(NUM, multi=True, min_n=1, max_n=30),
        },
        "params": {"n_estimators": {"type": "number", "default": 100}, "max_depth": {"type": "number", "default": 10}},
        "assumptions": [],
        "graphs": ["confusion_matrix","bar","roc"],
    },
    "K-Nearest Neighbors": {
        "category": "Machine Learning",
        "desc": "KNN classification.",
        "roles": {
            "y": role(CAT, label="Target Class"),
            "x": role(NUM, multi=True, min_n=1),
        },
        "params": {"k": {"type": "number", "default": 5}},
        "assumptions": [],
        "graphs": ["confusion_matrix","scatter"],
    },
    "Naive Bayes Classifier": {
        "category": "Machine Learning",
        "desc": "Gaussian Naive Bayes.",
        "roles": {
            "y": role(CAT, label="Target Class"),
            "x": role(NUM, multi=True, min_n=1),
        },
        "params": {},
        "assumptions": ["Feature independence"],
        "graphs": ["confusion_matrix"],
    },
    "Linear Discriminant Analysis": {
        "category": "Machine Learning",
        "desc": "LDA classification & dimensionality reduction.",
        "roles": {
            "y": role(CAT),
            "x": role(NUM, multi=True, min_n=2),
        },
        "params": {},
        "assumptions": ["Multivariate normality"],
        "graphs": ["scatter","confusion_matrix"],
    },

    # ===== EFFECT SIZES =====
    "Cohen's d (Effect Size)": {
        "category": "Effect Size",
        "desc": "Standardized mean difference between two groups.",
        "roles": {"value": role(NUM), "group": role(CAT)},
        "params": {},
        "assumptions": [],
        "graphs": ["boxplot","violin"],
    },
    "Cramér's V": {
        "category": "Effect Size",
        "desc": "Effect size for chi-square test.",
        "roles": {"var1": role(CAT), "var2": role(CAT)},
        "params": {},
        "assumptions": [],
        "graphs": ["heatmap"],
    },
}


def get_tool(name: str) -> dict:
    return STAT_TOOLS.get(name, {})


def list_tools_by_category() -> dict:
    out: dict[str, list] = {}
    for name, spec in STAT_TOOLS.items():
        out.setdefault(spec["category"], []).append(name)
    return out


def compatible_tools_for(buckets: dict) -> list[str]:
    """Return tools whose role contracts can be satisfied by the dataset's buckets."""
    avail = {t: bool(cols) for t, cols in buckets.items()}
    compat = []
    for name, spec in STAT_TOOLS.items():
        ok = True
        for r in spec["roles"].values():
            if r.get("optional"):
                continue
            if not any(avail.get(t, False) for t in r["types"]):
                ok = False
                break
        if ok:
            compat.append(name)
    return compat
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


# ============================================================
# DESCRIPTIVE
# ============================================================
def _descriptive_stats(df, vars_, p):
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    desc = sub.describe().T
    desc["skew"]     = sub.skew()
    desc["kurtosis"] = sub.kurtosis()
    desc["missing"]  = df[vars_].isna().sum().values
    desc = desc.round(4)
    return {
        "summary": {"variables": len(vars_), "n": len(sub)},
        "tables": {"Descriptive Summary": desc.reset_index().rename(columns={"index": "Variable"})},
        "extras": {"data": sub},
        "p_value": None, "alpha": p.get("alpha", 0.05),
    }


def _frequency_table(df, var, p):
    s = df[var].dropna()
    counts = s.value_counts()
    pct = (counts / len(s) * 100).round(2)
    table = pd.DataFrame({"Category": counts.index.astype(str), "Count": counts.values, "Percent (%)": pct.values})
    return {
        "summary": {"variable": var, "categories": int(s.nunique()), "n": len(s)},
        "tables": {"Frequency": table},
        "extras": {"data": df[[var]].dropna()},
        "p_value": None, "alpha": 0.05,
    }


def _crosstab(df, row, col, p):
    norm = p.get("normalize", "none")
    norm_arg = False if norm == "none" else norm
    ct = pd.crosstab(df[row], df[col], normalize=norm_arg)
    if norm != "none":
        ct = (ct * 100).round(2)
    return {
        "summary": {"rows": ct.shape[0], "cols": ct.shape[1]},
        "tables": {"Cross Tabulation": ct.reset_index()},
        "extras": {"data": df[[row, col]].dropna(), "row": row, "col": col,
                   "matrix": pd.crosstab(df[row], df[col])},
        "p_value": None, "alpha": 0.05,
    }


def _skew_kurt(df, vars_, p):
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    out = pd.DataFrame({
        "Variable": vars_,
        "Skewness": [sub[v].skew() for v in vars_],
        "Kurtosis": [sub[v].kurtosis() for v in vars_],
    }).round(4)
    return {"summary": {}, "tables": {"Distribution Shape": out},
            "extras": {"data": sub}, "p_value": None, "alpha": 0.05}


def _outliers_iqr(df, var, p):
    s = pd.to_numeric(df[var], errors="coerce").dropna()
    k = float(p.get("k", 1.5))
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = (s < lo) | (s > hi)
    return {
        "summary": {"n": len(s), "outliers": int(mask.sum()), "lower": round(lo,4), "upper": round(hi,4)},
        "tables": {"Outlier Bounds": pd.DataFrame({"Metric": ["Q1","Q3","IQR","Lower","Upper","Outliers"],
                                                     "Value": [q1,q3,iqr,lo,hi,int(mask.sum())]}).round(4)},
        "extras": {"data": pd.DataFrame({var: s, "is_outlier": mask})},
        "p_value": None, "alpha": 0.05,
    }


# ============================================================
# HYPOTHESIS TESTS
# ============================================================
def _one_sample_t(df, var, p):
    x = pd.to_numeric(df[var], errors="coerce").dropna()
    mu0 = float(p.get("mu0", 0.0))
    alt = p.get("tail", "two-sided")
    t, pv = sp_stats.ttest_1samp(x, mu0, alternative=alt if alt in ("two-sided","greater","less") else "two-sided")
    return {
        "summary": {"n": len(x), "mean": float(x.mean()), "mu0": mu0, "t": float(t), "p_value": float(pv)},
        "tables": {"One-Sample t-Test": pd.DataFrame({
            "Metric": ["n","Mean","Std","μ₀","t-stat","p-value","df"],
            "Value":  [len(x), x.mean(), x.std(), mu0, t, pv, len(x)-1]}).round(4)},
        "extras": {"data": pd.DataFrame({var: x})},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _independent_t(df, value, group, p):
    sub = df[[value, group]].dropna()
    levels = sub[group].astype(str).unique()
    if len(levels) != 2:
        return {"ok": False, "error": f"Group must have exactly 2 levels (found {len(levels)})."}
    g1 = pd.to_numeric(sub.loc[sub[group].astype(str)==levels[0], value], errors="coerce").dropna()
    g2 = pd.to_numeric(sub.loc[sub[group].astype(str)==levels[1], value], errors="coerce").dropna()
    eq = bool(p.get("equal_var", False))
    alt = p.get("tail", "two-sided")
    t, pv = sp_stats.ttest_ind(g1, g2, equal_var=eq, alternative=alt)
    pooled = np.sqrt(((g1.var(ddof=1)*(len(g1)-1)) + (g2.var(ddof=1)*(len(g2)-1))) / (len(g1)+len(g2)-2))
    d = (g1.mean() - g2.mean()) / pooled if pooled else float("nan")
    return {
        "summary": {"levels": list(levels), "n1": len(g1), "n2": len(g2),
                    "mean1": float(g1.mean()), "mean2": float(g2.mean()),
                    "t": float(t), "p_value": float(pv), "cohens_d": float(d)},
        "tables": {"Independent t-Test": pd.DataFrame({
            "Metric": [f"n ({levels[0]})", f"n ({levels[1]})",
                       f"Mean ({levels[0]})", f"Mean ({levels[1]})",
                       f"SD ({levels[0]})", f"SD ({levels[1]})",
                       "t-statistic", "p-value", "Cohen's d"],
            "Value": [len(g1), len(g2), g1.mean(), g2.mean(),
                     g1.std(), g2.std(), t, pv, d]}).round(4)},
        "extras": {"data": sub, "value": value, "group": group, "levels": list(levels)},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _paired_t(df, v1, v2, p):
    sub = df[[v1, v2]].apply(pd.to_numeric, errors="coerce").dropna()
    t, pv = sp_stats.ttest_rel(sub[v1], sub[v2])
    diff = sub[v1] - sub[v2]
    d = diff.mean() / diff.std() if diff.std() else float("nan")
    return {
        "summary": {"n": len(sub), "mean1": float(sub[v1].mean()), "mean2": float(sub[v2].mean()),
                    "t": float(t), "p_value": float(pv), "cohens_d": float(d)},
        "tables": {"Paired t-Test": pd.DataFrame({
            "Metric": ["n","Mean Diff","SD Diff","t","p-value","Cohen's d"],
            "Value":  [len(sub), diff.mean(), diff.std(), t, pv, d]}).round(4)},
        "extras": {"data": sub, "v1": v1, "v2": v2},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _z_test(df, var, p):
    x = pd.to_numeric(df[var], errors="coerce").dropna()
    mu0 = float(p.get("mu0", 0.0)); sigma = float(p.get("sigma", 1.0))
    z = (x.mean() - mu0) / (sigma / np.sqrt(len(x)))
    pv = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return {
        "summary": {"z": float(z), "p_value": float(pv), "n": len(x), "mean": float(x.mean())},
        "tables": {"Z-Test": pd.DataFrame({
            "Metric": ["n","Mean","μ₀","σ","z-stat","p-value"],
            "Value":  [len(x), x.mean(), mu0, sigma, z, pv]}).round(4)},
        "extras": {"data": pd.DataFrame({var: x})},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _proportion_z(df, var, p):
    s = df[var].dropna().astype(str)
    successes = (s == s.mode().iloc[0]).sum()
    n = len(s); p_hat = successes / n
    p0 = float(p.get("p0", 0.5))
    z = (p_hat - p0) / np.sqrt(p0 * (1-p0) / n)
    pv = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return {
        "summary": {"p_hat": float(p_hat), "p0": p0, "z": float(z), "p_value": float(pv), "n": n},
        "tables": {"Proportion Z-Test": pd.DataFrame({
            "Metric": ["n","Successes","p̂","p₀","z","p-value"],
            "Value":  [n, successes, p_hat, p0, z, pv]}).round(4)},
        "extras": {"data": df[[var]].dropna()},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _chi2_independence(df, v1, v2, p):
    ct = pd.crosstab(df[v1], df[v2])
    if ct.size == 0:
        return {"ok": False, "error": "Empty contingency table."}
    chi2, pv, dof, exp = sp_stats.chi2_contingency(ct)
    n = ct.sum().sum()
    cramer = np.sqrt(chi2 / (n * (min(ct.shape)-1))) if min(ct.shape) > 1 else 0.0
    return {
        "summary": {"chi2": float(chi2), "p_value": float(pv), "df": int(dof),
                    "cramers_v": float(cramer), "n": int(n)},
        "tables": {
            "Observed":    ct.reset_index(),
            "Expected":    pd.DataFrame(exp, index=ct.index, columns=ct.columns).round(2).reset_index(),
            "Test Result": pd.DataFrame({"Metric": ["χ²","df","p-value","Cramér's V","n"],
                                          "Value":  [chi2, dof, pv, cramer, n]}).round(4),
        },
        "extras": {"matrix": ct, "v1": v1, "v2": v2, "data": df[[v1,v2]].dropna()},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _chi2_gof(df, var, p):
    counts = df[var].dropna().value_counts().sort_index()
    expected = np.full(len(counts), counts.sum() / len(counts))
    chi2, pv = sp_stats.chisquare(counts.values, expected)
    return {
        "summary": {"chi2": float(chi2), "p_value": float(pv), "categories": int(len(counts))},
        "tables": {"Goodness of Fit": pd.DataFrame({
            "Category": counts.index.astype(str),
            "Observed": counts.values,
            "Expected": expected.round(2)})},
        "extras": {"data": df[[var]].dropna()},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _fisher_exact(df, v1, v2, p):
    ct = pd.crosstab(df[v1], df[v2])
    if ct.shape != (2,2):
        return {"ok": False, "error": f"Need 2×2 table; got {ct.shape}."}
    odds, pv = sp_stats.fisher_exact(ct)
    return {
        "summary": {"odds_ratio": float(odds), "p_value": float(pv)},
        "tables": {"Fisher's Exact Test": pd.DataFrame({
            "Metric": ["Odds Ratio","p-value"],
            "Value":  [odds, pv]}).round(4),
            "Contingency Table": ct.reset_index()},
        "extras": {"matrix": ct, "data": df[[v1,v2]].dropna()},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }
# ============================================================
# CORRELATION
# ============================================================
def _correlation(method):
    def _run(df, x, y, p):
        sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if method == "pearson":   r, pv = sp_stats.pearsonr(sub[x], sub[y])
        elif method == "spearman": r, pv = sp_stats.spearmanr(sub[x], sub[y])
        else:                       r, pv = sp_stats.kendalltau(sub[x], sub[y])
        return {
            "summary": {"r": float(r), "p_value": float(pv), "n": len(sub), "method": method},
            "tables": {f"{method.title()} Correlation": pd.DataFrame({
                "Metric": ["n","Coefficient (r)","p-value"],
                "Value":  [len(sub), r, pv]}).round(4)},
            "extras": {"data": sub, "x": x, "y": y},
            "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
        }
    return _run


def _correlation_matrix(df, vars_, p):
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    method = p.get("method", "pearson")
    cm = sub.corr(method=method).round(4)
    return {
        "summary": {"variables": len(vars_), "n": len(sub), "method": method},
        "tables": {"Correlation Matrix": cm.reset_index()},
        "extras": {"matrix": cm, "data": sub},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _point_biserial(df, binary, num, p):
    sub = df[[binary, num]].dropna()
    sub[binary] = pd.factorize(sub[binary])[0]
    sub[num]    = pd.to_numeric(sub[num], errors="coerce")
    sub = sub.dropna()
    r, pv = sp_stats.pointbiserialr(sub[binary], sub[num])
    return {
        "summary": {"r_pb": float(r), "p_value": float(pv), "n": len(sub)},
        "tables": {"Point-Biserial": pd.DataFrame({
            "Metric": ["n","r_pb","p-value"], "Value":[len(sub), r, pv]}).round(4)},
        "extras": {"data": sub, "binary": binary, "numeric": num},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


# ============================================================
# REGRESSION
# ============================================================
def _simple_linear(df, y, x, p):
    sub = df[[y, x]].apply(pd.to_numeric, errors="coerce").dropna()
    slope, intercept, r, pv, se = sp_stats.linregress(sub[x], sub[y])
    pred = intercept + slope * sub[x]
    resid = sub[y] - pred
    return {
        "summary": {"slope": float(slope), "intercept": float(intercept),
                    "r2": float(r**2), "p_value": float(pv), "n": len(sub)},
        "tables": {"Linear Regression": pd.DataFrame({
            "Metric": ["n","Slope (β₁)","Intercept (β₀)","R²","p-value","Std Error"],
            "Value":  [len(sub), slope, intercept, r**2, pv, se]}).round(4)},
        "extras": {"data": sub, "x": x, "y": y, "pred": pred, "resid": resid,
                   "equation": f"{y} = {intercept:.3f} + {slope:.3f} × {x}"},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _multiple_linear(df, y, xs, p):
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    sub = df[[y] + xs].apply(pd.to_numeric, errors="coerce").dropna()
    X = sm.add_constant(sub[xs])
    model = sm.OLS(sub[y], X).fit()
    coef_tbl = pd.DataFrame({
        "Variable": ["Intercept"] + xs,
        "Coefficient": model.params.values,
        "Std Error": model.bse.values,
        "t": model.tvalues.values,
        "p-value": model.pvalues.values,
    }).round(4)
    return {
        "summary": {"r2": float(model.rsquared), "adj_r2": float(model.rsquared_adj),
                    "f_pvalue": float(model.f_pvalue), "n": int(model.nobs)},
        "tables": {"Coefficients": coef_tbl,
                   "Model Fit": pd.DataFrame({"Metric": ["R²","Adj R²","F-stat","Prob (F)","AIC","BIC","n"],
                                              "Value":  [model.rsquared, model.rsquared_adj,
                                                          model.fvalue, model.f_pvalue,
                                                          model.aic, model.bic, model.nobs]}).round(4)},
        "extras": {"data": sub, "y": y, "xs": xs, "pred": model.predict(X), "resid": model.resid},
        "p_value": float(model.f_pvalue), "alpha": float(p.get("alpha",0.05)),
    }


def _polynomial(df, y, x, p):
    sub = df[[y, x]].apply(pd.to_numeric, errors="coerce").dropna()
    deg = int(p.get("degree", 2))
    coefs = np.polyfit(sub[x], sub[y], deg)
    pred = np.polyval(coefs, sub[x])
    ss_res = float(((sub[y]-pred)**2).sum()); ss_tot = float(((sub[y]-sub[y].mean())**2).sum())
    r2 = 1 - ss_res/ss_tot if ss_tot else 0
    return {
        "summary": {"degree": deg, "r2": float(r2), "n": len(sub)},
        "tables": {"Polynomial Fit": pd.DataFrame({
            "Coefficient": [f"x^{deg-i}" for i in range(deg+1)],
            "Value": coefs}).round(4)},
        "extras": {"data": sub, "x": x, "y": y, "pred": pd.Series(pred, index=sub.index)},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _logistic(df, y, xs, p):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
    except ImportError:
        return {"ok": False, "error": "scikit-learn required"}
    sub = df[[y] + xs].dropna()
    sub[xs] = sub[xs].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna()
    y_enc, levels = pd.factorize(sub[y])
    if len(levels) != 2:
        return {"ok": False, "error": f"Y must be binary (got {len(levels)} levels)."}
    model = LogisticRegression(max_iter=1000)
    model.fit(sub[xs], y_enc)
    preds = model.predict(sub[xs])
    probs = model.predict_proba(sub[xs])[:,1]
    acc = accuracy_score(y_enc, preds)
    try: auc = roc_auc_score(y_enc, probs)
    except Exception: auc = float("nan")
    coef_tbl = pd.DataFrame({
        "Variable": ["Intercept"] + xs,
        "Coefficient": [model.intercept_[0]] + list(model.coef_[0]),
        "Odds Ratio": [np.exp(model.intercept_[0])] + list(np.exp(model.coef_[0])),
    }).round(4)
    return {
        "summary": {"accuracy": float(acc), "auc": float(auc), "n": len(sub), "classes": list(levels)},
        "tables": {"Coefficients": coef_tbl,
                   "Performance": pd.DataFrame({"Metric": ["Accuracy","AUC","n"],
                                                "Value": [acc, auc, len(sub)]}).round(4)},
        "extras": {"data": sub, "y": y, "xs": xs, "preds": preds, "probs": probs, "y_true": y_enc, "levels": list(levels)},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _ridge_lasso(kind):
    def _run(df, y, xs, p):
        try:
            from sklearn.linear_model import Ridge, Lasso
        except ImportError:
            return {"ok": False, "error": "scikit-learn required"}
        sub = df[[y]+xs].apply(pd.to_numeric, errors="coerce").dropna()
        Cls = Ridge if kind=="ridge" else Lasso
        model = Cls(alpha=float(p.get("alpha",1.0))).fit(sub[xs], sub[y])
        pred = model.predict(sub[xs]); resid = sub[y] - pred
        ss_res = float((resid**2).sum()); ss_tot = float(((sub[y]-sub[y].mean())**2).sum())
        r2 = 1 - ss_res/ss_tot if ss_tot else 0
        coef_tbl = pd.DataFrame({"Variable":["Intercept"]+xs,
                                  "Coefficient":[model.intercept_]+list(model.coef_)}).round(4)
        return {
            "summary": {"r2": float(r2), "n": len(sub), "regularization": kind},
            "tables": {"Coefficients": coef_tbl},
            "extras": {"data": sub, "y": y, "xs": xs, "pred": pd.Series(pred, index=sub.index), "resid": resid},
            "p_value": None, "alpha": float(p.get("alpha",0.05)),
        }
    return _run


# ============================================================
# ANOVA
# ============================================================
def _one_way_anova(df, value, group, p):
    sub = df[[value, group]].dropna()
    sub[value] = pd.to_numeric(sub[value], errors="coerce"); sub = sub.dropna()
    groups = [g[value].values for _, g in sub.groupby(group)]
    if len(groups) < 2:
        return {"ok": False, "error": "Need at least 2 groups."}
    F, pv = sp_stats.f_oneway(*groups)
    means = sub.groupby(group)[value].agg(["count","mean","std"]).round(4).reset_index()
    return {
        "summary": {"F": float(F), "p_value": float(pv), "groups": int(len(groups)), "n": len(sub)},
        "tables": {"ANOVA Table": pd.DataFrame({"Source":["Between","Within"],
                                                  "F":[F,np.nan], "p-value":[pv,np.nan]}).round(4),
                   "Group Statistics": means},
        "extras": {"data": sub, "value": value, "group": group},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _two_way_anova(df, value, f1, f2, p):
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    sub = df[[value, f1, f2]].dropna()
    sub[value] = pd.to_numeric(sub[value], errors="coerce"); sub = sub.dropna()
    sub.columns = ["VAL","F1","F2"]
    model = ols("VAL ~ C(F1) + C(F2) + C(F1):C(F2)", data=sub).fit()
    table = sm.stats.anova_lm(model, typ=2).round(4).reset_index().rename(columns={"index":"Source"})
    pv = float(table["PR(>F)"].iloc[0]) if "PR(>F)" in table.columns else None
    return {
        "summary": {"r2": float(model.rsquared), "n": int(model.nobs)},
        "tables": {"Two-Way ANOVA": table},
        "extras": {"data": sub.rename(columns={"VAL":value,"F1":f1,"F2":f2}), "value": value, "f1": f1, "f2": f2},
        "p_value": pv, "alpha": float(p.get("alpha",0.05)),
    }


def _rm_anova(df, vars_, p):
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    F, pv = sp_stats.f_oneway(*[sub[c].values for c in vars_])
    return {
        "summary": {"F": float(F), "p_value": float(pv), "k": len(vars_), "n": len(sub)},
        "tables": {"Repeated Measures ANOVA": pd.DataFrame({
            "Metric":["F","p-value","Conditions","n"], "Value":[F,pv,len(vars_),len(sub)]}).round(4)},
        "extras": {"data": sub, "vars": vars_},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _ancova(df, value, group, cov, p):
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    sub = df[[value, group, cov]].dropna()
    sub[value] = pd.to_numeric(sub[value], errors="coerce")
    sub[cov]   = pd.to_numeric(sub[cov], errors="coerce")
    sub = sub.dropna()
    sub.columns = ["VAL","GRP","COV"]
    model = ols("VAL ~ C(GRP) + COV", data=sub).fit()
    table = sm.stats.anova_lm(model, typ=2).round(4).reset_index().rename(columns={"index":"Source"})
    return {
        "summary": {"r2": float(model.rsquared), "n": int(model.nobs)},
        "tables": {"ANCOVA": table},
        "extras": {"data": sub.rename(columns={"VAL":value,"GRP":group,"COV":cov})},
        "p_value": float(table["PR(>F)"].iloc[0]) if "PR(>F)" in table.columns else None,
        "alpha": float(p.get("alpha",0.05)),
    }


def _tukey(df, value, group, p):
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    sub = df[[value, group]].dropna()
    sub[value] = pd.to_numeric(sub[value], errors="coerce"); sub = sub.dropna()
    res = pairwise_tukeyhsd(sub[value], sub[group], alpha=float(p.get("alpha",0.05)))
    table = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    return {
        "summary": {"comparisons": len(table), "alpha": float(p.get("alpha",0.05))},
        "tables": {"Tukey HSD": table},
        "extras": {"data": sub, "value": value, "group": group},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }
# ============================================================
# NON-PARAMETRIC
# ============================================================
def _mann_whitney(df, value, group, p):
    sub = df[[value, group]].dropna()
    levels = sub[group].astype(str).unique()
    if len(levels) != 2:
        return {"ok": False, "error": f"Need exactly 2 groups (got {len(levels)})."}
    g1 = pd.to_numeric(sub.loc[sub[group].astype(str)==levels[0], value], errors="coerce").dropna()
    g2 = pd.to_numeric(sub.loc[sub[group].astype(str)==levels[1], value], errors="coerce").dropna()
    U, pv = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
    return {
        "summary": {"U": float(U), "p_value": float(pv), "n1": len(g1), "n2": len(g2)},
        "tables": {"Mann-Whitney U": pd.DataFrame({
            "Metric":[f"n ({levels[0]})",f"n ({levels[1]})","U","p-value"],
            "Value":[len(g1), len(g2), U, pv]}).round(4)},
        "extras": {"data": sub, "value": value, "group": group, "levels": list(levels)},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _wilcoxon(df, v1, v2, p):
    sub = df[[v1,v2]].apply(pd.to_numeric, errors="coerce").dropna()
    W, pv = sp_stats.wilcoxon(sub[v1], sub[v2])
    return {
        "summary": {"W": float(W), "p_value": float(pv), "n": len(sub)},
        "tables": {"Wilcoxon": pd.DataFrame({
            "Metric":["n","W","p-value"], "Value":[len(sub), W, pv]}).round(4)},
        "extras": {"data": sub},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _kruskal(df, value, group, p):
    sub = df[[value,group]].dropna()
    sub[value] = pd.to_numeric(sub[value], errors="coerce"); sub = sub.dropna()
    groups = [g[value].values for _,g in sub.groupby(group)]
    H, pv = sp_stats.kruskal(*groups)
    means = sub.groupby(group)[value].agg(["count","mean","median"]).round(4).reset_index()
    return {
        "summary": {"H": float(H), "p_value": float(pv), "groups": len(groups), "n": len(sub)},
        "tables": {"Kruskal-Wallis": pd.DataFrame({
            "Metric":["H","p-value","Groups","n"], "Value":[H, pv, len(groups), len(sub)]}).round(4),
                   "Group Stats": means},
        "extras": {"data": sub, "value": value, "group": group},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _friedman(df, vars_, p):
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    chi2, pv = sp_stats.friedmanchisquare(*[sub[c].values for c in vars_])
    return {
        "summary": {"chi2": float(chi2), "p_value": float(pv), "k": len(vars_)},
        "tables": {"Friedman": pd.DataFrame({"Metric":["χ²","p-value"],"Value":[chi2,pv]}).round(4)},
        "extras": {"data": sub, "vars": vars_},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _shapiro(df, var, p):
    x = pd.to_numeric(df[var], errors="coerce").dropna()
    if len(x) > 5000: x = x.sample(5000, random_state=0)
    W, pv = sp_stats.shapiro(x)
    return {
        "summary": {"W": float(W), "p_value": float(pv), "n": len(x)},
        "tables": {"Shapiro-Wilk": pd.DataFrame({"Metric":["W","p-value","n"],"Value":[W,pv,len(x)]}).round(4)},
        "extras": {"data": pd.DataFrame({var:x})},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _levene(df, value, group, p):
    sub = df[[value,group]].dropna()
    sub[value] = pd.to_numeric(sub[value], errors="coerce"); sub = sub.dropna()
    groups = [g[value].values for _,g in sub.groupby(group)]
    W, pv = sp_stats.levene(*groups)
    return {
        "summary": {"W": float(W), "p_value": float(pv), "groups": len(groups)},
        "tables": {"Levene's Test": pd.DataFrame({"Metric":["W","p-value","Groups"],"Value":[W,pv,len(groups)]}).round(4)},
        "extras": {"data": sub, "value": value, "group": group},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _ks(df, var, p):
    x = pd.to_numeric(df[var], errors="coerce").dropna()
    z = (x - x.mean()) / x.std()
    D, pv = sp_stats.kstest(z, "norm")
    return {
        "summary": {"D": float(D), "p_value": float(pv), "n": len(x)},
        "tables": {"KS Test": pd.DataFrame({"Metric":["D","p-value","n"],"Value":[D,pv,len(x)]}).round(4)},
        "extras": {"data": pd.DataFrame({var:x})},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


# ============================================================
# TIME SERIES / FORECASTING
# ============================================================
def _decompose(df, time, value, p):
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    sub = df[[time, value]].dropna().copy()
    sub[time] = pd.to_datetime(sub[time], errors="coerce")
    sub[value] = pd.to_numeric(sub[value], errors="coerce")
    sub = sub.dropna().sort_values(time)
    period = int(p.get("period", 12))
    if len(sub) < 2*period:
        return {"ok": False, "error": f"Need at least {2*period} observations."}
    res = seasonal_decompose(sub[value].values, period=period, extrapolate_trend="freq")
    decomp = pd.DataFrame({
        time: sub[time].values, "Observed": sub[value].values,
        "Trend": res.trend, "Seasonal": res.seasonal, "Residual": res.resid,
    })
    return {
        "summary": {"period": period, "n": len(sub)},
        "tables": {"Decomposition (head)": decomp.head(15).round(4)},
        "extras": {"data": decomp, "time": time, "value": value},
        "p_value": None, "alpha": 0.05,
    }


def _adf(df, value, p):
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    x = pd.to_numeric(df[value], errors="coerce").dropna()
    res = adfuller(x)
    return {
        "summary": {"adf_stat": float(res[0]), "p_value": float(res[1]), "lags": int(res[2]), "n": int(res[3])},
        "tables": {"ADF Test": pd.DataFrame({
            "Metric":["ADF Statistic","p-value","Lags","n","Critical 1%","Critical 5%","Critical 10%"],
            "Value":[res[0], res[1], res[2], res[3], res[4]["1%"], res[4]["5%"], res[4]["10%"]]}).round(4)},
        "extras": {"data": pd.DataFrame({value: x.values})},
        "p_value": float(res[1]), "alpha": float(p.get("alpha",0.05)),
    }


def _arima(df, value, p):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    x = pd.to_numeric(df[value], errors="coerce").dropna().reset_index(drop=True)
    order = (int(p.get("p",1)), int(p.get("d",1)), int(p.get("q",1)))
    h = int(p.get("horizon", 10))
    model = ARIMA(x, order=order).fit()
    fc = model.forecast(steps=h)
    fc_df = pd.DataFrame({"Step": range(1,h+1), "Forecast": fc.values}).round(4)
    return {
        "summary": {"order": order, "aic": float(model.aic), "bic": float(model.bic), "n": len(x)},
        "tables": {"ARIMA Forecast": fc_df},
        "extras": {"history": x, "forecast": fc, "value": value},
        "p_value": None, "alpha": 0.05,
    }


def _expsmooth(df, value, p):
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    x = pd.to_numeric(df[value], errors="coerce").dropna().reset_index(drop=True)
    h = int(p.get("horizon", 10))
    model = ExponentialSmoothing(x).fit()
    fc = model.forecast(h)
    fc_df = pd.DataFrame({"Step": range(1,h+1), "Forecast": fc.values}).round(4)
    return {
        "summary": {"aic": float(model.aic) if hasattr(model,"aic") else None, "n": len(x)},
        "tables": {"Exponential Smoothing": fc_df},
        "extras": {"history": x, "forecast": fc, "value": value},
        "p_value": None, "alpha": 0.05,
    }


def _autocorr(df, value, p):
    try:
        from statsmodels.tsa.stattools import acf, pacf
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    x = pd.to_numeric(df[value], errors="coerce").dropna()
    lags = int(p.get("lags", 20))
    a = acf(x, nlags=lags); pa = pacf(x, nlags=lags)
    table = pd.DataFrame({"Lag": range(len(a)), "ACF": a, "PACF": pa}).round(4)
    return {
        "summary": {"lags": lags, "n": len(x)},
        "tables": {"ACF / PACF": table},
        "extras": {"data": table},
        "p_value": None, "alpha": 0.05,
    }


# ============================================================
# CLUSTERING / DIMENSIONALITY
# ============================================================
def _kmeans(df, vars_, p):
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"ok": False, "error": "scikit-learn required"}
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    X = StandardScaler().fit_transform(sub)
    k = int(p.get("k", 3))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    sub = sub.copy(); sub["Cluster"] = km.labels_
    cnt = sub["Cluster"].value_counts().sort_index().reset_index()
    cnt.columns = ["Cluster","Count"]
    return {
        "summary": {"k": k, "inertia": float(km.inertia_), "n": len(sub)},
        "tables": {"Cluster Sizes": cnt,
                   "Cluster Centers (z-scored)": pd.DataFrame(km.cluster_centers_, columns=vars_).round(4).reset_index().rename(columns={"index":"Cluster"})},
        "extras": {"data": sub, "vars": vars_},
        "p_value": None, "alpha": 0.05,
    }


def _hclust(df, vars_, p):
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"ok": False, "error": "scipy required"}
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    X = StandardScaler().fit_transform(sub)
    Z = linkage(X, method=p.get("method","ward"))
    clusters = fcluster(Z, t=int(p.get("clusters",3)), criterion="maxclust")
    sub = sub.copy(); sub["Cluster"] = clusters
    return {
        "summary": {"clusters": int(p.get("clusters",3)), "method": p.get("method","ward"), "n": len(sub)},
        "tables": {"Cluster Sizes": sub["Cluster"].value_counts().sort_index().reset_index().rename(columns={"index":"Cluster","Cluster":"Count"})},
        "extras": {"data": sub, "linkage": Z, "vars": vars_},
        "p_value": None, "alpha": 0.05,
    }


def _dbscan(df, vars_, p):
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"ok": False, "error": "scikit-learn required"}
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    X = StandardScaler().fit_transform(sub)
    labs = DBSCAN(eps=float(p.get("eps",0.5)), min_samples=int(p.get("min_samples",5))).fit_predict(X)
    sub = sub.copy(); sub["Cluster"] = labs
    return {
        "summary": {"clusters": int(len(set(labs)) - (1 if -1 in labs else 0)),
                    "noise": int((labs==-1).sum()), "n": len(sub)},
        "tables": {"Cluster Sizes": sub["Cluster"].value_counts().sort_index().reset_index().rename(columns={"index":"Cluster","Cluster":"Count"})},
        "extras": {"data": sub, "vars": vars_},
        "p_value": None, "alpha": 0.05,
    }


def _pca(df, vars_, p):
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"ok": False, "error": "scikit-learn required"}
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    X = StandardScaler().fit_transform(sub)
    n = int(p.get("n_components", 2))
    pca = PCA(n_components=n).fit(X)
    scores = pca.transform(X)
    var_table = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(n)],
        "Explained Variance": pca.explained_variance_ratio_,
        "Cumulative": np.cumsum(pca.explained_variance_ratio_),
    }).round(4)
    loadings = pd.DataFrame(pca.components_.T, index=vars_,
                              columns=[f"PC{i+1}" for i in range(n)]).round(4).reset_index().rename(columns={"index":"Variable"})
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n)])
    return {
        "summary": {"n_components": n, "n": len(sub),
                    "total_variance": float(pca.explained_variance_ratio_.sum())},
        "tables": {"Explained Variance": var_table, "Loadings": loadings},
        "extras": {"scores": scores_df, "vars": vars_, "data": sub.reset_index(drop=True).join(scores_df)},
        "p_value": None, "alpha": 0.05,
    }


def _factor_analysis(df, vars_, p):
    try:
        from sklearn.decomposition import FactorAnalysis
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"ok": False, "error": "scikit-learn required"}
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    X = StandardScaler().fit_transform(sub)
    n = int(p.get("n_factors", 2))
    fa = FactorAnalysis(n_components=n).fit(X)
    loadings = pd.DataFrame(fa.components_.T, index=vars_,
                              columns=[f"Factor{i+1}" for i in range(n)]).round(4).reset_index().rename(columns={"index":"Variable"})
    return {
        "summary": {"n_factors": n, "n": len(sub)},
        "tables": {"Loadings": loadings},
        "extras": {"data": sub, "vars": vars_},
        "p_value": None, "alpha": 0.05,
    }
# ============================================================
# SURVIVAL
# ============================================================
def _km(df, duration, event, p):
    sub = df[[duration, event]].dropna()
    sub[duration] = pd.to_numeric(sub[duration], errors="coerce")
    sub[event] = pd.to_numeric(sub[event], errors="coerce")
    sub = sub.dropna().sort_values(duration)
    times = sub[duration].values; events = sub[event].values
    n = len(sub); surv = []; t_arr = []; at_risk = n
    cum = 1.0
    for t in np.unique(times):
        d = ((times == t) & (events == 1)).sum()
        if at_risk > 0:
            cum *= (1 - d/at_risk)
        t_arr.append(t); surv.append(cum)
        at_risk -= (times == t).sum()
    table = pd.DataFrame({"Time": t_arr, "Survival Probability": surv}).round(4)
    return {
        "summary": {"n": n, "events": int(events.sum()), "median_time": float(np.median(times))},
        "tables": {"Kaplan-Meier": table},
        "extras": {"data": sub, "time": t_arr, "surv": surv, "duration": duration, "event": event},
        "p_value": None, "alpha": 0.05,
    }


def _cox(df, duration, event, covariates, p):
    try:
        from lifelines import CoxPHFitter
        sub = df[[duration, event] + covariates].dropna()
        cph = CoxPHFitter().fit(sub, duration_col=duration, event_col=event)
        return {
            "summary": {"concordance": float(cph.concordance_index_), "n": len(sub)},
            "tables": {"Cox Coefficients": cph.summary.round(4).reset_index()},
            "extras": {"data": sub, "duration": duration, "event": event, "covariates": covariates},
            "p_value": None, "alpha": 0.05,
        }
    except ImportError:
        return {"ok": False, "error": "lifelines not installed (pip install lifelines)"}


# ============================================================
# ML BASICS
# ============================================================
def _ml_classifier(kind):
    def _run(df, y, xs, p):
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.metrics import accuracy_score, confusion_matrix
            from sklearn.model_selection import train_test_split
        except ImportError:
            return {"ok": False, "error": "scikit-learn required"}
        sub = df[[y]+xs].dropna()
        sub[xs] = sub[xs].apply(pd.to_numeric, errors="coerce"); sub = sub.dropna()
        y_enc, levels = pd.factorize(sub[y])
        Xtr, Xte, ytr, yte = train_test_split(sub[xs], y_enc, test_size=0.25, random_state=42,
                                                 stratify=y_enc if len(set(y_enc))>1 else None)
        models = {
            "decision_tree":  DecisionTreeClassifier(max_depth=int(p.get("max_depth",5)), random_state=42),
            "random_forest":  RandomForestClassifier(n_estimators=int(p.get("n_estimators",100)),
                                                       max_depth=int(p.get("max_depth",10)), random_state=42),
            "knn":            KNeighborsClassifier(n_neighbors=int(p.get("k",5))),
            "naive_bayes":    GaussianNB(),
            "lda":            LinearDiscriminantAnalysis(),
        }
        model = models[kind].fit(Xtr, ytr)
        preds = model.predict(Xte)
        acc = accuracy_score(yte, preds)
        cm = confusion_matrix(yte, preds)
        cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in levels],
                                columns=[f"Pred {l}" for l in levels]).reset_index().rename(columns={"index":"Class"})
        importance_tbl = None
        if hasattr(model, "feature_importances_"):
            importance_tbl = pd.DataFrame({"Feature": xs, "Importance": model.feature_importances_}).round(4).sort_values("Importance", ascending=False)
        tables = {"Performance": pd.DataFrame({"Metric":["Accuracy","Train n","Test n","Classes"],
                                                  "Value":[acc, len(Xtr), len(Xte), len(levels)]}).round(4),
                  "Confusion Matrix": cm_df}
        if importance_tbl is not None:
            tables["Feature Importance"] = importance_tbl
        return {
            "summary": {"accuracy": float(acc), "classes": list(levels), "n_train": len(Xtr), "n_test": len(Xte)},
            "tables": tables,
            "extras": {"data": sub, "y": y, "xs": xs, "y_pred": preds, "y_true": yte,
                       "cm": cm, "levels": list(levels), "model": model},
            "p_value": None, "alpha": 0.05,
        }
    return _run


# ============================================================
# EFFECT SIZE
# ============================================================
def _cohens_d(df, value, group, p):
    sub = df[[value, group]].dropna()
    levels = sub[group].astype(str).unique()
    if len(levels) != 2:
        return {"ok": False, "error": "Need 2 groups."}
    g1 = pd.to_numeric(sub.loc[sub[group].astype(str)==levels[0], value], errors="coerce").dropna()
    g2 = pd.to_numeric(sub.loc[sub[group].astype(str)==levels[1], value], errors="coerce").dropna()
    pooled = np.sqrt(((g1.var(ddof=1)*(len(g1)-1))+(g2.var(ddof=1)*(len(g2)-1)))/(len(g1)+len(g2)-2))
    d = (g1.mean()-g2.mean())/pooled if pooled else float("nan")
    return {
        "summary": {"cohens_d": float(d), "magnitude": _interp_d(d)},
        "tables": {"Cohen's d": pd.DataFrame({
            "Metric":["d","|d|","Magnitude"], "Value":[d, abs(d), _interp_d(d)]})},
        "extras": {"data": sub, "value": value, "group": group, "levels": list(levels)},
        "p_value": None, "alpha": 0.05,
    }


def _interp_d(d):
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


def _cramers_v(df, v1, v2, p):
    ct = pd.crosstab(df[v1], df[v2])
    chi2 = sp_stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    v = np.sqrt(chi2 / (n*(min(ct.shape)-1))) if min(ct.shape)>1 else 0
    return {
        "summary": {"cramers_v": float(v), "n": int(n)},
        "tables": {"Cramér's V": pd.DataFrame({"Metric":["V","n"],"Value":[v,n]}).round(4)},
        "extras": {"matrix": ct, "data": df[[v1,v2]].dropna()},
        "p_value": None, "alpha": 0.05,
    }


# ============================================================
# DISPATCHER
# ============================================================
DISPATCH = {
    "Descriptive Statistics":      lambda df,vars_,**p: _descriptive_stats(df, vars_["variables"], p),
    "Frequency Table":             lambda df,vars_,**p: _frequency_table(df, vars_["variable"], p),
    "Cross Tabulation":            lambda df,vars_,**p: _crosstab(df, vars_["row"], vars_["col"], p),
    "Skewness & Kurtosis":         lambda df,vars_,**p: _skew_kurt(df, vars_["variables"], p),
    "Outlier Detection (IQR)":     lambda df,vars_,**p: _outliers_iqr(df, vars_["variable"], p),

    "One-Sample t-Test":           lambda df,vars_,**p: _one_sample_t(df, vars_["variable"], p),
    "Independent t-Test":          lambda df,vars_,**p: _independent_t(df, vars_["value"], vars_["group"], p),
    "Paired t-Test":               lambda df,vars_,**p: _paired_t(df, vars_["var1"], vars_["var2"], p),
    "Z-Test (One Sample)":         lambda df,vars_,**p: _z_test(df, vars_["variable"], p),
    "Proportion Z-Test":           lambda df,vars_,**p: _proportion_z(df, vars_["variable"], p),
    "Chi-Square Test of Independence": lambda df,vars_,**p: _chi2_independence(df, vars_["var1"], vars_["var2"], p),
    "Chi-Square Goodness of Fit":  lambda df,vars_,**p: _chi2_gof(df, vars_["variable"], p),
    "Fisher's Exact Test":         lambda df,vars_,**p: _fisher_exact(df, vars_["var1"], vars_["var2"], p),

    "Pearson Correlation":         lambda df,vars_,**p: _correlation("pearson")(df, vars_["x"], vars_["y"], p),
    "Spearman Correlation":        lambda df,vars_,**p: _correlation("spearman")(df, vars_["x"], vars_["y"], p),
    "Kendall's Tau":               lambda df,vars_,**p: _correlation("kendall")(df, vars_["x"], vars_["y"], p),
    "Correlation Matrix":          lambda df,vars_,**p: _correlation_matrix(df, vars_["variables"], p),
    "Point-Biserial Correlation":  lambda df,vars_,**p: _point_biserial(df, vars_["binary"], vars_["numeric"], p),

    "Simple Linear Regression":    lambda df,vars_,**p: _simple_linear(df, vars_["y"], vars_["x"], p),
    "Multiple Linear Regression":  lambda df,vars_,**p: _multiple_linear(df, vars_["y"], vars_["x"], p),
    "Polynomial Regression":       lambda df,vars_,**p: _polynomial(df, vars_["y"], vars_["x"], p),
    "Logistic Regression":         lambda df,vars_,**p: _logistic(df, vars_["y"], vars_["x"], p),
    "Ridge Regression":            lambda df,vars_,**p: _ridge_lasso("ridge")(df, vars_["y"], vars_["x"], p),
    "Lasso Regression":            lambda df,vars_,**p: _ridge_lasso("lasso")(df, vars_["y"], vars_["x"], p),

    "One-Way ANOVA":               lambda df,vars_,**p: _one_way_anova(df, vars_["value"], vars_["group"], p),
    "Two-Way ANOVA":               lambda df,vars_,**p: _two_way_anova(df, vars_["value"], vars_["factor1"], vars_["factor2"], p),
    "Repeated Measures ANOVA":     lambda df,vars_,**p: _rm_anova(df, vars_["variables"], p),
    "ANCOVA":                      lambda df,vars_,**p: _ancova(df, vars_["value"], vars_["group"], vars_["covariate"], p),
    "Tukey HSD Post-Hoc":          lambda df,vars_,**p: _tukey(df, vars_["value"], vars_["group"], p),

    "Mann-Whitney U Test":         lambda df,vars_,**p: _mann_whitney(df, vars_["value"], vars_["group"], p),
    "Wilcoxon Signed-Rank Test":   lambda df,vars_,**p: _wilcoxon(df, vars_["var1"], vars_["var2"], p),
    "Kruskal-Wallis Test":         lambda df,vars_,**p: _kruskal(df, vars_["value"], vars_["group"], p),
    "Friedman Test":               lambda df,vars_,**p: _friedman(df, vars_["variables"], p),
    "Shapiro-Wilk Normality":      lambda df,vars_,**p: _shapiro(df, vars_["variable"], p),
    "Levene's Test":               lambda df,vars_,**p: _levene(df, vars_["value"], vars_["group"], p),
    "Kolmogorov-Smirnov Test":     lambda df,vars_,**p: _ks(df, vars_["variable"], p),

    "Time Series Decomposition":   lambda df,vars_,**p: _decompose(df, vars_["time"], vars_["value"], p),
    "Stationarity (ADF Test)":     lambda df,vars_,**p: _adf(df, vars_["value"], p),
    "ARIMA Forecast":              lambda df,vars_,**p: _arima(df, vars_["value"], p),
    "Exponential Smoothing":       lambda df,vars_,**p: _expsmooth(df, vars_["value"], p),
    "Autocorrelation (ACF/PACF)":  lambda df,vars_,**p: _autocorr(df, vars_["value"], p),

    "K-Means Clustering":          lambda df,vars_,**p: _kmeans(df, vars_["variables"], p),
    "Hierarchical Clustering":     lambda df,vars_,**p: _hclust(df, vars_["variables"], p),
    "DBSCAN":                      lambda df,vars_,**p: _dbscan(df, vars_["variables"], p),

    "PCA":                         lambda df,vars_,**p: _pca(df, vars_["variables"], p),
    "Factor Analysis":             lambda df,vars_,**p: _factor_analysis(df, vars_["variables"], p),

    "Kaplan-Meier Survival":       lambda df,vars_,**p: _km(df, vars_["duration"], vars_["event"], p),
    "Cox Proportional Hazards":    lambda df,vars_,**p: _cox(df, vars_["duration"], vars_["event"], vars_["covariates"], p),

    "Decision Tree Classifier":    lambda df,vars_,**p: _ml_classifier("decision_tree")(df, vars_["y"], vars_["x"], p),
    "Random Forest Classifier":    lambda df,vars_,**p: _ml_classifier("random_forest")(df, vars_["y"], vars_["x"], p),
    "K-Nearest Neighbors":         lambda df,vars_,**p: _ml_classifier("knn")(df, vars_["y"], vars_["x"], p),
    "Naive Bayes Classifier":      lambda df,vars_,**p: _ml_classifier("naive_bayes")(df, vars_["y"], vars_["x"], p),
    "Linear Discriminant Analysis":lambda df,vars_,**p: _ml_classifier("lda")(df, vars_["y"], vars_["x"], p),

    "Cohen's d (Effect Size)":     lambda df,vars_,**p: _cohens_d(df, vars_["value"], vars_["group"], p),
    "Cramér's V":                  lambda df,vars_,**p: _cramers_v(df, vars_["var1"], vars_["var2"], p),
}


def run_analysis(tool: str, df, variables: dict, params: dict) -> dict:
    """Run a statistical tool and always return a dict (with ok=True/False)."""
    fn = DISPATCH.get(tool)
    if fn is None:
        return {"ok": False, "error": f"Tool '{tool}' is not registered."}
    try:
        result = fn(df, variables, **params)
        if result is None:
            return {"ok": False, "error": "Empty result"}
        result.setdefault("ok", True)
        result.setdefault("tool", tool)
        return result
    except Exception as e:
        return {"ok": False, "error": str(e), "tool": tool}


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

"""
Plotly-based graph builder.
Each function returns a plotly.graph_objects.Figure.
"""


# Modern dark theme template
TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#11131b",
        plot_bgcolor="#11131b",
        font=dict(family="Inter, sans-serif", color="#ECEEF6", size=12),
        xaxis=dict(gridcolor="#232636", zerolinecolor="#232636"),
        yaxis=dict(gridcolor="#232636", zerolinecolor="#232636"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=50, b=40),
        colorway=["#7C5CFF","#4FD1C5","#F687B3","#F6AD55","#34d399","#60a5fa","#a78bfa","#f472b6"],
    )
)


def _apply(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(template=TEMPLATE, title=dict(text=title, x=0.02, font=dict(size=14)))
    return fig


GRAPH_CATALOG = {
    "histogram":       ("Histogram",          "_hist"),
    "boxplot":         ("Box Plot",           "_box"),
    "violin":          ("Violin Plot",        "_violin"),
    "kde":             ("KDE / Density",      "_kde"),
    "qq":              ("Q-Q Plot",           "_qq"),
    "scatter":         ("Scatter Plot",       "_scatter"),
    "regression":      ("Regression Line",    "_regression"),
    "residual":        ("Residual Plot",      "_residual"),
    "bar":             ("Bar Chart",          "_bar"),
    "stacked_bar":     ("Stacked Bar",        "_stacked_bar"),
    "pie":             ("Pie Chart",          "_pie"),
    "line":            ("Line Chart",         "_line"),
    "heatmap":         ("Heatmap",            "_heatmap"),
    "pair_plot":       ("Pair Plot",          "_pair"),
    "confusion_matrix":("Confusion Matrix",   "_cm"),
    "roc":             ("ROC Curve",          "_roc"),
    "dendrogram":      ("Dendrogram",         "_dendrogram"),
}


def list_graphs_for_tool(tool_spec: dict) -> list[tuple[str,str]]:
    ids = tool_spec.get("graphs", [])
    return [(g, GRAPH_CATALOG[g][0]) for g in ids if g in GRAPH_CATALOG]


# ============================================================
# Builders
# ============================================================
def _hist(result):
    data = result["extras"].get("data")
    if data is None or len(data) == 0: return None
    num_cols = data.select_dtypes(include="number").columns.tolist()
    if not num_cols: return None
    figs = []
    for c in num_cols[:4]:
        fig = px.histogram(data, x=c, nbins=30, marginal="box", opacity=0.85)
        figs.append(_apply(fig, f"Histogram — {c}"))
    return figs[0] if len(figs)==1 else figs


def _box(result):
    ex = result["extras"]; data = ex.get("data")
    if data is None: return None
    if "value" in ex and "group" in ex:
        fig = px.box(data, x=ex["group"], y=ex["value"], points="outliers", color=ex["group"])
        return _apply(fig, f"Box Plot — {ex['value']} by {ex['group']}")
    num = data.select_dtypes(include="number")
    if num.empty: return None
    fig = px.box(num.melt(var_name="Variable", value_name="Value"), x="Variable", y="Value", color="Variable")
    return _apply(fig, "Box Plot")


def _violin(result):
    ex = result["extras"]; data = ex.get("data")
    if data is None: return None
    if "value" in ex and "group" in ex:
        fig = px.violin(data, x=ex["group"], y=ex["value"], box=True, points="outliers", color=ex["group"])
        return _apply(fig, f"Violin — {ex['value']} by {ex['group']}")
    num = data.select_dtypes(include="number")
    if num.empty: return None
    fig = px.violin(num.melt(var_name="Variable", value_name="Value"), x="Variable", y="Value", box=True, color="Variable")
    return _apply(fig, "Violin Plot")


def _kde(result):
    data = result["extras"].get("data")
    if data is None: return None
    num = data.select_dtypes(include="number")
    if num.empty: return None
    fig = go.Figure()
    for c in num.columns[:5]:
        s = num[c].dropna()
        if len(s) < 2: continue
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(s)
        x = np.linspace(s.min(), s.max(), 200)
        fig.add_trace(go.Scatter(x=x, y=kde(x), mode="lines", name=c, fill="tozeroy"))
    return _apply(fig, "Density (KDE)")


def _qq(result):
    from scipy import stats as sp
    data = result["extras"].get("data")
    if data is None: return None
    num = data.select_dtypes(include="number")
    if num.empty: return None
    col = num.columns[0]
    s = num[col].dropna()
    qq = sp.probplot(s, dist="norm")
    theor, ordered = qq[0]
    slope, intercept, _ = qq[1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theor, y=ordered, mode="markers", name="Sample"))
    fig.add_trace(go.Scatter(x=theor, y=slope*theor+intercept, mode="lines", name="Reference", line=dict(color="#4FD1C5")))
    return _apply(fig, f"Q-Q Plot — {col}")


def _scatter(result):
    ex = result["extras"]; data = ex.get("data")
    if data is None: return None
    if "x" in ex and "y" in ex:
        color = "Cluster" if "Cluster" in data.columns else None
        fig = px.scatter(data, x=ex["x"], y=ex["y"], color=color, opacity=0.85)
        return _apply(fig, f"Scatter — {ex['y']} vs {ex['x']}")
    num = data.select_dtypes(include="number")
    if num.shape[1] >= 2:
        color = "Cluster" if "Cluster" in data.columns else None
        fig = px.scatter(data, x=num.columns[0], y=num.columns[1], color=color, opacity=0.85)
        return _apply(fig, "Scatter")
    return None


def _regression(result):
    ex = result["extras"]; data = ex.get("data")
    if data is None: return None
    if "x" in ex and "y" in ex:
        fig = px.scatter(data, x=ex["x"], y=ex["y"], trendline="ols", opacity=0.85)
        return _apply(fig, f"Regression — {ex['y']} ~ {ex['x']}")
    if "pred" in ex and "y" in ex:
        d = pd.DataFrame({"Actual": ex["data"][ex["y"]].values, "Predicted": ex["pred"].values})
        fig = px.scatter(d, x="Actual", y="Predicted", trendline="ols")
        return _apply(fig, "Predicted vs Actual")
    return None


def _residual(result):
    ex = result["extras"]
    if "resid" not in ex or "pred" not in ex: return None
    d = pd.DataFrame({"Predicted": np.asarray(ex["pred"]), "Residual": np.asarray(ex["resid"])})
    fig = px.scatter(d, x="Predicted", y="Residual", opacity=0.85)
    fig.add_hline(y=0, line=dict(color="#4FD1C5", dash="dash"))
    return _apply(fig, "Residual Plot")


def _bar(result):
    for tname, tdf in result.get("tables", {}).items():
        if isinstance(tdf, pd.DataFrame) and tdf.shape[1] >= 2:
            x_col = tdf.columns[0]
            num_cols = tdf.select_dtypes(include="number").columns.tolist()
            if num_cols:
                fig = px.bar(tdf, x=x_col, y=num_cols[0], color=x_col)
                return _apply(fig, tname)
    return None


def _stacked_bar(result):
    ex = result["extras"]
    if "matrix" in ex:
        m = ex["matrix"]
        long = m.reset_index().melt(id_vars=m.index.name or m.reset_index().columns[0], var_name="Col", value_name="Count")
        x_col = long.columns[0]
        fig = px.bar(long, x=x_col, y="Count", color="Col", barmode="stack")
        return _apply(fig, "Stacked Bar")
    return None


def _pie(result):
    for _, tdf in result.get("tables", {}).items():
        if isinstance(tdf, pd.DataFrame) and tdf.shape[1] >= 2:
            num = tdf.select_dtypes(include="number").columns.tolist()
            if num:
                fig = px.pie(tdf, names=tdf.columns[0], values=num[0], hole=0.45)
                return _apply(fig, "Pie Chart")
    return None


def _line(result):
    ex = result["extras"]
    if "history" in ex and "forecast" in ex:
        hist = pd.DataFrame({"t": range(len(ex["history"])), "Value": ex["history"].values, "Type":"History"})
        fc = pd.DataFrame({"t": range(len(ex["history"]), len(ex["history"])+len(ex["forecast"])),
                            "Value": ex["forecast"].values, "Type":"Forecast"})
        d = pd.concat([hist, fc], ignore_index=True)
        fig = px.line(d, x="t", y="Value", color="Type")
        return _apply(fig, "Forecast")
    if "data" in ex and isinstance(ex["data"], pd.DataFrame):
        df = ex["data"]
        if {"Trend","Seasonal","Residual","Observed"}.issubset(df.columns):
            time_col = ex.get("time", df.columns[0])
            fig = go.Figure()
            for c in ["Observed","Trend","Seasonal","Residual"]:
                fig.add_trace(go.Scatter(x=df[time_col], y=df[c], mode="lines", name=c))
            return _apply(fig, "Time Series Decomposition")
    if "time" in ex and "surv" in ex:
        fig = go.Figure(go.Scatter(x=ex["time"], y=ex["surv"], mode="lines+markers", line_shape="hv"))
        return _apply(fig, "Kaplan-Meier Survival")
    data = ex.get("data")
    if isinstance(data, pd.DataFrame):
        num = data.select_dtypes(include="number")
        if num.shape[1] >= 1:
            d = num.reset_index().melt(id_vars="index", var_name="Variable", value_name="Value")
            fig = px.line(d, x="index", y="Value", color="Variable")
            return _apply(fig, "Line Chart")
    return None


def _heatmap(result):
    ex = result["extras"]
    if "matrix" in ex:
        m = ex["matrix"]
        if isinstance(m, pd.DataFrame):
            fig = px.imshow(m, text_auto=True, color_continuous_scale="Viridis", aspect="auto")
            return _apply(fig, "Heatmap")
    return None


def _pair(result):
    data = result["extras"].get("data")
    if data is None: return None
    num = data.select_dtypes(include="number")
    if num.shape[1] < 2: return None
    cols = num.columns.tolist()[:5]
    color = "Cluster" if "Cluster" in data.columns else None
    fig = px.scatter_matrix(data, dimensions=cols, color=color, opacity=0.7)
    fig.update_traces(diagonal_visible=False)
    return _apply(fig, "Pair Plot")


def _cm(result):
    ex = result["extras"]
    if "cm" not in ex or "levels" not in ex: return None
    cm = ex["cm"]; levels = [str(l) for l in ex["levels"]]
    fig = px.imshow(cm, x=[f"Pred {l}" for l in levels], y=[f"Actual {l}" for l in levels],
                     text_auto=True, color_continuous_scale="Viridis", aspect="auto")
    return _apply(fig, "Confusion Matrix")


def _roc(result):
    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        return None
    ex = result["extras"]
    if "y_true" not in ex or "probs" not in ex: return None
    fpr, tpr, _ = roc_curve(ex["y_true"], ex["probs"])
    a = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {a:.3f}", line=dict(color="#7C5CFF", width=3)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random", line=dict(dash="dash", color="#6b6f82")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return _apply(fig, "ROC Curve")


def _dendrogram(result):
    try:
        from scipy.cluster.hierarchy import dendrogram
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import io, base64
    except ImportError:
        return None
    ex = result["extras"]
    if "linkage" not in ex: return None
    fig, ax = plt.subplots(figsize=(9,4), facecolor="#11131b")
    ax.set_facecolor("#11131b")
    dendrogram(ex["linkage"], ax=ax, color_threshold=0)
    ax.tick_params(colors="#ECEEF6"); ax.spines[:].set_color("#232636")
    ax.set_title("Dendrogram", color="#ECEEF6")
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#11131b"); plt.close(fig)
    img = base64.b64encode(buf.getvalue()).decode()
    return {"_img": f"data:image/png;base64,{img}"}


_BUILDERS = {
    "_hist": _hist, "_box": _box, "_violin": _violin, "_kde": _kde, "_qq": _qq,
    "_scatter": _scatter, "_regression": _regression, "_residual": _residual,
    "_bar": _bar, "_stacked_bar": _stacked_bar, "_pie": _pie, "_line": _line,
    "_heatmap": _heatmap, "_pair": _pair, "_cm": _cm, "_roc": _roc,
    "_dendrogram": _dendrogram,
}


def build_graph(graph_id: str, result: dict):
    if graph_id not in GRAPH_CATALOG: return None
    fn_name = GRAPH_CATALOG[graph_id][1]
    return _BUILDERS[fn_name](result)
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


def _validate(analysis: dict, buckets: dict) -> tuple[bool, list[str]]:
    """Validate role contracts and parameters. Return (ok, errors)."""
    errors: list[str] = []
    if not analysis.get("title", "").strip():
        errors.append("Provide a title.")
    if not analysis.get("tool"):
        errors.append("Select a statistical tool.")
        return False, errors

    spec = get_tool(analysis["tool"])
    for role_name, role_def in spec["roles"].items():
        sel = analysis["variables"].get(role_name)
        allowed_cols = []
        for t in role_def["types"]:
            allowed_cols += buckets.get(t, [])
        if not sel:
            if not role_def.get("optional"):
                errors.append(f"Select **{role_def.get('label') or role_name}**.")
            continue
        # Coerce to list
        sel_list = sel if isinstance(sel, list) else [sel]
        # Cardinality
        if role_def.get("multi"):
            if len(sel_list) < role_def.get("min", 1):
                errors.append(f"`{role_name}` needs at least {role_def.get('min',1)} variable(s).")
            if role_def.get("max") and len(sel_list) > role_def["max"]:
                errors.append(f"`{role_name}` allows at most {role_def['max']} variable(s).")
        # Type compliance
        for c in sel_list:
            if c not in allowed_cols:
                errors.append(f"`{c}` is not a valid type for **{role_def.get('label') or role_name}**.")
    return len(errors) == 0, errors


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



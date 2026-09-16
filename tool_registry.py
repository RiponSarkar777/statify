"""
Statify — Statistical Tool Registry
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Statistical-tool registry.

Each entry declares:
  - category
  - description
  - variable role contracts (which roles, which semantic types are accepted, cardinality)
  - parameters (with defaults & types)
  - assumptions (for reporting)
  - recommended graphs (for Page 4 graph picker)

Moved out of app.py — behavior unchanged.
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


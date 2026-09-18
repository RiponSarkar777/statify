"""
Statify — Dynamic Interpretation Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates plain-language explanations from the actual numbers in a
tool's result dict (from stats_engine.py). This is rule-based /
templated text generation — no AI or external API calls anywhere
in this file. Pure Python: no pandas, numpy, or Streamlit.

interpret_result() reads a result dict and returns a structured
interpretation (headline, explanation, statistical detail,
assumptions, conclusion). interpret_graph() returns a short
explanatory caption for a given graph type, from the internal
GRAPH_INTERP lookup.

Moved out of app.py — behavior unchanged. No wording altered.

Design note for later AI-integration phases: this file is the
grounding / fallback reference for the future AI interpreter, not
something to be replaced by it. Per the original project plan, the
AI interpreter should sit alongside interpret_result() as the
primary path, with this module remaining available as a verified,
deterministic fallback — not deleted.
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


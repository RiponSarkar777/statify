"""
Statify — Deterministic Statistical Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All deterministic statistical calculations. Every function here is a
pure function: dataframe + parameters in, a structured result dict
out. No Streamlit calls anywhere in this file.

Part 1 — Descriptive Statistics & Hypothesis Testing:
_descriptive_stats, _frequency_table, _crosstab, _skew_kurt,
_outliers_iqr, _one_sample_t, _independent_t, _paired_t, _z_test,
_proportion_z, _chi2_independence, _chi2_gof, _fisher_exact.

Part 2 — Correlation, Regression, ANOVA & Non-Parametric Tests:
_correlation, _correlation_matrix, _point_biserial, _simple_linear,
_multiple_linear, _polynomial, _logistic, _ridge_lasso,
_one_way_anova, _two_way_anova, _rm_anova, _ancova, _tukey,
_mann_whitney, _wilcoxon, _kruskal, _friedman, _shapiro, _levene, _ks.

Part 3 — Time Series, Multivariate, ML, Effect Sizes & Orchestration:
_decompose, _adf, _arima, _expsmooth, _autocorr, _kmeans, _hclust,
_dbscan, _pca, _factor_analysis, _km, _cox, _ml_classifier,
_cohens_d, _interp_d, _cramers_v, DISPATCH, run_analysis.

Moved out of app.py — behavior unchanged. No calculations altered.

Notes:
- Several functions do their own local imports of statsmodels,
  sklearn, or lifelines inside the function body (unchanged from
  the original) — this is intentional lazy-loading, not an omission.
- _interp_d here is a small internal helper used only by _cohens_d
  to label effect-size magnitude. It is unrelated to interpret_result()
  and interpret_graph(), which will live in interpreter.py (a later,
  separate extraction step) — the name is coincidentally similar but
  the two are independent and this one stays here.
- DISPATCH and run_analysis are the orchestration layer that ties
  every tool name (as used by tool_registry.py) to its calculation
  function. This is what app.py will call going forward.
"""
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ============================================================
# DESCRIPTIVE
# ============================================================
def _descriptive_stats(df, vars_, p):
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    desc = sub.describe().T
    desc["skew"]     = sub.skew()
    desc["kurtosis"] = sub.kurtosis()
    desc["missing"]  = df[vars_].isna().sum().values
    return {
        "summary": {"n_vars": len(vars_)},
        "tables": {"Descriptive Statistics": desc.round(4).reset_index().rename(columns={"index":"Variable"})},
        "extras": {"data": sub},
        "p_value": None, "alpha": float(p.get("alpha", 0.05)),
    }


def _frequency_table(df, var, p):
    s = df[var].dropna()
    freq = s.value_counts().rename("Frequency").to_frame()
    freq["Percent"] = (freq["Frequency"] / freq["Frequency"].sum() * 100).round(2)
    freq = freq.reset_index().rename(columns={"index": var})
    return {
        "summary": {"n": int(s.shape[0]), "n_categories": int(s.nunique())},
        "tables": {"Frequency Table": freq},
        "extras": {"data": s},
        "p_value": None, "alpha": float(p.get("alpha", 0.05)),
    }


def _crosstab(df, row, col, p):
    ct = pd.crosstab(df[row], df[col])
    return {
        "summary": {"rows": int(ct.shape[0]), "cols": int(ct.shape[1])},
        "tables": {"Cross Tabulation": ct.reset_index()},
        "extras": {"matrix": ct, "data": df[[row, col]].dropna()},
        "p_value": None, "alpha": float(p.get("alpha", 0.05)),
    }


def _skew_kurt(df, vars_, p):
    sub = df[vars_].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame({"Skewness": sub.skew(), "Kurtosis": sub.kurtosis()}).round(4).reset_index().rename(columns={"index":"Variable"})
    return {
        "summary": {"n_vars": len(vars_)},
        "tables": {"Skewness & Kurtosis": out},
        "extras": {"data": sub},
        "p_value": None, "alpha": float(p.get("alpha", 0.05)),
    }


def _outliers_iqr(df, var, p):
    s = pd.to_numeric(df[var], errors="coerce").dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    out = s[(s < lo) | (s > hi)]
    tbl = pd.DataFrame({var: out}).reset_index(drop=True)
    return {
        "summary": {"n_outliers": int(out.shape[0]), "lower_bound": round(float(lo),4), "upper_bound": round(float(hi),4)},
        "tables": {"Outliers (IQR method)": tbl},
        "extras": {"data": s, "bounds": (lo, hi)},
        "p_value": None, "alpha": float(p.get("alpha", 0.05)),
    }


# ============================================================
# HYPOTHESIS TESTS
# ============================================================
def _one_sample_t(df, var, p):
    s = pd.to_numeric(df[var], errors="coerce").dropna()
    test_value = float(p.get("test_value", 0))
    t, pv = sp_stats.ttest_1samp(s, test_value)
    return {
        "summary": {"n": int(s.shape[0]), "mean": round(float(s.mean()),4), "test_value": test_value},
        "tables": {"One-Sample t-Test": pd.DataFrame({
            "Metric": ["t-statistic","df","p-value","Mean","Std Dev"],
            "Value":  [round(float(t),4), int(s.shape[0]-1), round(float(pv),6), round(float(s.mean()),4), round(float(s.std()),4)]})},
        "extras": {"data": s},
        "p_value": float(pv), "alpha": float(p.get("alpha", 0.05)),
    }


def _independent_t(df, value, group, p):
    sub = df[[value, group]].dropna()
    groups = sub[group].unique()
    if len(groups) != 2:
        raise ValueError("Independent t-test requires exactly 2 groups.")
    g1 = pd.to_numeric(sub[sub[group]==groups[0]][value], errors="coerce").dropna()
    g2 = pd.to_numeric(sub[sub[group]==groups[1]][value], errors="coerce").dropna()
    t, pv = sp_stats.ttest_ind(g1, g2, equal_var=p.get("equal_var", True))
    return {
        "summary": {"group1": str(groups[0]), "group2": str(groups[1]), "n1": int(g1.shape[0]), "n2": int(g2.shape[0])},
        "tables": {"Independent Samples t-Test": pd.DataFrame({
            "Metric": ["t-statistic","p-value",f"Mean ({groups[0]})",f"Mean ({groups[1]})"],
            "Value":  [round(float(t),4), round(float(pv),6), round(float(g1.mean()),4), round(float(g2.mean()),4)]})},
        "extras": {"data": sub, "g1": g1, "g2": g2, "groups": groups},
        "p_value": float(pv), "alpha": float(p.get("alpha", 0.05)),
    }


def _paired_t(df, v1, v2, p):
    sub = df[[v1, v2]].dropna()
    a = pd.to_numeric(sub[v1], errors="coerce")
    b = pd.to_numeric(sub[v2], errors="coerce")
    t, pv = sp_stats.ttest_rel(a, b)
    return {
        "summary": {"n": int(sub.shape[0]), "mean_diff": round(float((a-b).mean()),4)},
        "tables": {"Paired Samples t-Test": pd.DataFrame({
            "Metric": ["t-statistic","p-value","Mean Difference"],
            "Value":  [round(float(t),4), round(float(pv),6), round(float((a-b).mean()),4)]})},
        "extras": {"data": sub, "diff": a-b},
        "p_value": float(pv), "alpha": float(p.get("alpha", 0.05)),
    }


def _z_test(df, var, p):
    s = pd.to_numeric(df[var], errors="coerce").dropna()
    pop_mean = float(p.get("pop_mean", 0))
    pop_std = float(p.get("pop_std", s.std()))
    n = s.shape[0]
    z = (s.mean() - pop_mean) / (pop_std / np.sqrt(n))
    pv = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return {
        "summary": {"n": int(n), "mean": round(float(s.mean()),4), "pop_mean": pop_mean},
        "tables": {"One-Sample Z-Test": pd.DataFrame({
            "Metric": ["z-statistic","p-value","Sample Mean","Population Mean"],
            "Value":  [round(float(z),4), round(float(pv),6), round(float(s.mean()),4), pop_mean]})},
        "extras": {"data": s},
        "p_value": float(pv), "alpha": float(p.get("alpha", 0.05)),
    }


def _proportion_z(df, var, p):
    s = df[var].dropna()
    success_val = p.get("success_value")
    n = s.shape[0]
    x = int((s == success_val).sum())
    p0 = float(p.get("hypothesized_prop", 0.5))
    phat = x / n
    se = np.sqrt(p0*(1-p0)/n)
    z = (phat - p0) / se
    pv = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return {
        "summary": {"n": int(n), "successes": x, "phat": round(float(phat),4)},
        "tables": {"One-Sample Proportion Z-Test": pd.DataFrame({
            "Metric": ["z-statistic","p-value","Sample Proportion","Hypothesized Proportion"],
            "Value":  [round(float(z),4), round(float(pv),6), round(float(phat),4), p0]})},
        "extras": {"data": s},
        "p_value": float(pv), "alpha": float(p.get("alpha", 0.05)),
    }


def _chi2_independence(df, v1, v2, p):
    ct = pd.crosstab(df[v1], df[v2])
    chi2, pv, dof, expected = sp_stats.chi2_contingency(ct)
    return {
        "summary": {"chi2": round(float(chi2),4), "dof": int(dof)},
        "tables": {"Chi-Square Test of Independence": pd.DataFrame({
            "Metric": ["Chi-Square","Degrees of Freedom","p-value"],
            "Value":  [round(float(chi2),4), int(dof), round(float(pv),6)]}),
            "Contingency Table": ct.reset_index()},
        "extras": {"matrix": ct, "expected": expected, "data": df[[v1,v2]].dropna()},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _chi2_gof(df, var, p):
    s = df[var].dropna()
    obs = s.value_counts()
    chi2, pv = sp_stats.chisquare(obs.values)
    return {
        "summary": {"chi2": round(float(chi2),4), "categories": int(obs.shape[0])},
        "tables": {"Chi-Square Goodness of Fit": pd.DataFrame({
            "Metric": ["Chi-Square","p-value"],
            "Value":  [round(float(chi2),4), round(float(pv),6)]}),
            "Observed Frequencies": obs.reset_index().rename(columns={"index":var, var:"Count"})},
        "extras": {"data": s},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _fisher_exact(df, v1, v2, p):
    ct = pd.crosstab(df[v1], df[v2])
    if ct.shape != (2,2):
        raise ValueError("Fisher's Exact Test requires a 2x2 table.")
    odds, pv = sp_stats.fisher_exact(ct.values)
    return {
        "summary": {"odds_ratio": round(float(odds),4)},
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
        a, b = sub[x], sub[y]
        if method == "pearson":
            r, pv = sp_stats.pearsonr(a, b)
        elif method == "spearman":
            r, pv = sp_stats.spearmanr(a, b)
        elif method == "kendall":
            r, pv = sp_stats.kendalltau(a, b)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        return {
            "summary": {"r": round(float(r),4), "n": int(sub.shape[0]), "method": method},
            "tables": {f"{method.title()} Correlation": pd.DataFrame({
                "Metric": ["r","p-value","n"],
                "Value":  [round(float(r),4), round(float(pv),6), int(sub.shape[0])]})},
            "extras": {"data": sub, "x": x, "y": y},
            "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
        }
    return _run


def _correlation_matrix(df, vars_, p):
    sub = df[vars_].apply(pd.to_numeric, errors="coerce")
    method = p.get("method", "pearson")
    corr = sub.corr(method=method)
    return {
        "summary": {"n_vars": len(vars_), "method": method},
        "tables": {"Correlation Matrix": corr.round(4).reset_index().rename(columns={"index":"Variable"})},
        "extras": {"matrix": corr, "data": sub},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _point_biserial(df, binary, num, p):
    sub = df[[binary, num]].dropna()
    b = pd.factorize(sub[binary])[0]
    n = pd.to_numeric(sub[num], errors="coerce")
    r, pv = sp_stats.pointbiserialr(b, n)
    return {
        "summary": {"r": round(float(r),4), "n": int(sub.shape[0])},
        "tables": {"Point-Biserial Correlation": pd.DataFrame({
            "Metric": ["r","p-value"], "Value": [round(float(r),4), round(float(pv),6)]})},
        "extras": {"data": sub},
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
        "summary": {"slope": round(float(slope),4), "intercept": round(float(intercept),4), "r2": round(float(r**2),4)},
        "tables": {"Simple Linear Regression": pd.DataFrame({
            "Metric": ["Slope","Intercept","R","R²","p-value","Std Error"],
            "Value":  [round(float(slope),4), round(float(intercept),4), round(float(r),4),
                       round(float(r**2),4), round(float(pv),6), round(float(se),4)]})},
        "extras": {"data": sub, "x": x, "y": y, "pred": pred, "resid": resid},
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
    degree = int(p.get("degree", 2))
    coeffs = np.polyfit(sub[x], sub[y], degree)
    pred = np.polyval(coeffs, sub[x])
    ss_res = np.sum((sub[y] - pred)**2)
    ss_tot = np.sum((sub[y] - sub[y].mean())**2)
    r2 = 1 - ss_res/ss_tot
    return {
        "summary": {"degree": degree, "r2": round(float(r2),4)},
        "tables": {"Polynomial Regression": pd.DataFrame({
            "Coefficient (highest degree first)": coeffs}).round(4)},
        "extras": {"data": sub, "x": x, "y": y, "pred": pred, "coeffs": coeffs},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _logistic(df, y, xs, p):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
    except ImportError:
        return {"ok": False, "error": "scikit-learn required"}
    sub = df[[y] + xs].dropna()
    yb = pd.factorize(sub[y])[0]
    X = sub[xs].apply(pd.to_numeric, errors="coerce")
    model = LogisticRegression(max_iter=1000).fit(X, yb)
    pred = model.predict(X)
    proba = model.predict_proba(X)[:,1]
    acc = accuracy_score(yb, pred)
    try:
        auc = roc_auc_score(yb, proba)
    except Exception:
        auc = None
    coef_tbl = pd.DataFrame({"Variable": xs, "Coefficient": model.coef_[0]}).round(4)
    return {
        "summary": {"accuracy": round(float(acc),4), "auc": round(float(auc),4) if auc else None},
        "tables": {"Coefficients": coef_tbl,
                   "Model Performance": pd.DataFrame({
                       "Metric": ["Accuracy","AUC"], "Value": [acc, auc]}).round(4)},
        "extras": {"data": sub, "y_true": yb, "y_pred": pred, "y_proba": proba, "model": model},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _ridge_lasso(kind):
    def _run(df, y, xs, p):
        try:
            from sklearn.linear_model import Ridge, Lasso
        except ImportError:
            return {"ok": False, "error": "scikit-learn required"}
        sub = df[[y] + xs].apply(pd.to_numeric, errors="coerce").dropna()
        alpha = float(p.get("alpha_reg", 1.0))
        Model = Ridge if kind == "ridge" else Lasso
        model = Model(alpha=alpha).fit(sub[xs], sub[y])
        pred = model.predict(sub[xs])
        resid = sub[y] - pred
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((sub[y]-sub[y].mean())**2)
        r2 = 1 - ss_res/ss_tot
        coef_tbl = pd.DataFrame({"Variable": xs, "Coefficient": model.coef_}).round(4)
        return {
            "summary": {"r2": round(float(r2),4), "alpha_reg": alpha},
            "tables": {"Coefficients": coef_tbl},
            "extras": {"data": sub, "y": y, "xs": xs, "pred": pred, "resid": resid},
            "p_value": None, "alpha": float(p.get("alpha",0.05)),
        }
    return _run
# ============================================================
# ANOVA
# ============================================================
def _one_way_anova(df, value, group, p):
    sub = df[[value, group]].dropna()
    groups = [pd.to_numeric(sub[sub[group]==g][value], errors="coerce").dropna() for g in sub[group].unique()]
    f, pv = sp_stats.f_oneway(*groups)
    return {
        "summary": {"f": round(float(f),4), "k_groups": len(groups)},
        "tables": {"One-Way ANOVA": pd.DataFrame({
            "Metric": ["F-statistic","p-value"], "Value": [round(float(f),4), round(float(pv),6)]})},
        "extras": {"data": sub, "groups": sub[group].unique()},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _two_way_anova(df, value, f1, f2, p):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    sub = df[[value, f1, f2]].dropna()
    formula = f'{value} ~ C({f1}) + C({f2}) + C({f1}):C({f2})'
    model = ols(formula, data=sub).fit()
    table = sm.stats.anova_lm(model, typ=2).round(4).reset_index().rename(columns={"index":"Source"})
    return {
        "summary": {"n": int(sub.shape[0])},
        "tables": {"Two-Way ANOVA": table},
        "extras": {"data": sub, "model": model},
        "p_value": float(table["PR(>F)"].iloc[0]) if "PR(>F)" in table else None,
        "alpha": float(p.get("alpha",0.05)),
    }


def _rm_anova(df, vars_, p):
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    f, pv = sp_stats.f_oneway(*[sub[c] for c in vars_])
    return {
        "summary": {"f": round(float(f),4), "k": len(vars_)},
        "tables": {"Repeated Measures ANOVA (approx)": pd.DataFrame({
            "Metric": ["F-statistic","p-value"], "Value": [round(float(f),4), round(float(pv),6)]})},
        "extras": {"data": sub},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _ancova(df, value, group, cov, p):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    sub = df[[value, group, cov]].dropna()
    formula = f'{value} ~ C({group}) + {cov}'
    model = ols(formula, data=sub).fit()
    table = sm.stats.anova_lm(model, typ=2).round(4).reset_index().rename(columns={"index":"Source"})
    return {
        "summary": {"n": int(sub.shape[0])},
        "tables": {"ANCOVA": table},
        "extras": {"data": sub, "model": model},
        "p_value": float(table["PR(>F)"].iloc[0]) if "PR(>F)" in table else None,
        "alpha": float(p.get("alpha",0.05)),
    }


def _tukey(df, value, group, p):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    sub = df[[value, group]].dropna()
    res = pairwise_tukeyhsd(pd.to_numeric(sub[value], errors="coerce"), sub[group],
                             alpha=float(p.get("alpha",0.05)))
    tbl = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
    return {
        "summary": {"n_comparisons": int(tbl.shape[0])},
        "tables": {"Tukey HSD Post-Hoc": tbl},
        "extras": {"data": sub},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }
# ============================================================
# NON-PARAMETRIC
# ============================================================
def _mann_whitney(df, value, group, p):
    sub = df[[value, group]].dropna()
    groups = sub[group].unique()
    g1 = pd.to_numeric(sub[sub[group]==groups[0]][value], errors="coerce").dropna()
    g2 = pd.to_numeric(sub[sub[group]==groups[1]][value], errors="coerce").dropna()
    u, pv = sp_stats.mannwhitneyu(g1, g2)
    return {
        "summary": {"u": round(float(u),4)},
        "tables": {"Mann-Whitney U Test": pd.DataFrame({
            "Metric": ["U-statistic","p-value"], "Value": [round(float(u),4), round(float(pv),6)]})},
        "extras": {"data": sub, "g1": g1, "g2": g2, "groups": groups},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _wilcoxon(df, v1, v2, p):
    sub = df[[v1, v2]].dropna()
    stat, pv = sp_stats.wilcoxon(sub[v1], sub[v2])
    return {
        "summary": {"statistic": round(float(stat),4)},
        "tables": {"Wilcoxon Signed-Rank Test": pd.DataFrame({
            "Metric": ["Statistic","p-value"], "Value": [round(float(stat),4), round(float(pv),6)]})},
        "extras": {"data": sub},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _kruskal(df, value, group, p):
    sub = df[[value, group]].dropna()
    groups = [pd.to_numeric(sub[sub[group]==g][value], errors="coerce").dropna() for g in sub[group].unique()]
    h, pv = sp_stats.kruskal(*groups)
    return {
        "summary": {"h": round(float(h),4), "k_groups": len(groups)},
        "tables": {"Kruskal-Wallis H Test": pd.DataFrame({
            "Metric": ["H-statistic","p-value"], "Value": [round(float(h),4), round(float(pv),6)]})},
        "extras": {"data": sub},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _friedman(df, vars_, p):
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    stat, pv = sp_stats.friedmanchisquare(*[sub[c] for c in vars_])
    return {
        "summary": {"statistic": round(float(stat),4), "k": len(vars_)},
        "tables": {"Friedman Test": pd.DataFrame({
            "Metric": ["Statistic","p-value"], "Value": [round(float(stat),4), round(float(pv),6)]})},
        "extras": {"data": sub},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _shapiro(df, var, p):
    s = pd.to_numeric(df[var], errors="coerce").dropna()
    stat, pv = sp_stats.shapiro(s)
    return {
        "summary": {"statistic": round(float(stat),4), "n": int(s.shape[0])},
        "tables": {"Shapiro-Wilk Normality Test": pd.DataFrame({
            "Metric": ["W-statistic","p-value"], "Value": [round(float(stat),4), round(float(pv),6)]})},
        "extras": {"data": s},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }


def _levene(df, value, group, p):
    sub = df[[value, group]].dropna()
    groups = [pd.to_numeric(sub[sub[group]==g][value], errors="coerce").dropna() for g in sub[group].unique()]
    stat, pv = sp_stats.levene(*groups)
    return {
        "summary": {"statistic": round(float(stat),4)},
        "tables": {"Levene's Test for Equal Variances": pd.DataFrame({
            "Metric": ["Statistic","p-value"], "Value": [round(float(stat),4), round(float(pv),6)]})},
        "extras": {"data": sub},
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
    sub = df[[time, value]].dropna().sort_values(time)
    sub[time] = pd.to_datetime(sub[time], errors="coerce")
    sub = sub.set_index(time)
    period = int(p.get("period", 12))
    result = seasonal_decompose(sub[value], model=p.get("model","additive"), period=period, extrapolate_trend="freq")
    out = pd.DataFrame({
        "Trend": result.trend, "Seasonal": result.seasonal, "Residual": result.resid, "Observed": result.observed
    }).reset_index()
    return {
        "summary": {"period": period, "model": p.get("model","additive")},
        "tables": {"Decomposition": out.round(4)},
        "extras": {"data": sub, "result": result},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _adf(df, value, p):
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    s = pd.to_numeric(df[value], errors="coerce").dropna()
    result = adfuller(s)
    return {
        "summary": {"adf_stat": round(float(result[0]),4), "p_value": round(float(result[1]),6)},
        "tables": {"Augmented Dickey-Fuller Test": pd.DataFrame({
            "Metric": ["ADF Statistic","p-value","Lags Used","Observations"],
            "Value":  [round(float(result[0]),4), round(float(result[1]),6), result[2], result[3]]})},
        "extras": {"data": s},
        "p_value": float(result[1]), "alpha": float(p.get("alpha",0.05)),
    }


def _arima(df, value, p):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    s = pd.to_numeric(df[value], errors="coerce").dropna()
    order = tuple(p.get("order", (1,1,1)))
    model = ARIMA(s, order=order).fit()
    steps = int(p.get("forecast_steps", 5))
    fc = model.forecast(steps=steps)
    return {
        "summary": {"order": order, "aic": round(float(model.aic),4)},
        "tables": {"Forecast": pd.DataFrame({"Step": range(1, steps+1), "Forecast": fc.values}).round(4)},
        "extras": {"data": s, "model": model, "forecast": fc},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _expsmooth(df, value, p):
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    s = pd.to_numeric(df[value], errors="coerce").dropna()
    model = ExponentialSmoothing(s, trend=p.get("trend","add"), seasonal=p.get("seasonal"), seasonal_periods=p.get("seasonal_periods")).fit()
    steps = int(p.get("forecast_steps", 5))
    fc = model.forecast(steps)
    return {
        "summary": {"aic": round(float(model.aic),4) if hasattr(model,"aic") else None},
        "tables": {"Forecast": pd.DataFrame({"Step": range(1, steps+1), "Forecast": fc.values}).round(4)},
        "extras": {"data": s, "model": model, "forecast": fc},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _autocorr(df, value, p):
    try:
        from statsmodels.tsa.stattools import acf, pacf
    except ImportError:
        return {"ok": False, "error": "statsmodels required"}
    s = pd.to_numeric(df[value], errors="coerce").dropna()
    nlags = int(p.get("nlags", 20))
    acf_vals = acf(s, nlags=nlags)
    pacf_vals = pacf(s, nlags=nlags)
    out = pd.DataFrame({"Lag": range(len(acf_vals)), "ACF": acf_vals, "PACF": pacf_vals})
    return {
        "summary": {"nlags": nlags},
        "tables": {"Autocorrelation": out.round(4)},
        "extras": {"data": s},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }
# ============================================================
# MULTIVARIATE
# ============================================================
def _kmeans(df, vars_, p):
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"ok": False, "error": "scikit-learn required"}
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    X = StandardScaler().fit_transform(sub)
    k = int(p.get("n_clusters", 3))
    model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sub = sub.copy()
    sub["Cluster"] = model.labels_
    return {
        "summary": {"k": k, "inertia": round(float(model.inertia_),4)},
        "tables": {"Cluster Assignments": sub.reset_index(drop=True)},
        "extras": {"data": sub, "model": model, "X": X},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _hclust(df, vars_, p):
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"ok": False, "error": "scipy/scikit-learn required"}
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    X = StandardScaler().fit_transform(sub)
    Z = linkage(X, method=p.get("method","ward"))
    k = int(p.get("n_clusters", 3))
    labels = fcluster(Z, k, criterion="maxclust")
    sub = sub.copy()
    sub["Cluster"] = labels
    return {
        "summary": {"k": k, "method": p.get("method","ward")},
        "tables": {"Cluster Assignments": sub.reset_index(drop=True)},
        "extras": {"data": sub, "linkage": Z, "X": X},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _dbscan(df, vars_, p):
    try:
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"ok": False, "error": "scikit-learn required"}
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    X = StandardScaler().fit_transform(sub)
    eps = float(p.get("eps", 0.5))
    min_samples = int(p.get("min_samples", 5))
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    sub = sub.copy()
    sub["Cluster"] = model.labels_
    n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    return {
        "summary": {"n_clusters": n_clusters, "n_noise": int((model.labels_==-1).sum())},
        "tables": {"Cluster Assignments": sub.reset_index(drop=True)},
        "extras": {"data": sub, "model": model, "X": X},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _pca(df, vars_, p):
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"ok": False, "error": "scikit-learn required"}
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    X = StandardScaler().fit_transform(sub)
    n_components = int(p.get("n_components", min(len(vars_), 2)))
    model = PCA(n_components=n_components).fit(X)
    scores = model.transform(X)
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_components)])
    var_tbl = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(n_components)],
        "Explained Variance Ratio": model.explained_variance_ratio_,
    }).round(4)
    loadings = pd.DataFrame(model.components_.T, index=vars_, columns=[f"PC{i+1}" for i in range(n_components)]).round(4).reset_index().rename(columns={"index":"Variable"})
    return {
        "summary": {"n_components": n_components, "total_variance_explained": round(float(model.explained_variance_ratio_.sum()),4)},
        "tables": {"Explained Variance": var_tbl, "Loadings": loadings},
        "extras": {"data": sub, "scores": scores_df, "model": model, "X": X},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _factor_analysis(df, vars_, p):
    try:
        from sklearn.decomposition import FactorAnalysis
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"ok": False, "error": "scikit-learn required"}
    sub = df[vars_].apply(pd.to_numeric, errors="coerce").dropna()
    X = StandardScaler().fit_transform(sub)
    n_factors = int(p.get("n_factors", 2))
    model = FactorAnalysis(n_components=n_factors, random_state=42).fit(X)
    loadings = pd.DataFrame(model.components_.T, index=vars_, columns=[f"Factor{i+1}" for i in range(n_factors)]).round(4).reset_index().rename(columns={"index":"Variable"})
    return {
        "summary": {"n_factors": n_factors},
        "tables": {"Factor Loadings": loadings},
        "extras": {"data": sub, "model": model, "X": X},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }
# ============================================================
# SURVIVAL ANALYSIS
# ============================================================
def _km(df, duration, event, p):
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return {"ok": False, "error": "lifelines required"}
    sub = df[[duration, event]].dropna()
    kmf = KaplanMeierFitter()
    kmf.fit(sub[duration], sub[event])
    surv = kmf.survival_function_.reset_index()
    surv.columns = ["Timeline", "Survival Probability"]
    return {
        "summary": {"n": int(sub.shape[0]), "events": int(sub[event].sum()), "median_survival": kmf.median_survival_time_},
        "tables": {"Survival Function": surv.round(4)},
        "extras": {"data": sub, "kmf": kmf},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _cox(df, duration, event, covariates, p):
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        return {"ok": False, "error": "lifelines required"}
    sub = df[[duration, event] + covariates].dropna()
    cph = CoxPHFitter()
    cph.fit(sub, duration_col=duration, event_col=event)
    summary = cph.summary.reset_index().rename(columns={"index":"Covariate", "covariate":"Covariate"})
    return {
        "summary": {"n": int(sub.shape[0]), "concordance": round(float(cph.concordance_index_),4)},
        "tables": {"Cox Proportional Hazards": summary.round(4)},
        "extras": {"data": sub, "model": cph},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }
# ============================================================
# MACHINE LEARNING CLASSIFIERS
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
        sub = df[[y] + xs].dropna()
        yb = pd.factorize(sub[y])[0]
        X = sub[xs].apply(pd.to_numeric, errors="coerce")
        X_train, X_test, y_train, y_test = train_test_split(X, yb, test_size=float(p.get("test_size",0.2)), random_state=42)
        models = {
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "random_forest": RandomForestClassifier(random_state=42),
            "knn": KNeighborsClassifier(n_neighbors=int(p.get("n_neighbors",5))),
            "naive_bayes": GaussianNB(),
            "lda": LinearDiscriminantAnalysis(),
        }
        model = models[kind]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred)
        cm_df = pd.DataFrame(cm, index=[f"Actual {i}" for i in range(cm.shape[0])], columns=[f"Predicted {i}" for i in range(cm.shape[1])])
        return {
            "summary": {"accuracy": round(float(acc),4), "n_train": len(X_train), "n_test": len(X_test)},
            "tables": {"Confusion Matrix": cm_df.reset_index().rename(columns={"index":""}),
                       "Performance": pd.DataFrame({"Metric":["Accuracy"], "Value":[acc]}).round(4)},
            "extras": {"data": sub, "model": model, "X_test": X_test, "y_test": y_test, "y_pred": pred},
            "p_value": None, "alpha": float(p.get("alpha",0.05)),
        }
    return _run
# ============================================================
# EFFECT SIZES
# ============================================================
def _cohens_d(df, value, group, p):
    sub = df[[value, group]].dropna()
    groups = sub[group].unique()
    if len(groups) != 2:
        return {"ok": False, "error": "Need 2 groups."}
    g1 = pd.to_numeric(sub[sub[group]==groups[0]][value], errors="coerce").dropna()
    g2 = pd.to_numeric(sub[sub[group]==groups[1]][value], errors="coerce").dropna()
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1-1)*g1.std()**2 + (n2-1)*g2.std()**2) / (n1+n2-2))
    d = (g1.mean() - g2.mean()) / pooled_std
    return {
        "summary": {"cohens_d": float(d), "magnitude": _interp_d(d)},
        "tables": {"Cohen's d": pd.DataFrame({
            "Metric":["d","|d|","Magnitude"], "Value":[d, abs(d), _interp_d(d)]})},
        "extras": {"data": sub, "g1": g1, "g2": g2},
        "p_value": None, "alpha": float(p.get("alpha",0.05)),
    }


def _interp_d(d):
    ad = abs(d)
    if ad < 0.2: return "Negligible"
    if ad < 0.5: return "Small"
    if ad < 0.8: return "Medium"
    return "Large"


def _cramers_v(df, v1, v2, p):
    ct = pd.crosstab(df[v1], df[v2])
    chi2, pv, dof, _ = sp_stats.chi2_contingency(ct)
    n = ct.sum().sum()
    phi2 = chi2/n
    r, k = ct.shape
    v = np.sqrt(phi2 / min(k-1, r-1))
    return {
        "summary": {"cramers_v": round(float(v),4)},
        "tables": {"Cramér's V": pd.DataFrame({
            "Metric":["Cramér's V","Chi-Square","p-value"], "Value":[round(float(v),4), round(float(chi2),4), round(float(pv),6)]})},
        "extras": {"matrix": ct, "data": df[[v1,v2]].dropna()},
        "p_value": float(pv), "alpha": float(p.get("alpha",0.05)),
    }
# ============================================================
# DISPATCH
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

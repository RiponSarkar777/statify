"""
Statify — Statistical Validation / Readiness (Step 5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sits strictly between analysis_plan.py and the deterministic
calculation stage (stats_engine.py, called elsewhere, never from
this file):

    Analysis Plan   answers "what could be run and how do steps relate?"
    Validation       answers "given the REAL dataframe, is each planned
    (this file)       step structurally and descriptively reasonable to
                        attempt, and what limitations are visible?"
    Calculation      answers "actually perform the operation and
                        produce the statistical result."

────────────────────────────────────────────────────────────────
THE BOUNDARY, STATED PLAINLY
────────────────────────────────────────────────────────────────
This is NOT "don't import stats_engine". The real boundary is:

    DO NOT COMPUTE INFERENTIAL STATISTICS OR USE THEIR OUTPUT AS
    VALIDATION EVIDENCE.

Concretely, this file never computes or reports a p-value, a test
statistic, a confidence interval, a regression coefficient, or a
correlation coefficient produced as an analysis result. It never
calls scipy.stats.shapiro, scipy.stats.levene, or any equivalent —
not even a "simplified" version — and it never imports or calls
stats_engine.py. Diagnostic-named plan steps (Shapiro-Wilk
Normality, Levene's Test) are validated the same way every other
step is: is the DATA sufficient to attempt this later, not "what
would this diagnostic show." The descriptive facts this file DOES
compute (skewness, a variance ratio, a missingness percentage) are
plain descriptive statistics — no null hypothesis, no p-value, no
"is this difference real" claim behind any of them — and Layer 3
below exists specifically to stop a descriptive fact from being
mis-attached to an analysis it doesn't actually speak to (e.g.
univariate skewness is not evidence about Pearson's declared
BIVARIATE normality assumption, and is deliberately withheld there).

────────────────────────────────────────────────────────────────
THREE-LAYER ARCHITECTURE (avoids a 55-tool if/elif chain)
────────────────────────────────────────────────────────────────
Layer 1 — Generic checks: apply to every step regardless of tool or
  shape (missingness, sample size, constant-variable detection,
  structural re-confirmation that named columns still exist/still
  have the expected type).

Layer 2 — Family/shape checks: dispatched by the ACTUAL TYPES of the
  matched variables (not just the role-key names — checked directly
  against tool_registry.py: 'var1'/'var2' is used by BOTH Paired
  t-Test [two numeric] and Chi-Square Test of Independence [two
  categorical], which need entirely different family checks, so
  dispatch here is by resolved semantic type, not by role-key
  spelling). Produces group-size/imbalance/variance-ratio for
  group-shaped steps, contingency/expected-cell-count facts for
  categorical-pair steps, skewness/outlier facts for numeric
  variables — always framed as descriptive facts, not warnings, until
  Layer 3 (or the generic thresholds) decide whether a fact rises to
  a warning for THIS particular kind of step.

Layer 3 — Small, explicit, named exceptions: a short, hand-curated
  table for the handful of cases where a family-level default would
  be actively misleading for a specific tool (Pearson, Spearman,
  Kendall's Tau, the regression family — see _LAYER_3_OVERRIDES
  below). This is NOT a second registry: it only suppresses or
  annotates what Layer 2 would otherwise produce; it never invents a
  new check from scratch. If this table ever grows substantially,
  that is a signal to revisit tool_registry.py's own assumptions
  metadata as a deliberate, separately-approved change — not
  something this file should silently expand into.

────────────────────────────────────────────────────────────────
SAMPLE-SIZE SEMANTICS
────────────────────────────────────────────────────────────────
READY does not mean "adequately powered", "statistically reliable",
"scientifically valid", or "assumptions proven". It means: this
validation layer found no material readiness problem within the
checks it is responsible for. NOT_READY means the operation cannot
reasonably proceed at all (a hard structural/data limitation, not a
judgment about power). Sample-size thresholds used below are
centralized in _SampleSizeThresholds, explicitly labeled as
rules-of-thumb, and are not a formal power analysis.
"""

import pandas as pd


# ============================================================
# Centralized, documented thresholds (Section 5 of the design)
# ============================================================
class _Thresholds:
    """
    All numeric thresholds used anywhere in this file live here, in
    one place, each documented with what it means and why it was
    chosen. These are commonly-cited rules of thumb, not a formal
    power analysis (which would require a known/assumed effect size,
    which this layer has no way to determine) — every warning message
    that cites one of these says so explicitly.
    """
    # Below this many usable (non-missing) observations, a step is a
    # hard NOT_READY: too few to compute anything meaningful, not
    # merely "small". This is a low, mathematical-definedness floor,
    # not a power threshold.
    MIN_USABLE_N_HARD = 3

    # Below this per-group / total-n, calculation is still POSSIBLE
    # but flagged NEEDS_ATTENTION as a widely-cited informal rule of
    # thumb (commonly seen in introductory statistics guidance) for
    # when small-sample caution is warranted. Not a formal power
    # calculation.
    MIN_GROUP_N_RULE_OF_THUMB = 20

    # Total-sample-size rule of thumb. CONTEXTUAL, NOT UNIVERSAL: this
    # is applied only inside the two-numeric-variable family check
    # (paired analyses, on complete pairs; relationship-shaped
    # analyses, on usable observations) — see
    # _two_numeric_sample_size_warning(). It is deliberately NOT
    # applied to every step regardless of shape (a blanket "n < 30"
    # rule used to live in Layer 1 and was removed: 30 is not a
    # meaningful reference point for group comparisons, categorical
    # tables, single-variable steps, or diagnostics, each of which has
    # its own, more relevant family-level check or none at all). It is
    # a widely-cited informal convention, NOT a requirement and NOT a
    # formal power analysis.
    MIN_TOTAL_N_RULE_OF_THUMB = 30

    # A group below this size cannot have a variance computed at all
    # (needs 2+ points) — hard floor, contributes to NOT_READY.
    MIN_N_FOR_VARIANCE = 2

    # Missingness thresholds (reusing the same 30% "high missingness"
    # precedent already established in this project's profiler.py
    # readiness-signal design, kept consistent rather than reinvented).
    HIGH_MISSINGNESS_PCT = 30.0

    # Group imbalance: ratio of largest to smallest group size.
    # Reusing the same threshold precedent from profiler.py's own
    # readiness signals (imbalance ratio > 4 flagged there too).
    GROUP_IMBALANCE_RATIO_WARNING = 4.0

    # Descriptive skewness magnitude considered "notably high" —
    # same value already used as a precedent in this project's
    # profiler.py readiness-signal design.
    HIGH_SKEW_THRESHOLD = 1.0

    # Expected-cell-count rule of thumb for contingency-table-style
    # analyses (a very standard, widely-cited convention — cells
    # with expected count under 5 make the chi-square approximation
    # less reliable). Descriptive fact, not a computed p-value.
    MIN_EXPECTED_CELL_COUNT = 5.0

    # Near-zero variance floor for "this is effectively constant".
    NEAR_ZERO_VARIANCE = 1e-8


# ============================================================
# Layer 3 — small, explicit, named overrides
# ============================================================
# tool name -> set of Layer-2 signal categories to SUPPRESS for that
# tool, plus optional extra notes to add. This never adds a new
# CHECK; it only withholds or annotates what Layer 2 would otherwise
# have produced, for a small number of named, justified cases.
_LAYER_3_OVERRIDES = {
    "Pearson Correlation": {
        "suppress": {"skewness"},
        "notes": [
            "Pearson Correlation's declared assumption is bivariate normality. "
            "Univariate skewness of each variable separately is not a direct "
            "proxy for that and is not reported here to avoid implying it was checked.",
        ],
    },
    "Spearman Correlation": {
        "suppress": {"skewness"},
        "notes": [
            "Spearman Correlation is a rank-based method; marginal skewness of "
            "the original values is not a meaningful readiness concern for it.",
        ],
    },
    "Kendall's Tau": {
        "suppress": {"skewness"},
        "notes": [
            "Kendall's Tau is a rank-based method; marginal skewness of the "
            "original values is not a meaningful readiness concern for it.",
        ],
    },
    "Simple Linear Regression": {
        "suppress": set(),
        "notes": [
            "Residual-based assumptions (normality of residuals, homoscedasticity) "
            "cannot be assessed before the model is fit and are not evaluated here.",
        ],
    },
    "Multiple Linear Regression": {
        "suppress": set(),
        "notes": [
            "Residual-based assumptions (normality of residuals, homoscedasticity) "
            "cannot be assessed before the model is fit and are not evaluated here.",
        ],
    },
    # --- Diagnostic tools (correction 2/3): Step 5 validates whether
    # these can structurally be RUN LATER, not whether their eventual
    # result will look a certain way. A generic descriptive signal
    # (skewness, group imbalance) computed for OTHER purposes must not
    # leak into a diagnostic step's own readiness status — that would
    # be using a descriptive proxy to imply something about a test
    # this file explicitly never runs. "skewness" and "group_warnings"
    # are suppressed for the diagnostics themselves; basic structural
    # evidence (n, missingness, constant-variable detection — all
    # still Layer 1) is retained, as required.
    "Shapiro-Wilk Normality": {
        "suppress": {"skewness"},
        "notes": [
            "This entry confirms the data is structurally sufficient to run "
            "Shapiro-Wilk later; the test itself is not executed at this stage, "
            "and descriptive skewness is not used as a proxy for its result.",
        ],
    },
    "Levene's Test": {
        # variance_ratio is also withheld here: a descriptive variance
        # ratio on the Levene step itself would be a proxy for exactly
        # what Levene's Test assesses, which this file must not offer
        # as a stand-in for the diagnostic's later result.
        "suppress": {"skewness", "group_warnings", "variance_ratio"},
        "notes": [
            "This entry confirms the data is structurally sufficient to run "
            "Levene's Test later; the test itself is not executed at this stage. "
            "Generic group-imbalance/small-group warnings are not applied to this "
            "diagnostic step, since they describe the comparison it would assess, "
            "not this step's own readiness to run.",
        ],
    },
    # --- Fisher's Exact Test (correction 4): Fisher's is specifically
    # designed for small contingency tables, so the chi-square-style
    # low-expected-cell-count warning (relevant to chi-square-style
    # approximations) is misleading here. The expected-cell-count
    # numbers themselves remain in evidence as descriptive information;
    # only the WARNING text is suppressed, narrowly, for this one tool
    # — Chi-Square Test of Independence keeps the warning unchanged.
    "Fisher's Exact Test": {
        "suppress": {"expected_cell_warning"},
        "notes": [
            "Fisher's Exact Test is specifically designed for small contingency "
            "tables, so a low-expected-cell-count warning (relevant to "
            "chi-square-style approximations) is not applied here.",
        ],
    },
}


# ============================================================
# Helpers
# ============================================================
def _flatten(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _resolve_column_type(df: pd.DataFrame, col: str) -> str | None:
    """Lightweight, local type resolution — numeric / categorical /
    boolean / datetime / text / None(missing). Deliberately simple
    and self-contained rather than importing profiler.py's
    classify_columns, to keep this file fully independent (no
    coupling to a module outside this step's own concern) and
    because only a coarse type is needed here, not the fuller
    classification profiler.py performs."""
    if col not in df.columns:
        return None
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "categorical"


def _missingness_pct(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 100.0
    return round(100.0 * df[col].isna().sum() / len(df), 2)


def _usable_n(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].dropna().shape[0])


def _is_constant(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = df[col].dropna()
    if len(s) == 0:
        return False
    if pd.api.types.is_numeric_dtype(s):
        return float(s.var()) < _Thresholds.NEAR_ZERO_VARIANCE if len(s) > 1 else True
    return s.nunique() <= 1


# ============================================================
# Layer 1 — Generic checks
# ============================================================
def _layer1_generic(df: pd.DataFrame, matched_variables: dict) -> dict:
    """
    Universal, tool-agnostic checks: do the named columns exist and
    have usable data at all. Returns a dict with blocking_issues,
    warnings, and evidence fragments — never a verdict by itself
    (the caller combines this with Layer 2/3 to reach a status).
    """
    blocking = []
    warnings = []
    evidence = {"missingness_pct": {}, "n_total": None}

    all_cols = []
    for v in matched_variables.values():
        all_cols.extend(_flatten(v))
    all_cols = list(dict.fromkeys(all_cols))  # de-dup, preserve order

    if not all_cols:
        blocking.append("No variables are named for this step.")
        return {"blocking_issues": blocking, "warnings": warnings, "evidence": evidence}

    usable_ns = []
    for col in all_cols:
        if col not in df.columns:
            blocking.append(f"Column '{col}' was not found in the dataset.")
            continue
        miss_pct = _missingness_pct(df, col)
        evidence["missingness_pct"][col] = miss_pct
        if miss_pct >= _Thresholds.HIGH_MISSINGNESS_PCT:
            warnings.append(
                f"'{col}' has {miss_pct}% missing values, which is high enough to be worth noting."
            )
        n = _usable_n(df, col)
        usable_ns.append(n)
        if n == 0:
            blocking.append(f"Column '{col}' has no usable (non-missing) values.")
        if _is_constant(df, col):
            blocking.append(f"Column '{col}' is constant (or near-constant) — no variation to analyze.")

    if usable_ns:
        evidence["n_total"] = min(usable_ns)  # the binding constraint across all named columns
        if evidence["n_total"] is not None and 0 < evidence["n_total"] < _Thresholds.MIN_USABLE_N_HARD:
            blocking.append(
                f"Only {evidence['n_total']} usable observation(s) available — "
                f"too few for this operation to be meaningfully defined."
            )
        # NOTE (correction 6): a blanket "n < 30 -> warning" rule used to
        # live here, firing for EVERY step regardless of shape. That is
        # too broad: it flagged steps (e.g. a well-structured categorical
        # comparison, or a step where a more specific family-level check
        # already covers sample size appropriately) as NEEDS_ATTENTION
        # solely because total n was under 30, with no regard for whether
        # 30 is a remotely relevant reference point for that kind of
        # analysis. Total-sample-size rule-of-thumb warnings are now
        # applied only inside the specific Layer-2 family check where a
        # total-n heuristic is actually contextually meaningful (see
        # _layer2_numeric_descriptive's relationship-shape handling
        # below) — group comparisons already have their own per-group
        # check (MIN_GROUP_N_RULE_OF_THUMB), and categorical-pair steps
        # already have their own expected-cell-count check; neither
        # needs a redundant, less-relevant blanket total-n warning on
        # top. Layer 1 keeps only the true hard floor (MIN_USABLE_N_HARD)
        # here, which applies to everything because "can this be
        # computed at all" is universal in a way "is 30 the right
        # reference number" is not.

    return {"blocking_issues": blocking, "warnings": warnings, "evidence": evidence}


# ============================================================
# Layer 2 — Family/shape checks (dispatched by resolved TYPES,
# not by role-key spelling — see module docstring for why)
# ============================================================
def _layer2_group_comparison(df: pd.DataFrame, value_col: str, group_col: str, suppress: set) -> dict:
    """Numeric outcome + categorical/boolean group. Shared by
    Independent t-Test, One-Way ANOVA, Mann-Whitney, Levene's Test,
    Cohen's d, etc. — the group-structure facts below are properties
    of the COMPARISON, not of any one specific test.

    'suppress' (from Layer 3) can withhold the group-imbalance/
    small-group WARNINGS specifically for a diagnostic tool like
    Levene's Test (correction 3): the group_sizes/imbalance_ratio
    FACTS still populate evidence either way (retaining basic
    structural evidence, as required) — only the warning text tied
    to those facts is withheld when 'group_warnings' is suppressed,
    since a diagnostic step's own readiness should reflect whether
    it can be RUN, not restate warnings that belong to the comparison
    it would assess.
    """
    warnings, evidence = [], {}
    if value_col not in df.columns or group_col not in df.columns:
        return {"warnings": warnings, "evidence": evidence}

    sub = df[[value_col, group_col]].dropna()
    group_sizes = sub.groupby(group_col, observed=True)[value_col].size().to_dict()
    group_sizes = {str(k): int(v) for k, v in group_sizes.items()}
    evidence["group_sizes"] = group_sizes

    suppress_group_warnings = "group_warnings" in suppress

    if len(group_sizes) >= 2:
        sizes = list(group_sizes.values())
        largest, smallest = max(sizes), min(sizes)
        if smallest > 0:
            ratio = round(largest / smallest, 2)
            evidence["group_imbalance_ratio"] = ratio
            if ratio > _Thresholds.GROUP_IMBALANCE_RATIO_WARNING and not suppress_group_warnings:
                warnings.append(
                    f"Group sizes are imbalanced (largest/smallest ratio = {ratio}), "
                    f"above the {_Thresholds.GROUP_IMBALANCE_RATIO_WARNING}x rule-of-thumb threshold."
                )
        small_groups = [g for g, n in group_sizes.items() if n < _Thresholds.MIN_GROUP_N_RULE_OF_THUMB]
        if small_groups and not suppress_group_warnings:
            warnings.append(
                f"Group(s) {small_groups} have fewer than {_Thresholds.MIN_GROUP_N_RULE_OF_THUMB} "
                f"observations — a common rule-of-thumb threshold for reliable group comparisons; "
                f"no formal power analysis was performed."
            )

        # Descriptive variance ratio (NOT Levene's test — no p-value,
        # no test statistic, just a plain ratio of sample variances).
        variances = sub.groupby(group_col, observed=True)[value_col].var()
        variances = variances.dropna()
        if len(variances) >= 2 and variances.min() > 0 and "variance_ratio" not in suppress:
            var_ratio = round(float(variances.max() / variances.min()), 2)
            evidence["variance_ratio"] = var_ratio

    return {"warnings": warnings, "evidence": evidence}


def _layer2_categorical_pair(df: pd.DataFrame, col_a: str, col_b: str, suppress: set) -> dict:
    """Two categorical/boolean variables — contingency-table-style
    descriptive facts (expected cell counts), not a chi-square
    calculation itself.

    'suppress' (from Layer 3) can withhold ONLY the low-expected-cell
    WARNING (key: "expected_cell_warning") for Fisher's Exact Test
    (correction 4) — the expected-count numbers themselves stay in
    evidence either way as descriptive information. This is
    deliberately not a broad categorical-pair exception: Chi-Square
    Test of Independence has no such suppression and keeps the warning.
    """
    warnings, evidence = [], {}
    if col_a not in df.columns or col_b not in df.columns:
        return {"warnings": warnings, "evidence": evidence}

    sub = df[[col_a, col_b]].dropna()
    if len(sub) == 0:
        return {"warnings": warnings, "evidence": evidence}

    ct = pd.crosstab(sub[col_a], sub[col_b])
    evidence["contingency_shape"] = list(ct.shape)

    row_totals = ct.sum(axis=1)
    col_totals = ct.sum(axis=0)
    grand_total = ct.values.sum()
    if grand_total > 0:
        expected = pd.DataFrame(
            [[row_totals[r] * col_totals[c] / grand_total for c in ct.columns] for r in ct.index],
            index=ct.index, columns=ct.columns,
        )
        below = int((expected < _Thresholds.MIN_EXPECTED_CELL_COUNT).values.sum())
        evidence["cells_below_expected_min"] = below
        evidence["min_expected_count"] = round(float(expected.values.min()), 2)
        if below > 0 and "expected_cell_warning" not in suppress:
            warnings.append(
                f"{below} of {expected.size} contingency-table cell(s) have an expected count "
                f"below {_Thresholds.MIN_EXPECTED_CELL_COUNT} — a common rule-of-thumb threshold "
                f"below which chi-square-style approximations are considered less reliable."
            )

    return {"warnings": warnings, "evidence": evidence}


# Tools that are genuinely PAIRED analyses — determined by actual tool
# identity (per the correction: "do not assume every two-numeric
# analysis is paired"), not by role-shape alone. Confirmed directly
# against tool_registry.py: both use the var1/var2 role-key pair, but
# so do Chi-Square Test of Independence and Fisher's Exact Test (which
# are categorical-pair, not paired-numeric) — shape alone cannot tell
# these apart, so tool identity is the correct signal here.
_PAIRED_ANALYSIS_TOOLS = {"Paired t-Test", "Wilcoxon Signed-Rank Test"}


def _two_numeric_sample_size_warning(n: int, what: str) -> str | None:
    """Contextual total-n heuristic for the two-numeric-variable family
    (see _Thresholds.MIN_TOTAL_N_RULE_OF_THUMB). Returns a warning
    string, or None if n is at/above the rule-of-thumb reference.
    Wording deliberately says this is a heuristic for this kind of
    analysis, NOT a requirement — it must never read as "n >= 30 is
    universally needed"."""
    if n < _Thresholds.MIN_TOTAL_N_RULE_OF_THUMB:
        return (
            f"{what} ({n}) is below a common rule-of-thumb reference "
            f"(often cited around {_Thresholds.MIN_TOTAL_N_RULE_OF_THUMB}) for this kind of "
            f"analysis. This is an informal heuristic, not a requirement, and no formal "
            f"power analysis was performed."
        )
    return None


def _layer2_paired_numeric(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """
    Complete-pair readiness check for genuinely paired analyses
    (Paired t-Test, Wilcoxon Signed-Rank Test — see
    _PAIRED_ANALYSIS_TOOLS). A paired analysis needs COMPLETE PAIRS:
    a row where col_a is present but col_b is missing (or vice versa)
    contributes to neither variable's paired n. Each variable can
    individually have plenty of usable values while the number of
    complete pairs is tiny — Layer 1 only looks at each column
    separately, so this check is what actually protects a paired step.

    This is a purely descriptive/structural COUNT
    (paired_df = df[[col_a, col_b]].dropna(); len(paired_df)). No
    paired t-statistic, no p-value, no differences are computed —
    only how many complete pairs exist.
    """
    blocking, warnings, evidence = [], [], {}
    if col_a not in df.columns or col_b not in df.columns:
        return {"blocking_issues": blocking, "warnings": warnings, "evidence": evidence}

    paired_df = df[[col_a, col_b]].dropna()
    paired_n = len(paired_df)
    evidence["paired_n"] = paired_n

    if paired_n < _Thresholds.MIN_USABLE_N_HARD:
        blocking.append(
            f"Only {paired_n} complete pair(s) of '{col_a}'/'{col_b}' are available — a paired "
            f"analysis needs both values present in the same row, and too few complete pairs "
            f"exist for this operation to be meaningfully defined."
        )
    else:
        w = _two_numeric_sample_size_warning(paired_n, f"Number of complete pairs of '{col_a}'/'{col_b}'")
        if w:
            warnings.append(w)

    return {"blocking_issues": blocking, "warnings": warnings, "evidence": evidence}


def _layer2_numeric_descriptive(df: pd.DataFrame, col: str, suppress: set) -> dict:
    """Single-numeric-variable descriptive signals — skewness and an
    IQR-based outlier count. Purely descriptive: a skewness VALUE
    and an outlier COUNT, no hypothesis test, no p-value. 'suppress'
    (from Layer 3) can withhold the skewness signal specifically for
    tools where it would be a misleading proxy (see module docstring)."""
    warnings, evidence = [], {}
    if col not in df.columns:
        return {"warnings": warnings, "evidence": evidence}
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) < 3:
        return {"warnings": warnings, "evidence": evidence}

    if "skewness" not in suppress:
        skew = round(float(s.skew()), 4)
        evidence.setdefault("skewness", {})[col] = skew
        if abs(skew) > _Thresholds.HIGH_SKEW_THRESHOLD:
            warnings.append(
                f"'{col}' has a notably high skewness ({skew}) — worth noting as a descriptive fact."
            )

    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        evidence.setdefault("n_outliers_iqr", {})[col] = n_out

    return {"warnings": warnings, "evidence": evidence}


def _layer2_multi_variable(df: pd.DataFrame, cols: list) -> dict:
    """Multi-variable steps (Descriptive Statistics, K-Means, PCA,
    etc.) — usable-row count after listwise deletion across all named
    columns, and which (if any) are constant."""
    warnings, evidence = [], {}
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return {"warnings": warnings, "evidence": evidence}
    sub = df[existing].dropna()
    evidence["n_usable_rows_all_variables"] = int(sub.shape[0])
    constant_cols = [c for c in existing if _is_constant(df, c)]
    if constant_cols:
        evidence["constant_variables"] = constant_cols
    return {"warnings": warnings, "evidence": evidence}


def _dispatch_layer2(df: pd.DataFrame, matched_variables: dict, tool_name: str) -> dict:
    """
    Decide which Layer-2 family check(s) apply, based on the RESOLVED
    TYPES of the matched variables (not the role-key names — see
    module docstring: 'var1'/'var2' is used by both Paired t-Test
    [numeric+numeric] and Chi-Square Test of Independence
    [categorical+categorical], which need different family checks).
    Applies Layer-3 suppressions where a tool name matches an entry
    in _LAYER_3_OVERRIDES.

    Two-numeric-variable steps split by TOOL IDENTITY, not shape:
    genuinely paired analyses (_PAIRED_ANALYSIS_TOOLS) get the
    complete-pair check; every other two-numeric step (correlation,
    regression, ...) is NOT assumed to be paired and keeps the
    per-variable behavior plus the contextual total-n heuristic.

    Returns blocking_issues as well as warnings/evidence, since a
    family-level check (currently: too few complete pairs) can
    establish a hard NOT_READY condition that Layer 1's
    per-column view cannot see.
    """
    suppress = _LAYER_3_OVERRIDES.get(tool_name, {}).get("suppress", set())
    layer3_notes = list(_LAYER_3_OVERRIDES.get(tool_name, {}).get("notes", []))

    all_warnings = []
    blocking = []
    evidence = {}

    cols_flat = []
    for v in matched_variables.values():
        cols_flat.extend(_flatten(v))
    types = {c: _resolve_column_type(df, c) for c in cols_flat}

    numeric_cols = [c for c, t in types.items() if t == "numeric"]
    cat_cols = [c for c, t in types.items() if t in ("categorical", "boolean")]

    if len(cols_flat) >= 3:
        r = _layer2_multi_variable(df, cols_flat)
        all_warnings += r["warnings"]
        evidence.update(r["evidence"])
    elif len(numeric_cols) == 1 and len(cat_cols) == 1:
        r = _layer2_group_comparison(df, numeric_cols[0], cat_cols[0], suppress)
        all_warnings += r["warnings"]
        evidence.update(r["evidence"])
        r2 = _layer2_numeric_descriptive(df, numeric_cols[0], suppress)
        all_warnings += r2["warnings"]
        evidence.update(r2["evidence"])
    elif len(cat_cols) == 2:
        r = _layer2_categorical_pair(df, cat_cols[0], cat_cols[1], suppress)
        all_warnings += r["warnings"]
        evidence.update(r["evidence"])
    elif len(numeric_cols) == 2:
        if tool_name in _PAIRED_ANALYSIS_TOOLS:
            rp = _layer2_paired_numeric(df, numeric_cols[0], numeric_cols[1])
            blocking += rp["blocking_issues"]
            all_warnings += rp["warnings"]
            evidence.update(rp["evidence"])
        elif set(matched_variables.keys()) == {"x", "y"}:
            # Relationship-shaped step (role keys are literally x/y:
            # correlation, simple regression, ...). The contextual
            # total-n heuristic applies ONLY here — keyed on the
            # step's role contract, not merely "two numeric columns",
            # so e.g. Descriptive Statistics or K-Means that happen to
            # name exactly two numeric variables are NOT swept in.
            n_usable = min(_usable_n(df, c) for c in numeric_cols)
            w = _two_numeric_sample_size_warning(n_usable, "Number of usable observations")
            if w and n_usable >= _Thresholds.MIN_USABLE_N_HARD:
                all_warnings.append(w)
        for c in numeric_cols:
            r = _layer2_numeric_descriptive(df, c, suppress)
            all_warnings += r["warnings"]
            evidence.update(r["evidence"])
    elif len(numeric_cols) == 1 and len(cols_flat) == 1:
        r = _layer2_numeric_descriptive(df, numeric_cols[0], suppress)
        all_warnings += r["warnings"]
        evidence.update(r["evidence"])
    elif len(cols_flat) >= 1:
        r = _layer2_multi_variable(df, cols_flat)
        all_warnings += r["warnings"]
        evidence.update(r["evidence"])

    if layer3_notes:
        evidence["notes"] = layer3_notes

    return {"blocking_issues": blocking, "warnings": all_warnings, "evidence": evidence}


# ============================================================
# Per-step validation
# ============================================================
def _validate_concrete_step(df: pd.DataFrame, step: dict) -> dict:
    """
    Validate one ordinary plan step (has a real tool and matched
    variables). Returns the step's contribution: blocking_issues,
    warnings, evidence, status. Never touches stats_engine, never
    computes a p-value/test statistic.
    """
    tool_name = step.get("tool")
    matched_variables = step.get("matched_variables") or {}

    l1 = _layer1_generic(df, matched_variables)
    blocking = list(l1["blocking_issues"])
    warnings = list(l1["warnings"])
    evidence = dict(l1["evidence"])

    if not blocking:
        l2 = _dispatch_layer2(df, matched_variables, tool_name)
        blocking += l2.get("blocking_issues", [])
        warnings += l2["warnings"]
        evidence.update(l2["evidence"])

    from tool_registry import STAT_TOOLS
    if tool_name in STAT_TOOLS:
        declared = STAT_TOOLS[tool_name].get("assumptions", [])
        if declared:
            evidence["declared_assumptions"] = list(declared)

    if blocking:
        status = "NOT_READY"
    elif warnings:
        status = "NEEDS_ATTENTION"
    else:
        status = "READY"

    return {"blocking_issues": blocking, "warnings": warnings, "evidence": evidence, "status": status}


def _validate_exploratory_step(df: pd.DataFrame, step: dict) -> dict:
    """
    broad_exploratory category-level entries (tool=None) are not
    ordinary analysis steps — there is no specific tool or matched
    variable set to check readiness for. The only honest thing to
    verify is whether the category's previously-eligible tools are
    still structurally available against the live dataframe (a cheap
    re-check of candidates.py's own eligible_tools list, since the
    dataframe may have changed since the plan was built).
    """
    from tool_registry import STAT_TOOLS
    still_valid, no_longer_valid = [], []
    for tool_name in step.get("eligible_tools") or []:
        spec = STAT_TOOLS.get(tool_name)
        if spec is None:
            no_longer_valid.append(tool_name)
            continue
        ok = True
        for role_def in spec["roles"].values():
            if role_def.get("optional"):
                continue
            # A coarse re-check only: is there at least one column of
            # an acceptable type still present in the dataframe.
            found = False
            for col in df.columns:
                if _resolve_column_type(df, col) in role_def["types"]:
                    found = True
                    break
            if not found:
                ok = False
                break
        (still_valid if ok else no_longer_valid).append(tool_name)

    return {
        "blocking_issues": [],
        "warnings": [],
        "evidence": {
            "eligible_tools_still_valid": still_valid,
            "eligible_tools_no_longer_valid": no_longer_valid,
        },
        "status": "NOT_APPLICABLE",
    }


# ============================================================
# Scope handling
# ============================================================
def _steps_in_scope(plan_steps: list, scope: str) -> tuple[list, list]:
    """Return (to_validate, to_skip) based on scope, per the approved
    design: quick=primary only; standard=primary+diagnostic+supporting
    +alternative (conditional, still validated so its readiness is
    known); comprehensive=every concrete step. Exploratory-role steps
    are ALWAYS routed to validation (not skipped) regardless of scope
    — scope controls how much effort goes into validating CONCRETE
    steps, but an exploratory entry's NOT_APPLICABLE status is a
    structural fact about what kind of plan entry it is (no tool, no
    matched variables), not something scope should hide behind the
    less-informative NOT_VALIDATED status. validate_plan() itself
    routes exploratory-role steps to _validate_exploratory_step,
    which always returns NOT_APPLICABLE — this function only needs to
    make sure they aren't skipped before reaching it."""
    if scope == "quick":
        include_roles = {"primary"}
    elif scope == "standard":
        include_roles = {"primary", "diagnostic", "supporting", "alternative"}
    else:  # comprehensive
        include_roles = {"primary", "diagnostic", "supporting", "alternative"}

    to_validate, to_skip = [], []
    for step in plan_steps:
        if step.get("role") == "exploratory" or step.get("role") in include_roles:
            to_validate.append(step)
        else:
            to_skip.append(step)
    return to_validate, to_skip


# ============================================================
# Public entry point
# ============================================================
def validate_plan(plan: dict, df: pd.DataFrame) -> dict:
    """
    Consume an Analysis Plan (from analysis_plan.py's build_analysis_plan)
    and the real dataframe, and produce a Validation Result. Never
    modifies the plan, never modifies the dataframe, never imports or
    calls stats_engine.py, never computes an inferential statistic.
    """
    scope = plan.get("scope", "standard")
    plan_steps = plan.get("plan", [])

    to_validate, to_skip = _steps_in_scope(plan_steps, scope)

    result_steps = []

    for step in to_skip:
        result_steps.append({
            "step_id": step["step_id"],
            "tool": step.get("tool"),
            "role": step.get("role"),
            "conditional": step.get("role") == "alternative",
            "validated": False,
            "status": "NOT_VALIDATED",
            "blocking_issues": [],
            "warnings": [],
            "evidence": {},
        })

    for step in to_validate:
        is_exploratory = step.get("role") == "exploratory" or step.get("tool") is None
        if is_exploratory:
            r = _validate_exploratory_step(df, step)
            validated_flag = False
        else:
            r = _validate_concrete_step(df, step)
            validated_flag = True

        result_steps.append({
            "step_id": step["step_id"],
            "tool": step.get("tool"),
            "role": step.get("role"),
            "conditional": step.get("role") == "alternative",
            "validated": validated_flag,
            "status": r["status"],
            "blocking_issues": r["blocking_issues"],
            "warnings": r["warnings"],
            "evidence": r["evidence"],
        })

    # Preserve original step order (to_skip / to_validate split it).
    order = {s["step_id"]: i for i, s in enumerate(plan_steps)}
    result_steps.sort(key=lambda s: order.get(s["step_id"], 0))

    # overall_status: worst case among steps that are neither
    # conditional (alternatives) nor NOT_VALIDATED/NOT_APPLICABLE.
    # An alternative's own readiness never determines the plan's
    # overall status — the plan never proposed to run it
    # unconditionally in the first place.
    contributing = [
        s for s in result_steps
        if not s["conditional"] and s["status"] in ("READY", "NEEDS_ATTENTION", "NOT_READY")
    ]
    if any(s["status"] == "NOT_READY" for s in contributing):
        overall = "NOT_READY"
    elif any(s["status"] == "NEEDS_ATTENTION" for s in contributing):
        overall = "NEEDS_ATTENTION"
    else:
        overall = "READY"

    return {
        "plan_scope": scope,
        "overall_status": overall,
        "steps": result_steps,
    }

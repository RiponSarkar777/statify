"""
Isolated tests for readiness.py (Step 5).

Run with:   python test_readiness.py      (or: pytest test_readiness.py)

Sections
  1. AST / static boundary checks (no inferential computation, no
     stats_engine, no scipy)
  2. Original Step 5 coverage (A-Z)
  3. Regression tests for the six corrections
  4. Protected-file byte-identity check

Nothing here modifies any project file.
"""
import ast
import hashlib
import inspect
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import readiness  # noqa: E402
from readiness import validate_plan  # noqa: E402
from profiler import classify_columns, dataset_capability_map, get_column_types  # noqa: E402
from objectives import objective_from_example_choice, objective_from_tool_selection  # noqa: E402
from candidates import find_candidate_analyses  # noqa: E402
from analysis_plan import build_analysis_plan  # noqa: E402
from tool_registry import STAT_TOOLS  # noqa: E402


# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------
def step(tool, matched, role="primary", sid="s1", depends_on=None, eligible=None):
    return {"step_id": sid, "tool": tool, "category": "x", "role": role,
            "matched_variables": matched, "eligible_tools": eligible,
            "depends_on": depends_on, "relevance_basis": None, "why": "test"}


def run(df, *steps, scope="standard"):
    return validate_plan({"scope": scope, "plan": list(steps)}, df)


def one(df, tool, matched, **kw):
    return run(df, step(tool, matched), **kw)["steps"][0]


def pipeline(df, objective):
    buckets = get_column_types(df)
    cap = dataset_capability_map(df, buckets)
    cand = find_candidate_analyses(df, cap, buckets, objective)
    plan = build_analysis_plan(cand, objective)
    return plan, validate_plan(plan, df)


def by_tool(val, tool, role=None):
    for s in val["steps"]:
        if s["tool"] == tool and (role is None or s["role"] == role):
            return s
    raise AssertionError(f"no step for {tool}/{role}")


FORBIDDEN_EVIDENCE_KEYS = {
    "p_value", "pvalue", "p-value", "statistic", "test_statistic", "t_statistic",
    "f_statistic", "chi2", "r_value", "coefficient", "confidence_interval",
    "ci_lower", "ci_upper", "w_statistic",
}


def assert_no_inferential_evidence(val):
    for s in val["steps"]:
        bad = [k for k in s["evidence"] if k.lower().replace(" ", "_") in FORBIDDEN_EVIDENCE_KEYS]
        assert not bad, f"{s['step_id']} {s['tool']} evidence contains {bad}"


# ================================================================
# 1. AST / STATIC BOUNDARY CHECKS
# ================================================================
def _imports(path):
    tree = ast.parse(open(path).read())
    mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            mods.append(n.module)
    return mods


def test_ast_imports_only_pandas_and_tool_registry():
    mods = set(_imports(os.path.join(HERE, "readiness.py")))
    assert mods <= {"pandas", "tool_registry"}, mods
    assert "scipy" not in mods and "stats_engine" not in mods
    assert not any(m and m.startswith("scipy") for m in mods)


def test_ast_no_inferential_function_calls():
    tree = ast.parse(open(os.path.join(HERE, "readiness.py")).read())
    called = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            called.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    inferential = {"shapiro", "levene", "ttest_ind", "ttest_rel", "ttest_1samp", "f_oneway",
                   "pearsonr", "spearmanr", "kendalltau", "chi2_contingency", "fisher_exact",
                   "kstest", "mannwhitneyu", "wilcoxon", "kruskal", "linregress", "run_analysis",
                   "OLS", "polyfit", "corr", "corrcoef", "ttest"}
    assert not (called & inferential), called & inferential


# ================================================================
# 2. ORIGINAL STEP 5 COVERAGE (A-Z)
# ================================================================
STEP_KEYS = {"step_id", "tool", "role", "conditional", "validated", "status",
             "blocking_issues", "warnings", "evidence"}


def _gc_df():
    return pd.DataFrame({"marks": [70, 85, 60, 90, 75, 88, 65, 92, 78, 81],
                         "gender": ["M", "F"] * 5})


def test_A_result_contract_and_step_ids():
    obj = objective_from_example_choice("compare_groups", outcome_variable="marks",
                                        grouping_variable="gender", scope="standard")
    plan, val = pipeline(_gc_df(), obj)
    assert set(val) == {"plan_scope", "overall_status", "steps"}
    assert all(set(s) == STEP_KEYS for s in val["steps"])
    assert {s["step_id"] for s in val["steps"]} == {s["step_id"] for s in plan["plan"]}


def test_B_missingness():
    df = pd.DataFrame({"marks": [70, 85, 60, 90, 75, None, None, None, None, 81],
                       "gender": ["M", "F"] * 5})
    s = one(df, "Independent t-Test", {"value": "marks", "group": "gender"})
    assert any("missing" in w.lower() for w in s["warnings"])
    assert s["evidence"]["missingness_pct"]["marks"] > 0


def test_C_constant_variable_not_ready():
    df = pd.DataFrame({"marks": [70] * 10, "gender": ["M", "F"] * 5})
    s = one(df, "Independent t-Test", {"value": "marks", "group": "gender"})
    assert s["status"] == "NOT_READY"
    assert any("constant" in b.lower() for b in s["blocking_issues"])


def test_D_E_F_group_size_imbalance_variance_ratio():
    df = pd.DataFrame({"marks": [70, 85, 60, 90, 75, 88, 65, 92, 78, 81, 95, 55],
                       "gender": ["M"] * 10 + ["F"] * 2})
    s = one(df, "Independent t-Test", {"value": "marks", "group": "gender"})
    assert s["evidence"]["group_sizes"] == {"M": 10, "F": 2}
    assert s["evidence"]["group_imbalance_ratio"] == 5.0
    assert any("imbalanc" in w.lower() for w in s["warnings"])
    assert "variance_ratio" in s["evidence"]


def test_G_skewness_evidence_for_group_comparison():
    s = one(_gc_df(), "Independent t-Test", {"value": "marks", "group": "gender"})
    assert "skewness" in s["evidence"]


def test_H_categorical_pair_structure_and_warning():
    df = pd.DataFrame({"a": ["X"] * 20 + ["Y"] * 2, "b": ["P"] * 11 + ["Q"] * 11})
    s = one(df, "Chi-Square Test of Independence", {"var1": "a", "var2": "b"})
    assert "contingency_shape" in s["evidence"] and "min_expected_count" in s["evidence"]
    assert any("expected count" in w.lower() for w in s["warnings"])


def test_I_J_status_semantics():
    tiny = pd.DataFrame({"marks": [70, 85], "gender": ["M", "F"]})
    assert one(tiny, "Independent t-Test", {"value": "marks", "group": "gender"})["status"] == "NOT_READY"
    small = pd.DataFrame({"marks": [70, 85, 60, 90, 75, 88], "gender": ["M", "F"] * 3})
    assert one(small, "Independent t-Test", {"value": "marks", "group": "gender"})["status"] == "NEEDS_ATTENTION"
    large = pd.DataFrame({"marks": list(range(60, 140)), "gender": ["M", "F"] * 40})
    assert one(large, "Independent t-Test", {"value": "marks", "group": "gender"})["status"] == "READY"


def test_K_L_M_diagnostics_never_carry_inferential_evidence():
    obj = objective_from_example_choice("compare_groups", outcome_variable="marks",
                                        grouping_variable="gender", scope="standard")
    _, val = pipeline(_gc_df(), obj)
    assert_no_inferential_evidence(val)
    sh = by_tool(val, "Shapiro-Wilk Normality")
    assert "statistic" not in sh["evidence"] and "p_value" not in sh["evidence"]
    assert sh["status"] in ("READY", "NEEDS_ATTENTION", "NOT_READY")


def test_P_Q_R_S_layer3_overrides():
    df = pd.DataFrame({"a": list(range(1, 11)), "b": [40, 45, 50, 55, 60, 65, 70, 75, 80, 85]})
    p = one(df, "Pearson Correlation", {"x": "a", "y": "b"})
    assert "skewness" not in p["evidence"]
    assert any("bivariate" in n.lower() for n in p["evidence"]["notes"])
    for t in ("Spearman Correlation", "Kendall's Tau"):
        s = one(df, t, {"x": "a", "y": "b"})
        assert "skewness" not in s["evidence"]
        assert any("rank" in n.lower() for n in s["evidence"]["notes"])
    for t in ("Simple Linear Regression",):
        s = one(df, t, {"y": "b", "x": "a"})
        assert any("residual" in n.lower() for n in s["evidence"]["notes"])


def test_T_alternatives_conditional_true():
    df = pd.DataFrame({"a": list(range(1, 11)), "b": list(range(40, 50))})
    obj = objective_from_tool_selection("Pearson Correlation", {"x": "a", "y": "b"})
    _, val = pipeline(df, obj)
    alts = [s for s in val["steps"] if s["role"] == "alternative"]
    assert alts and all(s["conditional"] is True for s in alts)
    assert all(s["conditional"] is False for s in val["steps"] if s["role"] != "alternative")


def test_U_conditional_alternative_not_ready_does_not_affect_overall():
    df = pd.DataFrame({"good_x": list(range(1, 31)), "good_y": list(range(31, 61)), "const": [5] * 30})
    val = run(df,
              step("Pearson Correlation", {"x": "good_x", "y": "good_y"}, sid="s1"),
              step("Spearman Correlation", {"x": "good_x", "y": "const"}, role="alternative",
                   sid="s2", depends_on=["s1"]))
    assert by_tool(val, "Pearson Correlation")["status"] == "READY"
    alt = by_tool(val, "Spearman Correlation")
    assert alt["status"] == "NOT_READY" and alt["conditional"] is True
    assert val["overall_status"] == "READY"


def test_V_broad_exploratory_not_applicable_at_every_scope():
    for scope in ("quick", "standard", "comprehensive"):
        _, val = pipeline(_gc_df(), objective_from_example_choice("explore", scope=scope))
        assert val["steps"]
        for s in val["steps"]:
            assert s["status"] == "NOT_APPLICABLE" and s["validated"] is False
            assert "eligible_tools_still_valid" in s["evidence"]
        assert val["overall_status"] == "READY"


def test_W_scope_behavior():
    def v(scope):
        obj = objective_from_example_choice("compare_groups", outcome_variable="marks",
                                            grouping_variable="gender", scope=scope)
        return pipeline(_gc_df(), obj)[1]
    q, s, c = v("quick"), v("standard"), v("comprehensive")
    assert all(x["validated"] for x in q["steps"] if x["role"] == "primary")
    assert all((not x["validated"]) and x["status"] == "NOT_VALIDATED"
               for x in q["steps"] if x["role"] != "primary")
    for val in (s, c):
        assert all(x["validated"] for x in val["steps"]
                   if x["role"] in ("primary", "diagnostic", "supporting"))
    assert (q["plan_scope"], s["plan_scope"], c["plan_scope"]) == ("quick", "standard", "comprehensive")


def test_X_empty_dataframe_does_not_crash():
    obj = objective_from_example_choice("compare_groups", outcome_variable="marks",
                                        grouping_variable="gender", scope="standard")
    _, val = pipeline(pd.DataFrame(), obj)
    assert val["overall_status"] in ("READY", "NOT_READY")
    s = one(pd.DataFrame(), "Independent t-Test", {"value": "marks", "group": "gender"})
    assert s["status"] == "NOT_READY"


def test_Y_missing_column_not_ready():
    s = one(_gc_df(), "Independent t-Test", {"value": "nope", "group": "gender"})
    assert s["status"] == "NOT_READY" and any("nope" in b for b in s["blocking_issues"])


def test_edge_cases_no_silent_pass():
    allmiss = pd.DataFrame({"marks": [None] * 10, "gender": ["M", "F"] * 5})
    assert one(allmiss, "Independent t-Test", {"value": "marks", "group": "gender"})["status"] == "NOT_READY"
    onelevel = pd.DataFrame({"marks": [70, 85, 60, 90, 75], "gender": ["M"] * 5})
    assert one(onelevel, "Independent t-Test", {"value": "marks", "group": "gender"})["status"] in ("NOT_READY", "NEEDS_ATTENTION")
    mixed = pd.DataFrame({"marks": [70, "bad", 60, 90, 75, 88, 65, 92, 78, 81], "gender": ["M", "F"] * 5})
    assert one(mixed, "Independent t-Test", {"value": "marks", "group": "gender"})["status"] in ("READY", "NEEDS_ATTENTION", "NOT_READY")
    const = pd.DataFrame({"a": [5] * 10, "b": list(range(10))})
    assert one(const, "Pearson Correlation", {"x": "a", "y": "b"})["status"] == "NOT_READY"


def test_public_contract_unchanged():
    val = run(_gc_df(), step("Independent t-Test", {"value": "marks", "group": "gender"}))
    assert set(val) == {"plan_scope", "overall_status", "steps"}
    assert set(val["steps"][0]) == STEP_KEYS
    assert set(inspect.signature(validate_plan).parameters) == {"plan", "df"}


# ================================================================
# 3. REGRESSION TESTS FOR THE SIX CORRECTIONS
# ================================================================

# ---- Correction 1: paired analyses are validated on COMPLETE PAIRS ----
def _gappy_pair_df():
    """Each variable individually has 8 usable values; only 2 rows have both."""
    a = [1.0, 2, 3, 4, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7, 8]
    b = [np.nan] * 6 + [1, 2, 3, 4, 5, 6, 9, 10]
    return pd.DataFrame({"a": a, "b": b})


def test_c1_paired_individually_sufficient_but_pairs_insufficient_is_not_ready():
    df = _gappy_pair_df()
    assert df["a"].notna().sum() >= 3 and df["b"].notna().sum() >= 3      # each column alone is fine
    assert len(df[["a", "b"]].dropna()) == 2                               # complete pairs are not
    for tool in ("Paired t-Test", "Wilcoxon Signed-Rank Test"):
        s = one(df, tool, {"var1": "a", "var2": "b"})
        assert s["status"] == "NOT_READY", tool
        assert s["evidence"]["paired_n"] == 2
        assert any("complete pair" in b.lower() for b in s["blocking_issues"])


def test_c1_demonstrates_check_is_on_complete_pairs_not_per_column():
    """Same per-column usable counts (8/8); only the ROW ALIGNMENT differs."""
    gappy = _gappy_pair_df()
    aligned = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 7, 8] + [np.nan] * 6,
                            "b": [9.0, 8, 7, 6, 5, 4, 3, 2] + [np.nan] * 6})
    assert gappy["a"].notna().sum() == aligned["a"].notna().sum() == 8
    assert gappy["b"].notna().sum() == aligned["b"].notna().sum() == 8
    g = one(gappy, "Paired t-Test", {"var1": "a", "var2": "b"})
    a = one(aligned, "Paired t-Test", {"var1": "a", "var2": "b"})
    assert g["evidence"]["paired_n"] == 2 and g["status"] == "NOT_READY"
    assert a["evidence"]["paired_n"] == 8 and a["status"] != "NOT_READY"


def test_c1_paired_pair_count_is_recorded_and_normal_semantics_retained():
    n = 12
    df = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float) * 1.5 + (np.arange(n) % 3)})
    s = one(df, "Paired t-Test", {"var1": "a", "var2": "b"})
    assert s["evidence"]["paired_n"] == 12
    assert s["status"] == "NEEDS_ATTENTION"          # usable, small -> caution, not blocked
    big = pd.DataFrame({"a": np.arange(40, dtype=float), "b": np.arange(40, dtype=float) * 1.5 + (np.arange(40) % 3)})
    assert one(big, "Paired t-Test", {"var1": "a", "var2": "b"})["status"] == "READY"


def test_c1_not_every_two_numeric_analysis_is_treated_as_paired():
    df = _gappy_pair_df()
    for tool, mv in (("Pearson Correlation", {"x": "a", "y": "b"}),
                     ("Spearman Correlation", {"x": "a", "y": "b"}),
                     ("Simple Linear Regression", {"y": "b", "x": "a"}),
                     ("Descriptive Statistics", {"variables": ["a", "b"]})):
        s = one(df, tool, mv)
        assert "paired_n" not in s["evidence"], tool
        assert not any("complete pair" in b.lower() for b in s["blocking_issues"]), tool


def test_c1_paired_check_computes_no_inferential_quantity():
    src = inspect.getsource(readiness._layer2_paired_numeric)
    assert "dropna()" in src
    for banned in ("ttest", "wilcoxon(", "stats.", "mean()", "std("):
        assert banned not in src, banned


# ---- Correction 2: Shapiro-Wilk is validated structurally only ----
def test_c2_shapiro_not_needs_attention_from_high_skewness():
    df = pd.DataFrame({"x": [1.0] * 25 + [500, 900, 2000]})
    assert abs(df["x"].skew()) > readiness._Thresholds.HIGH_SKEW_THRESHOLD      # the skew really is high
    s = one(df, "Shapiro-Wilk Normality", {"variable": "x"})
    assert s["status"] == "READY"
    assert not any("skew" in w.lower() for w in s["warnings"])
    assert "skewness" not in s["evidence"]
    assert s["evidence"]["n_total"] == 28 and "missingness_pct" in s["evidence"]   # structural evidence kept


def test_c2_skew_warning_still_fires_for_other_tools_so_suppression_is_targeted():
    vals = [1.0] * 40 + [50, 60, 70, 80, 90, 100, 120, 150, 200, 300]
    df = pd.DataFrame({"v": vals, "g": ["A", "B"] * 25})
    assert abs(df["v"].skew()) > readiness._Thresholds.HIGH_SKEW_THRESHOLD
    s = one(df, "Independent t-Test", {"value": "v", "group": "g"})
    assert any("skew" in w.lower() for w in s["warnings"]) and "skewness" in s["evidence"]


def test_c2_shapiro_structural_blocks_still_apply():
    assert one(pd.DataFrame({"x": [5.0] * 10}), "Shapiro-Wilk Normality", {"variable": "x"})["status"] == "NOT_READY"
    assert one(pd.DataFrame({"x": [1.0, 2.0]}), "Shapiro-Wilk Normality", {"variable": "x"})["status"] == "NOT_READY"
    assert one(pd.DataFrame({"x": [None] * 5}), "Shapiro-Wilk Normality", {"variable": "x"})["status"] == "NOT_READY"


# ---- Correction 3: Levene's is validated structurally only ----
def _imbalanced_small_groups_df():
    return pd.DataFrame({"v": list(range(1, 16)), "g": ["A"] * 13 + ["B"] * 2})


def test_c3_levene_not_needs_attention_from_group_imbalance_or_small_groups():
    df = _imbalanced_small_groups_df()
    gen = one(df, "Independent t-Test", {"value": "v", "group": "g"})
    assert gen["status"] == "NEEDS_ATTENTION"                 # the generic layer DOES flag this data
    assert any("imbalanc" in w.lower() for w in gen["warnings"])
    lev = one(df, "Levene's Test", {"value": "v", "group": "g"})
    assert lev["status"] == "READY"
    assert lev["warnings"] == []
    assert lev["evidence"]["group_sizes"] == {"A": 13, "B": 2}        # structural evidence retained
    assert "variance_ratio" not in lev["evidence"]                     # no proxy for Levene's own result
    assert "skewness" not in lev["evidence"]


def test_c3_levene_structural_blocks_still_apply():
    df = pd.DataFrame({"v": [5.0] * 10, "g": ["A", "B"] * 5})
    assert one(df, "Levene's Test", {"value": "v", "group": "g"})["status"] == "NOT_READY"
    assert one(_gc_df(), "Levene's Test", {"value": "missing_col", "group": "gender"})["status"] == "NOT_READY"


# ---- Correction 4: Fisher's Exact Test does not get the low-expected-cell warning ----
def _small_table_df():
    return pd.DataFrame({"a": ["X"] * 6 + ["Y"] * 4, "b": ["P", "Q"] * 5})


def test_c4_chi_square_keeps_low_expected_cell_warning():
    s = one(_small_table_df(), "Chi-Square Test of Independence", {"var1": "a", "var2": "b"})
    assert s["evidence"]["cells_below_expected_min"] > 0
    assert any("expected count" in w.lower() for w in s["warnings"])
    assert s["status"] == "NEEDS_ATTENTION"


def test_c4_fisher_does_not_get_the_warning_but_keeps_descriptive_evidence():
    s = one(_small_table_df(), "Fisher's Exact Test", {"var1": "a", "var2": "b"})
    assert s["evidence"]["cells_below_expected_min"] > 0            # numbers still reported
    assert "min_expected_count" in s["evidence"]
    assert not any("expected count" in w.lower() for w in s["warnings"])
    assert s["status"] == "READY"


def test_c4_not_a_broad_categorical_pair_exception():
    s = one(_small_table_df(), "Cramér's V", {"var1": "a", "var2": "b"})
    assert any("expected count" in w.lower() for w in s["warnings"])
    fisher_only = [t for t, o in readiness._LAYER_3_OVERRIDES.items() if "expected_cell_warning" in o["suppress"]]
    assert fisher_only == ["Fisher's Exact Test"]


# ---- Correction 5: type-resolution consistency (verified, unchanged) ----
def test_c5_resolver_agrees_with_profiler_on_every_pipeline_reachable_type():
    df = pd.DataFrame({
        "num_int": [1, 2, 3, 4, 5, 6],
        "num_float": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        "flag": [True, False, True, False, True, False],
        "grp_str": ["A", "B", "A", "B", "A", "B"],
        "grp_obj": pd.Series(["X", "Y", "X", "Y", "X", "Y"], dtype="object"),
        "when": pd.to_datetime(["2024-01-0%d" % i for i in range(1, 7)]),
    })
    profiled = classify_columns(df)
    for col in df.columns:
        assert readiness._resolve_column_type(df, col) == profiled[col], col


def test_c5_known_divergences_exist_but_are_unreachable_through_the_pipeline():
    # documented divergence: profiler distinguishes free 'text' from 'categorical'; readiness has no such bucket
    hi_card = pd.DataFrame({"c": [f"unique comment number {i}" for i in range(60)]})
    assert classify_columns(hi_card)["c"] == "text"
    assert readiness._resolve_column_type(hi_card, "c") == "categorical"
    # ...but no registered tool role accepts 'text', so such a column can never be placed in
    # matched_variables by candidates.py / analysis_plan.py in the first place.
    accepts_text = [(n, r) for n, s in STAT_TOOLS.items() for r, d in s["roles"].items() if "text" in d["types"]]
    assert accepts_text == []


# ---- Correction 6: no blanket "n < 30" warning ----
def test_c6_layer1_no_longer_emits_a_total_n_warning():
    """Behavioral + static replacement for the original source-string
    assertion (which tested comment text inside _layer1_generic rather
    than actual behavior, and would have stayed green even if the
    blanket <30 warning were reintroduced under a different variable
    name or moved into a comment-free rewrite)."""
    # 1. Behavioral: _layer1_generic itself produces no warning and no
    #    blocking issue for an otherwise-usable n=12 single-variable
    #    step — n=12 is below 30, so if the old blanket rule were still
    #    active here, this would fail.
    df = pd.DataFrame({"x": np.arange(12, dtype=float) + (np.arange(12) % 4)})
    l1 = readiness._layer1_generic(df, {"variable": "x"})
    assert l1["warnings"] == [] and l1["blocking_issues"] == []
    assert l1["evidence"]["n_total"] == 12

    # 2. Behavioral, at the full validate_plan level, across several
    #    step shapes that are NOT relationship-shaped (x/y) — none of
    #    these should see a rule-of-thumb warning at n=12, confirming
    #    the removal is real and not merely a single lucky case.
    for tool, mv in (("Descriptive Statistics", {"variables": ["x"]}),
                     ("Shapiro-Wilk Normality", {"variable": "x"})):
        s = one(df, tool, mv)
        assert s["status"] == "READY", (tool, s["warnings"])
        assert not any("rule-of-thumb" in w or "30" in w for w in s["warnings"]), (tool, s["warnings"])

    # 3. Static: _layer1_generic no longer REFERENCES the threshold
    #    constant in its executable body at all (AST-based — checks
    #    actual Name/Attribute nodes used in the function, not comment
    #    text, so this can't be fooled by rewording a comment and can't
    #    be broken by one either).
    tree = ast.parse(inspect.getsource(readiness._layer1_generic))
    referenced_attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "MIN_TOTAL_N_RULE_OF_THUMB" not in referenced_attrs

    # 4. The heuristic itself still exists elsewhere, correctly
    #    contextualized to the relationship-shaped (x/y) family, and is
    #    actually reachable/used from there (already exercised
    #    end-to-end by test_c6_retained_heuristic_is_contextual_and_
    #    worded_as_a_heuristic, re-confirmed narrowly here).
    assert hasattr(readiness._Thresholds, "MIN_TOTAL_N_RULE_OF_THUMB")
    dispatch_src = inspect.getsource(readiness._dispatch_layer2)
    assert "_two_numeric_sample_size_warning" in dispatch_src
    assert '"x", "y"' in dispatch_src  # gated on the x/y role-key shape specifically


def test_c6_unrelated_analyses_with_n_below_30_are_not_flagged_solely_on_n():
    n = 12
    df = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float) * 2 + (np.arange(n) % 3),
                       "c": np.arange(n, dtype=float) % 5 + np.arange(n) * 0.1})
    for tool, mv in (("Descriptive Statistics", {"variables": ["a"]}),
                     ("Descriptive Statistics", {"variables": ["a", "b"]}),
                     ("Descriptive Statistics", {"variables": ["a", "b", "c"]}),
                     ("K-Means Clustering", {"variables": ["a", "b"]}),
                     ("Shapiro-Wilk Normality", {"variable": "a"})):
        s = one(df, tool, mv)
        assert s["status"] == "READY", (tool, mv, s["warnings"])
        assert not any("rule-of-thumb" in w for w in s["warnings"]), (tool, mv)
    # a categorical table with adequate expected counts and n=24 < 30
    tbl = pd.DataFrame({"a": ["X"] * 12 + ["Y"] * 12, "b": (["P"] * 6 + ["Q"] * 6) * 2})
    s = one(tbl, "Chi-Square Test of Independence", {"var1": "a", "var2": "b"})
    assert s["evidence"]["cells_below_expected_min"] == 0 and s["status"] == "READY"


def test_c6_hard_minimum_still_blocks():
    assert one(pd.DataFrame({"x": [1.0, 2.0]}), "Descriptive Statistics", {"variables": ["x"]})["status"] == "NOT_READY"
    assert one(pd.DataFrame({"x": [None] * 4}), "Descriptive Statistics", {"variables": ["x"]})["status"] == "NOT_READY"


def test_c6_retained_heuristic_is_contextual_and_worded_as_a_heuristic():
    n = 12
    df = pd.DataFrame({"a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float) * 2 + (np.arange(n) % 3)})
    s = one(df, "Pearson Correlation", {"x": "a", "y": "b"})
    assert s["status"] == "NEEDS_ATTENTION"
    msg = " ".join(s["warnings"]).lower()
    assert "not a requirement" in msg and "heuristic" in msg and "power analysis" in msg
    big = pd.DataFrame({"a": np.arange(40, dtype=float), "b": np.arange(40, dtype=float) * 2 + (np.arange(40) % 3)})
    assert one(big, "Pearson Correlation", {"x": "a", "y": "b"})["status"] == "READY"


def test_c6_thresholds_are_centralized():
    """There is exactly one place readiness thresholds live
    (_Thresholds), and every sample-size / imbalance / expected-count
    threshold used by the readiness checks is a named attribute on it
    rather than a second, separate constant defined elsewhere."""
    src = open(os.path.join(HERE, "readiness.py")).read()
    tree = ast.parse(src)
    threshold_classes = [n for n in ast.walk(tree)
                         if isinstance(n, ast.ClassDef) and "Threshold" in n.name]
    assert len(threshold_classes) == 1, threshold_classes
    names = {n.targets[0].id for n in threshold_classes[0].body if isinstance(n, ast.Assign)}
    assert {"MIN_USABLE_N_HARD", "MIN_GROUP_N_RULE_OF_THUMB", "MIN_TOTAL_N_RULE_OF_THUMB",
            "GROUP_IMBALANCE_RATIO_WARNING", "MIN_EXPECTED_CELL_COUNT",
            "HIGH_SKEW_THRESHOLD"} <= names
    # No second, separate module-level numeric constant duplicates a
    # threshold's purpose (scoped to the module's TOP-LEVEL statements
    # only — ast.walk would also match ordinary local variables like
    # `ok = True` inside functions, which is not what this checks for).
    module_level_numeric_consts = {
        n.targets[0].id for n in tree.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and isinstance(n.value, ast.Constant) and isinstance(n.value.value, (int, float))
    }
    assert not module_level_numeric_consts, module_level_numeric_consts


def test_no_inferential_evidence_across_all_correction_scenarios():
    scenarios = [
        run(_gappy_pair_df(), step("Paired t-Test", {"var1": "a", "var2": "b"})),
        run(pd.DataFrame({"x": [1.0] * 25 + [500, 900, 2000]}), step("Shapiro-Wilk Normality", {"variable": "x"})),
        run(_imbalanced_small_groups_df(), step("Levene's Test", {"value": "v", "group": "g"})),
        run(_small_table_df(), step("Fisher's Exact Test", {"var1": "a", "var2": "b"})),
        run(_small_table_df(), step("Chi-Square Test of Independence", {"var1": "a", "var2": "b"})),
    ]
    for v in scenarios:
        assert_no_inferential_evidence(v)


# ================================================================
# 4. PROTECTED FILES: BYTE-IDENTITY
# ================================================================
PROTECTED_MD5 = {
    "app.py": "d73f932c8d8e9228d5eed1a1953a978c",
    "theme.py": "f1c8aa2db2fef06efa95f07d406ffc47",
    "io_layer.py": "485055f91e4624d0a57cbc7f00a5a0b1",
    "profiler.py": "f394861140a82fdac66ed6f8d7f94727",
    "cleaning.py": "323c87ff89c66c1cd57dc81831e524d5",
    "tool_registry.py": "bf1de5b1ec09b538053fe0e7d06d8513",
    "recommender.py": "ea76aee2407bb79e87ae753457cb4166",
    "validator.py": "37825aae77833aca52e6bd28148b7b1c",
    "stats_engine.py": "9e88527ca53afe2b15b8f1e4ac3e7f67",
    "graph_engine.py": "89864281211c58876307458fe24e986d",
    "interpreter.py": "b4148928be934816a5294ec8e46f397f",
    "export.py": "26f6a86d4b0d78afa05c9166d7e88495",
    "objectives.py": "9be6d40d82b2408b7910bd0bf4d21350",
    "candidates.py": "634b91f29cd3918a8f7ac354a9634a89",
    "analysis_plan.py": "a6df2d0a8bd4b9f82a469d6008b5bff8",
}


def test_protected_files_byte_identical():
    for name, expected in PROTECTED_MD5.items():
        actual = hashlib.md5(open(os.path.join(HERE, name), "rb").read()).hexdigest()
        assert actual == expected, f"{name} changed: {actual}"


def test_step5_is_unwired():
    for name in os.listdir(HERE):
        if name.endswith(".py") and name not in ("readiness.py", "test_readiness.py"):
            mods = _imports(os.path.join(HERE, name))
            assert "readiness" not in mods, f"{name} imports readiness"


# ----------------------------------------------------------------
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"[PASS] {name}")
        except Exception as e:  # noqa: BLE001
            failed.append((name, repr(e)))
            print(f"[FAIL] {name}: {e!r}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)

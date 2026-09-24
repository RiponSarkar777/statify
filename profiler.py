"""
Statify — Data Profiler
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Column-type detection and dataset-level summary.
Returns a dict mapping each column to one of:
'numeric', 'categorical', 'datetime', 'boolean', 'text'.
Moved out of app.py — behavior unchanged.
"""
import pandas as pd


def _is_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == object:
        sample = series.dropna().astype(str).head(20)
        if len(sample) == 0:
            return False
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            return parsed.notna().mean() > 0.7
        except Exception:
            return False
    return False


def classify_columns(df: pd.DataFrame) -> dict:
    """Map each column to a semantic type."""
    types = {}
    for col in df.columns:
        s = df[col]
        # NOTE (fix): pandas 3.x introduced a dedicated string dtype
        # distinct from legacy `object`. The original check here was
        # `s.dtype == object`, which silently stopped matching ordinary
        # text/categorical columns on pandas >= 3.0 (they'd all fall
        # through to "text"). is_object_dtype/is_string_dtype together
        # cover both the legacy and modern cases. This is the only
        # change in this function — thresholds and branching order are
        # otherwise unchanged from the original.
        is_stringlike = pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
        if pd.api.types.is_bool_dtype(s):
            types[col] = "boolean"
        elif _is_datetime(s):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            types[col] = "numeric"
        elif s.nunique(dropna=True) <= max(20, int(0.05 * len(s))) and is_stringlike:
            types[col] = "categorical"
        elif is_stringlike:
            types[col] = "categorical" if s.nunique(dropna=True) <= 50 else "text"
        else:
            types[col] = "text"
    return types


def get_column_types(df: pd.DataFrame) -> dict:
    """Bucket columns by semantic type."""
    cls = classify_columns(df)
    buckets = {"numeric": [], "categorical": [], "datetime": [], "boolean": [], "text": []}
    for col, t in cls.items():
        buckets[t].append(col)
    return buckets


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """Return high-level metadata for the upload page."""
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "buckets": get_column_types(df),
    }


# ============================================================
# DATASET CAPABILITY MAP  (new — structural check only)
# ============================================================
# Answers one narrow question: "could this dataset structurally
# support at least one tool in each analysis category?" This is
# NOT recommendation or ranking — it does not pick a best tool, it
# does not consider the user's purpose, and it does not check
# statistical readiness (sample size, missingness, assumptions).
# Those are later, separate stages. This only checks the same kind
# of structural contract recommender.py's compatible_tools_for()
# already checks per-tool, rolled up per-category, plus one addition:
# a minimal cardinality check so a categorical/boolean column with
# only one distinct value doesn't falsely count as "supported" for
# anything requiring group comparison.

MIN_CATEGORY_LEVELS = 2  # a categorical/boolean column needs at least
                           # this many distinct non-null values to be
                           # usable for a role that needs grouping


def _usable_categorical_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Filter a list of categorical/boolean column names down to those
    with enough distinct non-null values to actually form groups."""
    usable = []
    for col in columns:
        if col not in df.columns:
            continue
        n_levels = df[col].dropna().nunique()
        if n_levels >= MIN_CATEGORY_LEVELS:
            usable.append(col)
    return usable


def _category_supported(df: pd.DataFrame, buckets: dict, roles: dict) -> bool:
    """Check whether one tool's role contract can structurally be
    satisfied by this dataset's buckets. Same bucket/min/max/multi/
    optional logic as recommender.py's compatible_tools_for(), plus
    a cardinality check for categorical/boolean roles."""
    for r in roles.values():
        if r.get("optional"):
            continue
        needs_grouping = bool({"categorical", "boolean"} & set(r["types"]))
        if needs_grouping:
            usable_cols = []
            for t in r["types"]:
                if t in ("categorical", "boolean"):
                    usable_cols += _usable_categorical_columns(df, buckets.get(t, []))
                else:
                    usable_cols += buckets.get(t, [])
            if not usable_cols:
                return False
        else:
            if not any(buckets.get(t) for t in r["types"]):
                return False
    return True


def dataset_capability_map(df: pd.DataFrame, buckets: dict | None = None) -> dict:
    """
    For each analysis category declared in tool_registry.py, report
    whether this dataset can structurally support at least one tool
    in that category, and a short plain-language reason.

    This is intentionally narrow: structural possibility only, not
    a recommendation, not a ranking, not a readiness/assumption check.
    """
    from tool_registry import STAT_TOOLS  # local import: profiler.py
                                            # stays independently usable
                                            # even if tool_registry.py
                                            # is unavailable for some
                                            # other caller

    if buckets is None:
        buckets = get_column_types(df)

    categories: dict[str, list[str]] = {}
    for tool_name, spec in STAT_TOOLS.items():
        categories.setdefault(spec["category"], []).append(tool_name)

    capability_map = {}
    for category, tool_names in categories.items():
        supporting_tools = []
        for tool_name in tool_names:
            spec = STAT_TOOLS[tool_name]
            if _category_supported(df, buckets, spec["roles"]):
                supporting_tools.append(tool_name)

        supported = len(supporting_tools) > 0
        if supported:
            reason = f"{len(supporting_tools)} of {len(tool_names)} tool(s) in this category have their basic requirements met."
        else:
            reason = "No tool in this category has its basic column-type requirements met by this dataset."

        capability_map[category] = {
            "supported": supported,
            "reason": reason,
            "supporting_tools": supporting_tools,
        }

    return capability_map
    

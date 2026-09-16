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
        if pd.api.types.is_bool_dtype(s):
            types[col] = "boolean"
        elif _is_datetime(s):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            types[col] = "numeric"
        elif s.nunique(dropna=True) <= max(20, int(0.05 * len(s))) and s.dtype == object:
            types[col] = "categorical"
        elif s.dtype == object:
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

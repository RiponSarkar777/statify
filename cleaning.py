"""
Statify — Data Cleaning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pure-function data cleaning operations (return new DataFrame).
Moved out of app.py — behavior unchanged.
"""
import numpy as np
import pandas as pd


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

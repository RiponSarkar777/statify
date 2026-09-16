"""
Statify — File Loader
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-format file loader. Supports CSV, TSV, XLSX, XLS, JSON, SPSS (.sav).
Moved out of app.py — behavior unchanged.
"""
import io
import json

import pandas as pd


def detect_file_type(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _read_csv_smart(buffer) -> pd.DataFrame:
    raw = buffer.read()
    for enc in ("utf-8", "latin1", "cp1252"):
        for sep in (",", ";", "\t", "|"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep, engine="python")
                if df.shape[1] >= 2:
                    return df
            except Exception:
                continue
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8", engine="python")


def load_file(uploaded_file) -> tuple[pd.DataFrame | None, str, str]:
    if uploaded_file is None:
        return None, "error", "No file provided."
    ext = detect_file_type(uploaded_file.name)
    try:
        uploaded_file.seek(0)
        if ext == "csv":
            df = _read_csv_smart(uploaded_file)
        elif ext == "tsv":
            df = pd.read_csv(uploaded_file, sep="\t")
        elif ext == "xlsx":
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        elif ext == "xls":
            df = pd.read_excel(uploaded_file, engine="xlrd")
        elif ext == "json":
            data = json.loads(uploaded_file.read().decode("utf-8"))
            df = pd.json_normalize(data) if isinstance(data, list) else pd.json_normalize([data])
        elif ext == "sav":
            try:
                import pyreadstat
                tmp_path = f"/tmp/_statify_{uploaded_file.name}"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_file.read())
                df, _ = pyreadstat.read_sav(tmp_path)
            except ImportError:
                return None, "error", "SPSS support requires pyreadstat. Run: pip install pyreadstat"
        else:
            return None, "error", f"Unsupported file format: .{ext}"
    except Exception as e:
        return None, "error", f"Could not read file: {e}"

    if df is None or df.empty:
        return None, "error", "File is empty."
    df.columns = [str(c).strip() for c in df.columns]
    df = df.reset_index(drop=True)
    return df, "success", f"Loaded {len(df):,} rows × {len(df.columns)} columns"

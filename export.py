"""
Statify — Multi-Format Report Exporter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generates downloadable reports from a finished analysis: PDF
(reportlab), Excel (openpyxl via pandas), HTML (string), and a ZIP
of chart PNGs. Each function takes the same plain data (analysis
dict, result dict, interpretation dict, list of figures) that the
Results page already has in hand — no dependency on any other
extracted module.

Moved out of app.py — behavior unchanged. No formatting altered.

Note: several functions do their own local imports (reportlab,
base64) inside the function body — unchanged from the original,
intentional lazy-loading, not an omission. _fig_to_png_bytes
silently returns None if kaleido is missing (unchanged try/except
behavior) — this is not new error-swallowing, it matches the
original file exactly.
"""
import io
import zipfile
import pandas as pd
from datetime import datetime


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
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=20, spaceAfter=6)
    h2 = ParagraphStyle("H2X", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body = styles["BodyText"]

    story = []
    story.append(Paragraph(analysis.get("title","Statistical Analysis"), title_style))
    story.append(Paragraph(f"Tool: {analysis.get('tool','')}", body))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Interpretation", h2))
    for key in ["headline","explanation","statistical","assumptions","conclusion"]:
        val = interpretation.get(key)
        if val:
            story.append(Paragraph(val, body))
            story.append(Spacer(1, 6))

    for tname, tdf in result.get("tables", {}).items():
        story.append(Paragraph(tname, h2))
        if hasattr(tdf, "columns"):
            data = [list(tdf.columns)] + tdf.astype(str).values.tolist()
            data = data[:200]  # cap rows
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#7C5CFF")),
                ("TEXTCOLOR", (0,0), (-1,0), colors.white),
                ("FONTSIZE", (0,0), (-1,-1), 7),
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F5FA")]),
            ]))
            story.append(t)
            story.append(Spacer(1, 10))

    if figures:
        story.append(PageBreak())
        story.append(Paragraph("Charts", h2))
        for title, fig in figures:
            try:
                img_bytes = _fig_to_png_bytes(fig)
                if img_bytes:
                    story.append(Paragraph(title, body))
                    story.append(Image(io.BytesIO(img_bytes), width=16*cm, height=9*cm))
                    story.append(Spacer(1, 10))
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
    parts = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{analysis.get('title','Report')}</title>
<style>
body {{ font-family: -apple-system, Inter, sans-serif; background:#0d0f17; color:#ECEEF6; padding:2rem; max-width:900px; margin:auto; }}
h1 {{ color:#7C5CFF; }}
h2 {{ border-bottom:1px solid #232636; padding-bottom:6px; margin-top:2rem; }}
table {{ border-collapse:collapse; width:100%; margin:1rem 0; font-size:0.85rem; }}
th,td {{ border:1px solid #232636; padding:6px 10px; text-align:left; }}
th {{ background:#7C5CFF; color:white; }}
tr:nth-child(even) {{ background:#161925; }}
.card {{ background:#161925; border:1px solid #232636; border-radius:10px; padding:1rem; margin:0.5rem 0; }}
</style></head><body>
<h1>{analysis.get('title','Statistical Report')}</h1>
<p><b>Tool:</b> {analysis.get('tool','')}<br><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<h2>Interpretation</h2>
"""]
    for key in ["headline","explanation","statistical","assumptions","conclusion"]:
        val = interpretation.get(key)
        if val:
            parts.append(f'<div class="card">{val}</div>')

    for tname, tdf in result.get("tables", {}).items():
        parts.append(f"<h2>{tname}</h2>")
        if hasattr(tdf, "to_html"):
            parts.append(tdf.to_html(index=False, border=0))

    if figures:
        parts.append("<h2>Charts</h2>")
        for title, fig in figures:
            try:
                if isinstance(fig, dict) and "_img" in fig:
                    parts.append(f'<div class="card"><h3>{title}</h3><img src="{fig["_img"]}" style="max-width:100%;"/></div>')
                else:
                    parts.append(f'<div class="card"><h3>{title}</h3>{fig.to_html(full_html=False, include_plotlyjs="cdn")}</div>')
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

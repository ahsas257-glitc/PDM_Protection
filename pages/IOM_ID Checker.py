import time
import re
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import altair as alt

from gspread.exceptions import WorksheetNotFound, APIError

from design.theme import load_css, page_header
from design.components import kpi
from services.streamlit_gsheets import get_worksheet, get_spreadsheet, retry_api


# ============================================================
# Page Config
# ============================================================
st.set_page_config(page_title="IOM ID Checker | PDM", page_icon="🆔", layout="wide")
load_css()
page_header("🆔 IOM ID Checker", "Resolve Raw Kobo IDs against Sample (IOMID/CaseNumber) + fallback by Name/Father/Province")

# ============================================================
# Theme (PDM)
# ============================================================
PDM_COLORS = {
    "teal": "#14B8A6",
    "blue": "#3B82F6",
    "indigo": "#6366F1",
    "violet": "#8B5CF6",
    "pink": "#EC4899",
    "amber": "#F59E0B",
    "red": "#EF4444",
    "green": "#10B981",
    "slate": "#64748B",
    "gray": "#9CA3AF",
}

# ============================================================
# Altair Global Theme
# ============================================================
def _pdm_altair_theme():
    return {
        "config": {
            "background": "transparent",
            "view": {"stroke": "transparent"},
            "font": "Inter",
            "axis": {
                "labelColor": "#94A3B8",
                "titleColor": "#CBD5E1",
                "gridColor": "rgba(148,163,184,0.12)",
                "domainColor": "rgba(148,163,184,0.18)",
                "tickColor": "rgba(148,163,184,0.18)",
                "labelFontSize": 12,
                "titleFontSize": 12,
                "titleFontWeight": 700,
            },
            "legend": {
                "labelColor": "#CBD5E1",
                "titleColor": "#E2E8F0",
                "labelFontSize": 12,
                "titleFontSize": 12,
                "titleFontWeight": 800,
                "symbolType": "circle",
                "symbolSize": 120,
            },
            "title": {
                "color": "#E2E8F0",
                "fontSize": 14,
                "fontWeight": 900,
                "anchor": "start",
                "offset": 10,
            },
            "tooltip": {
                "content": "data",
                "fill": "rgba(2,6,23,0.96)",
                "stroke": "rgba(148,163,184,0.22)",
                "color": "#E2E8F0",
            },
        }
    }

alt.themes.register("pdm_modern_iomid", _pdm_altair_theme)
alt.themes.enable("pdm_modern_iomid")

# ============================================================
# Secrets + Client
# ============================================================
try:
    sa = dict(st.secrets["gcp_service_account"])
    spreadsheet_id = st.secrets["app"]["spreadsheet_id"]
except Exception:
    st.error("Secrets are not configured. Please set .streamlit/secrets.toml (service account + spreadsheet_id).")
    st.stop()

# ============================================================
# Sheet Names + Column Names
# ============================================================
RAW_SHEET = "Raw_Kobo_Data"
SAMPLE_SHEET = "Sample"
NOT_FOUND_SHEET = "IOM_Not_found"

RAW_IOM_COL = "A.7. IOM ID Number"
RAW_DATE_COL = "start"
RAW_INTERVIEWER_COL = "A.2. Interviewer name"
RAW_PROVINCE_COL = "A.8. Province"

# NEW: Raw name columns (for fallback search)
RAW_NAME_COL = "A.4.1. Name of Respondent"
RAW_FATHER_COL = "a.4.2. Father name of Respondent"

# Sample columns
SAMPLE_IOM_COL = "IOMID"
SAMPLE_CASE_COL = "CaseNumber"

# NEW: sample name columns (for fallback search)
SAMPLE_NAME_COL = "Name"
SAMPLE_FATHER_COL = "FatherName"
SAMPLE_PROVINCE_COL = "Province"

# NEW: output columns we will write back to Raw sheet
RAW_RESOLVED_IOM_COL = "Resolved_IOMID"
RAW_RESOLUTION_METHOD_COL = "Resolution_Method"
RAW_RESOLUTION_NOTE_COL = "Resolution_Note"

# ============================================================
# Utilities (robust + fast)
# ============================================================
def normalize_text(x) -> str:
    """Trim, collapse spaces, uppercase. None/blank -> ''."""
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    s = " ".join(s.split())
    return s.upper()

def digits_only(x) -> str:
    """Extract digits from value. None/blank -> ''."""
    if x is None:
        return ""
    s = str(x)
    if s.lower() in {"", "nan", "none"}:
        return ""
    d = re.sub(r"\D+", "", s)
    return d

def normalize_name(x) -> str:
    """
    Normalize names for matching:
    - trim/collapse spaces
    - uppercase
    - remove some punctuation
    """
    s = normalize_text(x)
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = " ".join(s.split())
    return s

def _dedupe_headers(headers: list[str]) -> list[str]:
    seen = {}
    out = []
    for h in headers:
        h = str(h).strip()
        if h in seen:
            seen[h] += 1
            out.append(f"{h}__{seen[h]}")
        else:
            seen[h] = 0
            out.append(h)
    return out

def _retry(fn, tries=3, sleep_s=0.6):
    last = None
    for i in range(tries):
        try:
            return fn()
        except APIError as e:
            last = e
            time.sleep(sleep_s * (i + 1))
        except Exception as e:
            last = e
            time.sleep(sleep_s * (i + 1))
    raise last

@st.cache_data(ttl=120, show_spinner=False)
def load_sheet_df(sheet_name: str) -> pd.DataFrame:
    """
    Fast + stable Google Sheet loader (cached).
    Adds a helper column `_row` = actual Google Sheet row number (1-based).
    """
    ws = get_worksheet(sa, spreadsheet_id, sheet_name)
    values = retry_api(ws.get_all_values)
    if not values:
        return pd.DataFrame()

    headers = _dedupe_headers(values[0])
    data = values[1:]
    if not data:
        df = pd.DataFrame(columns=headers)
        df["_row"] = pd.Series(dtype="int64")
        return df

    df = pd.DataFrame(data, columns=headers)
    # Row number in sheet: header is row 1, first data row is row 2
    df["_row"] = list(range(2, 2 + len(df)))
    return df

def get_or_create_ws(title: str, rows: int = 5000, cols: int = 30):
    """Return worksheet handle (create if missing). Uses cached spreadsheet + retry."""
    _, sh = get_spreadsheet(sa, spreadsheet_id)
    try:
        return retry_api(sh.worksheet, title)
    except WorksheetNotFound:
        return retry_api(sh.add_worksheet, title=title, rows=rows, cols=cols)

def chunked(lst, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def a1_col(n: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA ..."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def ensure_headers(ws, required_cols: list[str]) -> dict[str, int]:
    """
    Ensure the given columns exist in header row.
    Returns mapping col_name -> 1-based column index.
    """
    header = _retry(lambda: ws.row_values(1))
    header = [str(x).strip() for x in header]
    header_map = {h: i + 1 for i, h in enumerate(header) if h != ""}

    to_add = [c for c in required_cols if c not in header_map]
    if to_add:
        new_header = header + to_add
        # Update whole header row up to new length
        end_col = a1_col(len(new_header))
        _retry(lambda: ws.update(f"A1:{end_col}1", [new_header], value_input_option="RAW"))
        header_map = {h: i + 1 for i, h in enumerate(new_header) if h != ""}

    return header_map

# ============================================================
# Modern chart helpers
# ============================================================
def donut_chart(df_: pd.DataFrame, title: str, color_range: list[str]):
    d = df_.copy()
    total = float(d["count"].sum()) if len(d) else 0.0
    d["pct"] = d["count"] / max(1.0, total)

    donut = (
        alt.Chart(d)
        .mark_arc(innerRadius=82, outerRadius=128, cornerRadius=8)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("label:N", scale=alt.Scale(range=color_range), legend=alt.Legend(title="")),
            tooltip=[
                alt.Tooltip("label:N", title=""),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("pct:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=320, title=title)
    )

    match_pct = 0.0
    if len(d) and (d["label"] == "Resolved").any():
        match_pct = float(d.loc[d["label"] == "Resolved", "pct"].iloc[0]) * 100.0
    center = (
        alt.Chart(pd.DataFrame({"t": [f"{match_pct:.0f}%"]}))
        .mark_text(fontSize=28, fontWeight=900, color="#E2E8F0")
        .encode(text="t:N")
    )

    return donut + center

def modern_barh(df_: pd.DataFrame, title: str, x_title: str = "Count", max_rows: int = 20):
    d = df_.copy()
    if d.empty:
        return alt.Chart(pd.DataFrame({"label": [], "count": []})).mark_bar().properties(height=320, title=title)

    d = d.sort_values("count", ascending=False).head(max_rows).sort_values("count", ascending=True)

    return (
        alt.Chart(d)
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
        .encode(
            x=alt.X("count:Q", title=x_title),
            y=alt.Y("label:N", sort=None, title=""),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="turbo"), legend=None),
            tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("count:Q", title="Count")],
        )
        .properties(height=360, title=title)
    )

# ============================================================
# Controls
# ============================================================
st.sidebar.header("Controls")

if st.sidebar.button("Refresh data (clear cache)"):
    st.cache_data.clear()
    st.rerun()

ignore_blank_ids = st.sidebar.checkbox("Ignore blank IOM IDs", value=True)

use_normalization = st.sidebar.checkbox("Normalize IDs (trim + uppercase)", value=True)
use_digits_fallback = st.sidebar.checkbox("Fallback: match by digits only (e.g. MIL432386 → 432386)", value=True)
use_name_fallback = st.sidebar.checkbox("Fallback: match by Name + Father + Province", value=True)

# ============================================================
# Load data (cached)
# ============================================================
try:
    raw_df = load_sheet_df(RAW_SHEET)
except WorksheetNotFound:
    st.error(f"Worksheet '{RAW_SHEET}' was not found.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load '{RAW_SHEET}': {e}")
    st.stop()

try:
    sample_df = load_sheet_df(SAMPLE_SHEET)
except WorksheetNotFound:
    st.error(f"Worksheet '{SAMPLE_SHEET}' was not found.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load '{SAMPLE_SHEET}': {e}")
    st.stop()

if raw_df.empty:
    st.warning("Raw_Kobo_Data has no data.")
    st.stop()

if sample_df.empty:
    st.warning("Sample sheet has no data.")
    st.stop()

# Required columns (IDs + sample search columns)
missing_cols = []
if RAW_IOM_COL not in raw_df.columns:
    missing_cols.append(f"{RAW_SHEET}: '{RAW_IOM_COL}'")

for c in [SAMPLE_IOM_COL, SAMPLE_CASE_COL]:
    if c not in sample_df.columns:
        missing_cols.append(f"{SAMPLE_SHEET}: '{c}'")

if use_name_fallback:
    for c in [RAW_NAME_COL, RAW_FATHER_COL, RAW_PROVINCE_COL]:
        if c not in raw_df.columns:
            missing_cols.append(f"{RAW_SHEET}: '{c}' (needed for Name fallback)")
    for c in [SAMPLE_NAME_COL, SAMPLE_FATHER_COL, SAMPLE_PROVINCE_COL]:
        if c not in sample_df.columns:
            missing_cols.append(f"{SAMPLE_SHEET}: '{c}' (needed for Name fallback)")

if missing_cols:
    st.error("Required columns are missing:\n\n" + "\n".join([f"- {x}" for x in missing_cols]))
    st.stop()

# ============================================================
# Optional date filter (safe)
# ============================================================
filtered_raw = raw_df.copy()
date_range_applied = False

if RAW_DATE_COL in raw_df.columns:
    raw_dt = pd.to_datetime(raw_df[RAW_DATE_COL], errors="coerce")
    if raw_dt.notna().any():
        filtered_raw[RAW_DATE_COL] = raw_dt

        min_d = raw_dt.dropna().min().date()
        max_d = raw_dt.dropna().max().date()

        date_range = st.sidebar.date_input(
            "Date range (Raw data)",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_d, end_d = date_range
            mask = (filtered_raw[RAW_DATE_COL].dt.date >= start_d) & (filtered_raw[RAW_DATE_COL].dt.date <= end_d)
            filtered_raw = filtered_raw.loc[mask].copy()
            date_range_applied = True

# ============================================================
# Build lookup indexes from Sample
# ============================================================
# Normalize ID columns
sample_iom = sample_df[SAMPLE_IOM_COL].astype(str).fillna("")
sample_case = sample_df[SAMPLE_CASE_COL].astype(str).fillna("")

if use_normalization:
    sample_df["_iom_norm"] = sample_iom.map(normalize_text)
    sample_df["_case_norm"] = sample_case.map(normalize_text)
else:
    sample_df["_iom_norm"] = sample_iom.str.strip()
    sample_df["_case_norm"] = sample_case.str.strip()

if use_digits_fallback:
    sample_df["_iom_digits"] = sample_df[SAMPLE_IOM_COL].map(digits_only)
    sample_df["_case_digits"] = sample_df[SAMPLE_CASE_COL].map(digits_only)
else:
    sample_df["_iom_digits"] = ""
    sample_df["_case_digits"] = ""

# Maps:
# - iom_norm -> iom_norm (canonical)
iom_norm_set = set(sample_df["_iom_norm"].dropna().astype(str).tolist())

# - case_norm -> iom_norm (if multiple, keep first but we also track ambiguity)
case_to_iom = {}
case_amb = set()
for _, r in sample_df[["._row" if "._row" in sample_df.columns else "_row", "_case_norm", "_iom_norm"]].iterrows():
    c = r["_case_norm"]
    i = r["_iom_norm"]
    if c == "":
        continue
    if c in case_to_iom and case_to_iom[c] != i:
        case_amb.add(c)
    else:
        case_to_iom[c] = i

# Digits maps
iom_digits_to_iom = {}
iom_digits_amb = set()
case_digits_to_iom = {}
case_digits_amb = set()

if use_digits_fallback:
    for _, r in sample_df[["_iom_digits", "_case_digits", "_iom_norm"]].iterrows():
        idig = r["_iom_digits"]
        cdig = r["_case_digits"]
        iomn = r["_iom_norm"]

        if idig:
            if idig in iom_digits_to_iom and iom_digits_to_iom[idig] != iomn:
                iom_digits_amb.add(idig)
            else:
                iom_digits_to_iom[idig] = iomn

        if cdig:
            if cdig in case_digits_to_iom and case_digits_to_iom[cdig] != iomn:
                case_digits_amb.add(cdig)
            else:
                case_digits_to_iom[cdig] = iomn

# Name fallback key map
namekey_to_ioms = {}
if use_name_fallback:
    s_name = sample_df[SAMPLE_NAME_COL].map(normalize_name)
    s_fath = sample_df[SAMPLE_FATHER_COL].map(normalize_name)
    s_prov = sample_df[SAMPLE_PROVINCE_COL].map(normalize_text)
    sample_df["_name_key"] = (s_name + "||" + s_fath + "||" + s_prov).astype(str)

    for _, r in sample_df[["_name_key", "_iom_norm"]].iterrows():
        k = r["_name_key"]
        v = r["_iom_norm"]
        if k.strip("|") == "":
            continue
        namekey_to_ioms.setdefault(k, set()).add(v)

# ============================================================
# Resolver (per row)
# ============================================================
def resolve_row(raw_id: str, name_key: str | None):
    """
    Returns: (resolved_iom_norm, method, note)
    method: EXACT_IOM, CASE_TO_IOM, DIGITS_IOM, DIGITS_CASE, NAME_FALLBACK, UNRESOLVED, AMBIGUOUS
    """
    rid = raw_id if raw_id is not None else ""

    # Clean
    rid_norm = normalize_text(rid) if use_normalization else str(rid).strip()
    rid_digits = digits_only(rid) if use_digits_fallback else ""

    if ignore_blank_ids and rid_norm == "":
        return ("", "SKIP_BLANK", "blank id ignored")

    # 1) exact match in IOMID
    if rid_norm != "" and rid_norm in iom_norm_set:
        return (rid_norm, "EXACT_IOM", "")

    # 2) exact match in CaseNumber -> get IOMID
    if rid_norm != "" and rid_norm in case_to_iom:
        if rid_norm in case_amb:
            return ("", "AMBIGUOUS", f"CaseNumber '{rid_norm}' maps to multiple IOMIDs")
        return (case_to_iom[rid_norm], "CASE_TO_IOM", "")

    # 3) digits-only match (IOMID)
    if use_digits_fallback and rid_digits:
        if rid_digits in iom_digits_amb:
            return ("", "AMBIGUOUS", f"Digits '{rid_digits}' match multiple IOMIDs")
        if rid_digits in iom_digits_to_iom:
            return (iom_digits_to_iom[rid_digits], "DIGITS_IOM", f"raw='{rid_norm}' digits='{rid_digits}'")

        # 4) digits-only match (CaseNumber -> IOMID)
        if rid_digits in case_digits_amb:
            return ("", "AMBIGUOUS", f"Digits '{rid_digits}' match multiple CaseNumbers")
        if rid_digits in case_digits_to_iom:
            return (case_digits_to_iom[rid_digits], "DIGITS_CASE", f"raw='{rid_norm}' digits='{rid_digits}'")

    # 5) fallback by name/father/province
    if use_name_fallback and name_key:
        ioms = namekey_to_ioms.get(name_key, set())
        if len(ioms) == 1:
            return (next(iter(ioms)), "NAME_FALLBACK", "")
        if len(ioms) > 1:
            return ("", "AMBIGUOUS", "Name/Father/Province matches multiple IOMIDs")

    return ("", "UNRESOLVED", "")

# ============================================================
# Prepare filtered raw + name keys
# ============================================================
filtered_raw["_raw_id"] = filtered_raw[RAW_IOM_COL].astype(str).fillna("")
filtered_raw["_raw_norm"] = filtered_raw["_raw_id"].map(normalize_text) if use_normalization else filtered_raw["_raw_id"].str.strip()
filtered_raw["_raw_digits"] = filtered_raw["_raw_id"].map(digits_only) if use_digits_fallback else ""

if use_name_fallback:
    filtered_raw["_name_key"] = (
        filtered_raw[RAW_NAME_COL].map(normalize_name)
        + "||"
        + filtered_raw[RAW_FATHER_COL].map(normalize_name)
        + "||"
        + filtered_raw[RAW_PROVINCE_COL].map(normalize_text)
    ).astype(str)
else:
    filtered_raw["_name_key"] = None

# Resolve
res = filtered_raw.apply(lambda r: resolve_row(r["_raw_id"], r["_name_key"]), axis=1, result_type="expand")
res.columns = ["resolved_iom_norm", "resolution_method", "resolution_note"]
filtered_raw = pd.concat([filtered_raw, res], axis=1)

# Build views
checked_df = filtered_raw.copy() if not ignore_blank_ids else filtered_raw[filtered_raw["_raw_norm"] != ""].copy()

resolved_df = checked_df[checked_df["resolved_iom_norm"] != ""].copy()
unresolved_df = checked_df[checked_df["resolved_iom_norm"] == ""].copy()

# For display: unresolved unique IDs
unresolved_ids = unresolved_df["_raw_norm"].value_counts().reset_index()
unresolved_ids.columns = ["raw_iom_value", "occurrences"]

# Method breakdown
method_counts = checked_df["resolution_method"].value_counts().reset_index()
method_counts.columns = ["label", "count"]

# ============================================================
# Tabs
# ============================================================
tab_overview, tab_unresolved, tab_resolved, tab_writeback, tab_sync_nf = st.tabs(
    ["Overview", "Unresolved", "Resolved preview", "Write back to Raw_Kobo_Data", "Sync unresolved to IOM_Not_found"]
)

# ------------------ Overview ------------------
with tab_overview:
    total_raw = int(len(raw_df))
    total_in_range = int(len(filtered_raw))
    total_checked = int(len(checked_df))
    total_resolved = int(len(resolved_df))
    total_unresolved = int(len(unresolved_df))

    st.markdown("### Summary")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi("Raw Records", f"{total_raw:,}", "")
    with c2:
        kpi("In Range", f"{total_in_range:,}", "Filtered window" if date_range_applied else "No date filter")
    with c3:
        kpi("Checked Rows", f"{total_checked:,}", "")
    with c4:
        kpi("Resolved", f"{total_resolved:,}", "")
    with c5:
        kpi("Unresolved", f"{total_unresolved:,}", f"Unique: {len(unresolved_ids):,}")
    with c6:
        kpi("Ambiguous", f"{int((checked_df['resolution_method']=='AMBIGUOUS').sum()):,}", "")

    st.markdown("---")

    breakdown = pd.DataFrame(
        [{"label": "Resolved", "count": total_resolved}, {"label": "Unresolved", "count": total_unresolved}]
    )
    st.altair_chart(
        donut_chart(
            breakdown,
            title="Resolution Rate",
            color_range=[PDM_COLORS["green"], PDM_COLORS["red"]],
        ),
        use_container_width=True,
    )

    st.markdown("### Resolution methods")
    st.dataframe(method_counts, use_container_width=True, hide_index=True)

# ------------------ Unresolved ------------------
with tab_unresolved:
    st.markdown("### Unresolved (Unique list)")
    if not unresolved_ids.empty:
        top_missing = unresolved_ids.head(20).rename(columns={"raw_iom_value": "label", "occurrences": "count"})
        st.altair_chart(
            modern_barh(top_missing, "Top unresolved raw ID values", x_title="Occurrences", max_rows=20),
            use_container_width=True,
        )

    st.dataframe(
        unresolved_ids.sort_values("occurrences", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download unresolved IDs (CSV)",
        data=unresolved_ids.to_csv(index=False).encode("utf-8"),
        file_name="iom_unresolved_ids.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("### Unresolved rows (Preview)")
    preview_cols = [RAW_IOM_COL]
    for c in [RAW_NAME_COL, RAW_FATHER_COL, RAW_PROVINCE_COL, RAW_DATE_COL, RAW_INTERVIEWER_COL]:
        if c in unresolved_df.columns:
            preview_cols.append(c)

    prev = unresolved_df[preview_cols + ["resolution_method", "resolution_note", "_row"]].copy()
    st.dataframe(prev.head(400), use_container_width=True, hide_index=True)

# ------------------ Resolved preview ------------------
with tab_resolved:
    st.markdown("### Resolved rows (Preview)")

    preview_cols = [RAW_IOM_COL, "resolved_iom_norm", "resolution_method"]
    for c in [RAW_NAME_COL, RAW_FATHER_COL, RAW_PROVINCE_COL, RAW_DATE_COL]:
        if c in resolved_df.columns:
            preview_cols.append(c)

    st.dataframe(resolved_df[preview_cols + ["_row"]].head(600), use_container_width=True, hide_index=True)

    st.download_button(
        "Download resolved rows (CSV)",
        data=resolved_df[preview_cols + ["_row"]].to_csv(index=False).encode("utf-8"),
        file_name="iom_resolved_rows.csv",
        mime="text/csv",
    )

# ------------------ Write back to Raw_Kobo_Data ------------------
with tab_writeback:
    st.markdown("### Write resolved IOMID back to `Raw_Kobo_Data`")

    st.info(
        "This will write 3 columns into Raw_Kobo_Data:\n"
        f"- `{RAW_RESOLVED_IOM_COL}` (resolved IOMID)\n"
        f"- `{RAW_RESOLUTION_METHOD_COL}`\n"
        f"- `{RAW_RESOLUTION_NOTE_COL}`\n\n"
        "It does NOT overwrite your original raw column. It adds new columns (or updates them)."
    )

    write_only_resolved = st.checkbox("Write only resolved rows (skip unresolved/blank)", value=True)
    write_scope = st.radio(
        "Write scope",
        ["Current filtered range only", "All rows (ignore date filter)"],
        index=0,
        horizontal=True,
    )

    if st.button("Write back now", type="primary"):
        try:
            raw_ws = get_worksheet(sa, spreadsheet_id, RAW_SHEET)

            # Decide scope DF
            if write_scope == "All rows (ignore date filter)":
                # recompute resolution on full raw_df (to keep it consistent)
                dfw = raw_df.copy()
                dfw["_raw_id"] = dfw[RAW_IOM_COL].astype(str).fillna("")
                dfw["_name_key"] = None
                if use_name_fallback:
                    dfw["_name_key"] = (
                        dfw[RAW_NAME_COL].map(normalize_name)
                        + "||"
                        + dfw[RAW_FATHER_COL].map(normalize_name)
                        + "||"
                        + dfw[RAW_PROVINCE_COL].map(normalize_text)
                    ).astype(str)
                res2 = dfw.apply(lambda r: resolve_row(r["_raw_id"], r["_name_key"]), axis=1, result_type="expand")
                res2.columns = ["resolved_iom_norm", "resolution_method", "resolution_note"]
                dfw = pd.concat([dfw, res2], axis=1)
            else:
                dfw = filtered_raw.copy()

            # Rows to write
            if write_only_resolved:
                dfw = dfw[dfw["resolved_iom_norm"] != ""].copy()

            if dfw.empty:
                st.warning("Nothing to write (no rows in selected scope).")
                st.stop()

            # Ensure output headers exist and get col indices
            header_map = ensure_headers(
                raw_ws,
                [RAW_RESOLVED_IOM_COL, RAW_RESOLUTION_METHOD_COL, RAW_RESOLUTION_NOTE_COL],
            )

            col_resolved = header_map[RAW_RESOLVED_IOM_COL]
            col_method = header_map[RAW_RESOLUTION_METHOD_COL]
            col_note = header_map[RAW_RESOLUTION_NOTE_COL]

            # Build batch updates (range per row, 1x3)
            updates = []
            for _, r in dfw[["_row", "resolved_iom_norm", "resolution_method", "resolution_note"]].iterrows():
                row_no = int(r["_row"])
                v1 = str(r["resolved_iom_norm"] or "")
                v2 = str(r["resolution_method"] or "")
                v3 = str(r["resolution_note"] or "")

                a = a1_col(col_resolved)
                c = a1_col(col_note)
                rng = f"{a}{row_no}:{c}{row_no}"
                updates.append({"range": rng, "values": [[v1, v2, v3]]})

            # Chunk for API safety
            for part in chunked(updates, 300):
                _retry(lambda p=part: raw_ws.batch_update(p, value_input_option="RAW"))

            st.success(f"Wrote back {len(updates):,} rows into `{RAW_SHEET}`.")
        except Exception as e:
            st.error(f"Write back failed: {e}")

# ------------------ Sync unresolved to IOM_Not_found ------------------
with tab_sync_nf:
    st.markdown("### Sync unresolved rows to `IOM_Not_found`")

    st.info(
        "This will append only NEW unresolved items to `IOM_Not_found`.\n"
        "It uses the raw value + context (Name/Father/Province) so you can fix Sample later."
    )

    include_context = st.checkbox("Include context columns (name/father/province/date/interviewer)", value=True)

    if st.button("Sync unresolved now", type="primary"):
        try:
            nf_ws = get_or_create_ws(NOT_FOUND_SHEET, rows=5000, cols=40)

            # Load existing not-found sheet
            existing_vals = _retry(lambda: nf_ws.get_all_values())
            if existing_vals:
                ex_headers = _dedupe_headers(existing_vals[0])
                ex_data = existing_vals[1:]
                ex_df = pd.DataFrame(ex_data, columns=ex_headers) if ex_data else pd.DataFrame(columns=ex_headers)
            else:
                ex_df = pd.DataFrame()

            # Existing keys (avoid duplicates)
            existing_keys = set()
            if not ex_df.empty:
                key_col = None
                for cand in ["raw_iom_value", "iom_id", RAW_IOM_COL]:
                    if cand in ex_df.columns:
                        key_col = cand
                        break
                if key_col:
                    existing_keys = set(ex_df[key_col].map(normalize_text).tolist())

            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            base_cols = ["timestamp_utc", "raw_iom_value", "resolution_method", "resolution_note"]
            out = unresolved_df.copy()
            out["raw_iom_value"] = out[RAW_IOM_COL].astype(str).fillna("").map(normalize_text if use_normalization else lambda x: str(x).strip())
            out["timestamp_utc"] = ts

            keep_cols = base_cols.copy()
            if include_context:
                for c in [RAW_NAME_COL, RAW_FATHER_COL, RAW_PROVINCE_COL, RAW_DATE_COL, RAW_INTERVIEWER_COL]:
                    if c in out.columns and c not in keep_cols:
                        keep_cols.append(c)

            out2 = out[keep_cols].copy()
            out2["_k"] = out2["raw_iom_value"].map(normalize_text)
            to_add = out2[~out2["_k"].isin(existing_keys)].drop(columns=["_k"])

            if to_add.empty:
                st.success("No new unresolved rows to add. `IOM_Not_found` is already up to date.")
            else:
                # Ensure header exists
                if not existing_vals:
                    _retry(lambda: nf_ws.append_row(list(to_add.columns), value_input_option="RAW"))

                rows = to_add.fillna("").astype(str).values.tolist()
                for part in chunked(rows, 300):
                    _retry(lambda p=part: nf_ws.append_rows(p, value_input_option="RAW"))

                st.success(f"Synced {len(rows):,} new unresolved rows to `{NOT_FOUND_SHEET}`.")

        except Exception as e:
            st.error(f"Sync failed: {e}")

    st.markdown("---")
    st.markdown("### Current `IOM_Not_found` (Preview)")
    try:
        nf_df = load_sheet_df(NOT_FOUND_SHEET)
        if nf_df.empty:
            st.info("`IOM_Not_found` is empty (or has only headers).")
        else:
            st.dataframe(nf_df.tail(300), use_container_width=True, hide_index=True)
    except WorksheetNotFound:
        st.info("`IOM_Not_found` does not exist yet. Click **Sync unresolved now** to create it.")
    except Exception as e:
        st.warning(f"Could not load `{NOT_FOUND_SHEET}` preview: {e}")

# ============================================================
# Sidebar footer
# ============================================================
with st.sidebar:
    st.markdown("---")
    st.markdown(
        """
        <div class="pdm-sidebar-footer">
          Made by Shabeer Ahmad Ahsaas
        </div>
        """,
        unsafe_allow_html=True,
    )

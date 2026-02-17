import io
import time
from datetime import datetime, timezone

import streamlit as st
import pandas as pd

from design.theme import load_css, page_header
from design.components import kpi
from services.sheets import append_new_rows_with_rejection_and_dedupe
from services.streamlit_gsheets import get_worksheet


# ============================================================
# Page Config + Styling
# ============================================================
st.set_page_config(page_title="G-Sheet Updater | PDM", page_icon="📤", layout="wide")
load_css()
page_header("G-Sheet Updater", "Upload Excel → filter → preview → append (dedupe + rejection log)")

# ============================================================
# Secrets
# ============================================================
try:
    sa = dict(st.secrets["gcp_service_account"])
    spreadsheet_id = st.secrets["app"]["spreadsheet_id"]
except Exception:
    st.error("Secrets are not configured. Please set .streamlit/secrets.toml (service account + spreadsheet_id).")
    st.stop()

# ============================================================
# Sheets
# ============================================================
TARGET_WORKSHEET = "Raw_Kobo_Data"
REJECTION_WORKSHEET = "Rejection_Log"

try:
    target_ws = get_worksheet(sa, spreadsheet_id, TARGET_WORKSHEET)
except Exception:
    st.error(f"Worksheet '{TARGET_WORKSHEET}' was not found. Please create it in your spreadsheet.")
    st.stop()

try:
    rejection_ws = get_worksheet(sa, spreadsheet_id, REJECTION_WORKSHEET)
except Exception:
    st.error(f"Worksheet '{REJECTION_WORKSHEET}' was not found. Please create it in your spreadsheet.")
    st.stop()

# ============================================================
# Helpers
# ============================================================
def safe_read_excel(uploaded_file) -> pd.DataFrame:
    """
    Reads Excel safely:
    - supports .xlsx/.xls
    - returns DataFrame with empty strings for NA (after normalization)
    """
    try:
        # streamlit provides a BytesIO-like object
        df = pd.read_excel(uploaded_file)
    except Exception:
        # fallback: read bytes
        data = uploaded_file.getvalue()
        df = pd.read_excel(io.BytesIO(data))
    return df


def normalize_datetimes_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    Google Sheets append should receive plain strings, not Timestamp objects.
    Converts any datetime-like columns to YYYY-mm-dd HH:MM:SS.
    Also handles stray datetime objects inside object columns.
    """
    df = df.copy()

    # Convert true datetime dtypes
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for c in dt_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    # Make sure NA -> ""
    df = df.astype(object).where(pd.notnull(df), "")

    # Guard: convert any remaining Timestamp/datetime embedded in object columns
    def _fix(x):
        if hasattr(x, "to_pydatetime"):
            try:
                return x.to_pydatetime().strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return str(x)
        if hasattr(x, "isoformat"):
            try:
                # avoid timezone oddities; keep human readable
                return x.isoformat(sep=" ")
            except Exception:
                return str(x)
        return x

    return df.applymap(_fix)


def parse_start_col(df: pd.DataFrame, start_col: str) -> pd.DataFrame:
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    return df


# ============================================================
# Sidebar Controls
# ============================================================
st.sidebar.header("Controls")

start_col = st.sidebar.text_input("Date column name", value="start").strip() or "start"
cutoff_mode = st.sidebar.selectbox("Filter mode", ["Cutoff date", "Date range"], index=0)

if cutoff_mode == "Cutoff date":
    cutoff_date = st.sidebar.date_input("Cutoff date (keep >=)", value=datetime(2026, 2, 11).date())
else:
    cutoff_date = None

# Optional: require _uuid
require_uuid = st.sidebar.checkbox("Require _uuid (skip if missing)", value=True)

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

# ============================================================
# Upload
# ============================================================
st.markdown("### Upload an Excel file")
uploaded = st.file_uploader("Excel file (.xlsx / .xls)", type=["xlsx", "xls"])

if not uploaded:
    st.info("Upload an Excel file to continue.")
    st.stop()

# ============================================================
# Read Excel
# ============================================================
try:
    raw_df = safe_read_excel(uploaded)
except Exception as e:
    st.exception(e)
    st.stop()

if raw_df is None or raw_df.empty:
    st.warning("Uploaded file has no rows.")
    st.stop()

# Normalize NaN early for display
raw_df = raw_df.where(pd.notnull(raw_df), "")

# ============================================================
# Validate + Date filtering
# ============================================================
if start_col not in raw_df.columns:
    st.error(f"Column '{start_col}' not found in uploaded file.")
    st.stop()

df = parse_start_col(raw_df, start_col=start_col)

invalid_start_count = int(df[start_col].isna().sum())
if invalid_start_count:
    st.warning(f"{invalid_start_count} rows have invalid/empty '{start_col}' and will be excluded.")

df = df[df[start_col].notna()].copy()
if df.empty:
    st.warning("All rows were excluded due to invalid dates.")
    st.stop()

# Build date filter UI (dynamic)
min_d = df[start_col].min().date()
max_d = df[start_col].max().date()

if cutoff_mode == "Cutoff date":
    cutoff_ts = pd.Timestamp(cutoff_date)
    before_count = int(len(df))
    df = df[df[start_col] >= cutoff_ts].copy()
    after_count = int(len(df))
    st.info(f"Filtered out {before_count - after_count} old records ({start_col} < {cutoff_date}).")
else:
    date_range = st.sidebar.date_input(
        "Date range (keep within)",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d1, d2 = date_range
    else:
        d1, d2 = min_d, max_d

    before_count = int(len(df))
    df = df[(df[start_col].dt.date >= d1) & (df[start_col].dt.date <= d2)].copy()
    after_count = int(len(df))
    st.info(f"Filtered out {before_count - after_count} rows outside range ({d1} → {d2}).")

if df.empty:
    st.warning("No valid data to insert after filtering.")
    st.stop()

# ============================================================
# KPI summary
# ============================================================
st.markdown("### Summary")
k1, k2, k3, k4 = st.columns(4)

with k1:
    kpi("Rows in file", f"{len(raw_df):,}", "")
with k2:
    kpi("Valid dates", f"{len(raw_df) - invalid_start_count:,}", f"Column: {start_col}")
with k3:
    kpi("After filter", f"{len(df):,}", "")
with k4:
    uuid_present = "_uuid" in df.columns
    if uuid_present:
        missing_uuid = int((df["_uuid"].astype(str).str.strip() == "").sum())
        kpi("Missing _uuid", f"{missing_uuid:,}", "Will be skipped" if require_uuid else "Not required")
    else:
        kpi("_uuid column", "Missing", "Will skip if required")

st.markdown("---")

# ============================================================
# Prepare for Sheets append (string conversion)
# ============================================================
# Convert start back to string too (keep consistent)
df_for_export = df.copy()
df_for_export[start_col] = df_for_export[start_col].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

df_for_export = normalize_datetimes_to_string(df_for_export)

# Preview
st.markdown("### Preview (Filtered Data)")
st.dataframe(df_for_export.head(30), use_container_width=True, hide_index=True)

# Convert to records
records = df_for_export.to_dict(orient="records")

# ============================================================
# Action
# ============================================================
left, right = st.columns([1, 2])
with left:
    run_btn = st.button("Append new rows", type="primary")
with right:
    st.caption(
        "Appending uses your service: dedupe by key + log rejected rows in Rejection_Log. "
        "All datetime values are converted to strings before upload."
    )

if run_btn:
    t0 = time.perf_counter()

    # Hard guard if user wants _uuid but it doesn't exist
    if require_uuid and "_uuid" not in df_for_export.columns:
        st.error("You enabled 'Require _uuid' but the uploaded file has no '_uuid' column.")
        st.stop()

    with st.spinner("Appending rows to Google Sheet (dedupe + rejection log)..."):
        try:
            result = append_new_rows_with_rejection_and_dedupe(
                target_ws=target_ws,
                rejection_ws=rejection_ws,
                rows=records,
                skip_if_key_missing=require_uuid,
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    dt = time.perf_counter() - t0

    st.success(
        "Done. "
        f"Input: {result.get('total_input', 0)} | "
        f"Inserted: {result.get('inserted', 0)} | "
        f"Skipped rejected: {result.get('skipped_rejected', 0)} | "
        f"Skipped duplicates: {result.get('skipped_duplicates', 0)} | "
        f"Skipped missing _uuid: {result.get('skipped_missing_key', 0)} | "
        f"Runtime: {dt:.2f}s"
    )

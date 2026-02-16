import streamlit as st
import pandas as pd

from design.theme import load_css, page_header
from services.sheets import append_new_rows_with_rejection_and_dedupe
from services.streamlit_gsheets import get_worksheet

# ------------------ Page Config ------------------
st.set_page_config(page_title="G-Sheet Updater | PDM", page_icon="", layout="wide")
load_css()
page_header(" G-Sheet Updater")

# ------------------ Secrets ------------------
try:
    sa = dict(st.secrets["gcp_service_account"])
    spreadsheet_id = st.secrets["app"]["spreadsheet_id"]
except Exception:
    st.error("Secrets are not configured. Please set .streamlit/secrets.toml (service account + spreadsheet_id).")
    st.stop()


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

# ------------------ Upload ------------------
st.markdown("### Upload an Excel file")
uploaded = st.file_uploader("Excel file (.xlsx)", type=["xlsx", "xls"])

if uploaded:
    try:
        df = pd.read_excel(uploaded).fillna("")
    except Exception as e:
        st.exception(e)
        st.stop()

    # ------------------ Date Filter ------------------
    if "start" not in df.columns:
        st.error("Column 'start' not found in uploaded file.")
        st.stop()

    # start -> datetime
    df["start"] = pd.to_datetime(df["start"], errors="coerce")

    # حذف startهای نامعتبر
    invalid_start_count = int(df["start"].isna().sum())
    if invalid_start_count > 0:
        st.warning(f"{invalid_start_count} rows have invalid/empty 'start' and will be excluded.")
    df = df[df["start"].notna()].copy()

    cutoff_date = pd.Timestamp("2026-02-11")

    before_count = len(df)
    df = df[df["start"] >= cutoff_date].copy()
    after_count = len(df)

    st.info(f"Filtered out {before_count - after_count} old records (start < 2026-02-11).")

    # ------------------ Convert ALL datetime columns to string (IMPORTANT) ------------------
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    for col in datetime_cols:
        df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    # بیمه نهایی: اگر داخل object ها هم datetime باشد
    df = df.astype(object).where(pd.notnull(df), "")
    df = df.applymap(lambda x: x.isoformat(sep=" ") if hasattr(x, "isoformat") else x)

    # ------------------ Preview ------------------
    st.markdown("#### Preview (Filtered Data)")
    st.dataframe(df.head(30), use_container_width=True)

    # ------------------ Convert to Records ------------------
    records = df.to_dict(orient="records")

    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        run_btn = st.button("Append new rows", type="primary")

    if run_btn:
        if df.empty:
            st.warning("No valid data to insert after filtering.")
            st.stop()

        with st.spinner("Loading _uuid sets and appending allowed rows..."):
            try:
                result = append_new_rows_with_rejection_and_dedupe(
                    target_ws=target_ws,
                    rejection_ws=rejection_ws,
                    rows=records,
                    skip_if_key_missing=True,
                )
            except Exception as e:
                st.exception(e)
                st.stop()

        st.success(
            "Done. "
            f"Input: {result['total_input']} | "
            f"Inserted: {result['inserted']} | "
            f"Skipped rejected: {result['skipped_rejected']} | "
            f"Skipped duplicates: {result['skipped_duplicates']} | "
            f"Skipped missing _uuid: {result['skipped_missing_key']}"
        )

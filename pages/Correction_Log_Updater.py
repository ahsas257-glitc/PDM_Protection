import time
import streamlit as st
import pandas as pd

from design.theme import load_css, page_header
from services.sheets import scan_raw_kobo_and_log_corrections, looks_non_english
from services.streamlit_gsheets import get_worksheet

st.set_page_config(page_title="Correction Log Updater | PDM", page_icon="", layout="wide")
load_css()
page_header(
    "Correction Log Updater",
    "Scan Raw_Kobo_Data and track Correction_Log progress (new_value + assignment coverage)",
)


def mini_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="pdm-card" style="padding:12px 14px;">
          <div style="font-size:0.78rem; opacity:0.72; line-height:1.1;">{title}</div>
          <div style="font-size:1.35rem; font-weight:800; margin-top:6px; line-height:1.1;">{value}</div>
          <div style="font-size:0.78rem; opacity:0.62; margin-top:6px; line-height:1.2;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_get(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df))
    return df[col].astype(str).fillna("").str.strip()


def load_correction_log_df(corr_ws) -> pd.DataFrame:
    # Prefer records to keep header mapping reliable
    records = corr_ws.get_all_records()
    return pd.DataFrame(records) if records else pd.DataFrame()


def compute_progress_metrics(df: pd.DataFrame) -> dict:
    # Expected columns (handle missing gracefully)
    old_s = safe_get(df, "old_value")
    new_s = safe_get(df, "new_value")
    assigned_s = safe_get(df, "Assigned_To")

    total_rows = int(len(df))

    # Old value presence
    old_present = int((old_s != "").sum())

    # new_value completion
    new_filled = int((new_s != "").sum())
    pending = int(total_rows - new_filled)

    # Validate new_value should NOT be Arabic-script (Dari/Farsi/Arabic)
    # Count only where new_value is filled
    new_invalid = int(new_s[new_s != ""].apply(looks_non_english).sum())
    new_valid = int(max(0, new_filled - new_invalid))

    # Assignment coverage
    assigned = int((assigned_s != "").sum())
    unassigned = int(total_rows - assigned)
    unique_assignees = int(assigned_s[assigned_s != ""].nunique())

    completion_rate = (new_filled / total_rows * 100.0) if total_rows else 0.0
    assignment_rate = (assigned / total_rows * 100.0) if total_rows else 0.0
    valid_rate = (new_valid / new_filled * 100.0) if new_filled else 0.0

    return {
        "total_rows": total_rows,
        "old_present": old_present,
        "new_filled": new_filled,
        "pending": pending,
        "new_valid": new_valid,
        "new_invalid": new_invalid,
        "assigned": assigned,
        "unassigned": unassigned,
        "unique_assignees": unique_assignees,
        "completion_rate": completion_rate,
        "assignment_rate": assignment_rate,
        "valid_rate": valid_rate,
    }


def build_progress_charts(df: pd.DataFrame, metrics: dict):
    # Progress breakdown chart: Old vs Completed vs Pending
    progress_df = pd.DataFrame(
        {
            "count": [metrics["new_filled"], metrics["pending"]],
        },
        index=["Completed (new_value filled)", "Pending (new_value empty)"],
    )

    # new_value quality chart: valid vs invalid (only among filled new_value)
    quality_df = pd.DataFrame(
        {
            "count": [metrics["new_valid"], metrics["new_invalid"]],
        },
        index=["Valid new_value (English)", "Invalid new_value (Arabic-script)"],
    )

    # Assignment chart: assigned vs unassigned
    assign_df = pd.DataFrame(
        {
            "count": [metrics["assigned"], metrics["unassigned"]],
        },
        index=["Assigned", "Unassigned"],
    )

    # Per-assignee workload (top 12)
    assigned_s = safe_get(df, "Assigned_To")
    if len(df) and (assigned_s != "").any():
        per_assignee = (
            assigned_s[assigned_s != ""]
            .value_counts()
            .head(12)
            .rename_axis("assignee")
            .reset_index(name="count")
            .set_index("assignee")
        )
    else:
        per_assignee = pd.DataFrame({"count": []})

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("#### Correction progress")
        st.bar_chart(progress_df)

        st.markdown("#### Assignment coverage")
        st.bar_chart(assign_df)

    with right:
        st.markdown("#### new_value quality (English-only target)")
        st.bar_chart(quality_df)

        st.markdown("#### Top assignees (by assigned records)")
        if per_assignee.empty:
            st.info("No assignments found in 'Assigned_To'.")
        else:
            st.bar_chart(per_assignee)


# Secrets
try:
    sa = dict(st.secrets["gcp_service_account"])
    spreadsheet_id = st.secrets["app"]["spreadsheet_id"]
except Exception:
    st.error("Secrets are not configured. Please set .streamlit/secrets.toml (service account + spreadsheet_id).")
    st.stop()


RAW_WORKSHEET = "Raw_Kobo_Data"
CORR_WORKSHEET = "Correction_Log"

try:
    raw_ws = get_worksheet(sa, spreadsheet_id, RAW_WORKSHEET)
except Exception:
    st.error(f"Worksheet '{RAW_WORKSHEET}' was not found. Please create it in your spreadsheet.")
    st.stop()

try:
    corr_ws = get_worksheet(sa, spreadsheet_id, CORR_WORKSHEET)
except Exception:
    st.error(f"Worksheet '{CORR_WORKSHEET}' was not found. Please create it in your spreadsheet.")
    st.stop()

# Load Correction_Log analytics (before run)
df_before = load_correction_log_df(corr_ws)
metrics_before = compute_progress_metrics(df_before)

last_scan = st.session_state.get("last_scan_ts", "—")
last_duration = st.session_state.get("last_scan_duration", None)
dur_txt = f"{last_duration:.2f}s" if isinstance(last_duration, (int, float)) else "—"

st.markdown("### Key metrics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    mini_card("Total issues", f"{metrics_before['total_rows']:,}", "Total records in Correction_Log")
with c2:
    mini_card("Completed", f"{metrics_before['new_filled']:,}", f"{metrics_before['completion_rate']:.2f}% completion")
with c3:
    mini_card("Pending", f"{metrics_before['pending']:,}", "new_value is empty")
with c4:
    mini_card("Assigned coverage", f"{metrics_before['assigned']:,}", f"{metrics_before['assignment_rate']:.2f}% assigned")

st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

c5, c6, c7, c8 = st.columns(4)
with c5:
    mini_card("Unique assignees", f"{metrics_before['unique_assignees']:,}", "Distinct values in Assigned_To")
with c6:
    mini_card("Unassigned", f"{metrics_before['unassigned']:,}", "Assigned_To is empty")
with c7:
    mini_card("Valid new_value", f"{metrics_before['new_valid']:,}", f"{metrics_before['valid_rate']:.2f}% of filled new_value")
with c8:
    mini_card("Last scan", f"{last_scan}", f"Duration: {dur_txt}")

st.markdown("---")

st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

run = st.button("Run scan and update Correction_Log", type="primary")

if run:
    t0 = time.perf_counter()
    with st.spinner("Scanning Raw_Kobo_Data and updating Correction_Log..."):
        try:
            result = scan_raw_kobo_and_log_corrections(
                raw_ws=raw_ws,
                corr_ws=corr_ws,
                uuid_col="_uuid",
                max_rows=None,
            )
        except Exception as e:
            st.exception(e)
            st.stop()
    duration_s = time.perf_counter() - t0

    st.session_state["last_scan_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["last_scan_duration"] = float(duration_s)

    # Reload analytics after run
    df_after = load_correction_log_df(corr_ws)
    metrics_after = compute_progress_metrics(df_after)

    found = int(result.get("found", 0))
    inserted = int(result.get("inserted", 0))
    skipped = int(result.get("skipped_existing", 0))

    st.markdown("### Run results")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        mini_card("Detected (scan)", f"{found:,}", "Non-English values")
    with r2:
        mini_card("Inserted (new)", f"{inserted:,}", "Appended into Correction_Log")
    with r3:
        mini_card("Skipped (duplicate)", f"{skipped:,}", "Already existed in Correction_Log")
    with r4:
        mini_card("Scan duration", f"{duration_s:.2f}s", "End-to-end runtime")

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    st.markdown("### Progress (after run)")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        mini_card("Total issues", f"{metrics_after['total_rows']:,}", "Total records in Correction_Log")
    with p2:
        mini_card("Completed", f"{metrics_after['new_filled']:,}", f"{metrics_after['completion_rate']:.2f}% completion")
    with p3:
        mini_card("Valid new_value", f"{metrics_after['new_valid']:,}", f"{metrics_after['valid_rate']:.2f}% of filled new_value")
    with p4:
        mini_card("Assigned coverage", f"{metrics_after['assigned']:,}", f"{metrics_after['assignment_rate']:.2f}% assigned")

    st.markdown("#### Completion gauge")
    st.progress(int(round(metrics_after["completion_rate"])))
    st.caption(
        f"Completion: {metrics_after['completion_rate']:.2f}%  |  "
        f"Assigned: {metrics_after['assignment_rate']:.2f}%  |  "
        f"new_value valid (English): {metrics_after['valid_rate']:.2f}%"
    )

    st.markdown("---")
    st.markdown("### Analytics charts")
    build_progress_charts(df_after, metrics_after)

    st.success("Correction_Log analytics updated successfully.")
else:
    st.markdown("### Analytics charts")
    if df_before.empty:
        st.info("Correction_Log is empty or missing headers. Run a scan first or verify the sheet structure.")
    else:
        build_progress_charts(df_before, metrics_before)

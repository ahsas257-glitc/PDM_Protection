import time
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timezone

from design.theme import load_css, page_header
from services.sheets import scan_raw_kobo_and_log_corrections, looks_non_english
from services.streamlit_gsheets import get_worksheet

# ============================================================
# Page Config + Styling
# ============================================================
st.set_page_config(page_title="Correction Log Updater | PDM", page_icon="🧹", layout="wide")
load_css()
page_header(
    "Correction Log Updater",
    "Scan Raw_Kobo_Data and track Correction_Log progress (new_value + assignment coverage)",
)

# ============================================================
# Theme (match your PDM palette)
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
# Altair theme (modern, glass-friendly)
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

alt.themes.register("pdm_modern_corrections", _pdm_altair_theme)
alt.themes.enable("pdm_modern_corrections")

# ============================================================
# UI helpers
# ============================================================
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
    if df is None or df.empty:
        return pd.Series([], dtype=str)
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=str)
    return df[col].astype(str).fillna("").str.strip()

def load_correction_log_df(corr_ws) -> pd.DataFrame:
    # Prefer records (stable header mapping)
    try:
        records = corr_ws.get_all_records()
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ============================================================
# Metrics
# ============================================================
def compute_progress_metrics(df: pd.DataFrame) -> dict:
    old_s = safe_get(df, "old_value")
    new_s = safe_get(df, "new_value")
    assigned_s = safe_get(df, "Assigned_To")

    total_rows = int(len(df))

    old_present = int((old_s != "").sum())

    new_filled = int((new_s != "").sum())
    pending = int(total_rows - new_filled)

    # "looks_non_english" is your rule: Arabic-script etc.
    new_invalid = int(new_s[new_s != ""].apply(looks_non_english).sum()) if new_filled else 0
    new_valid = int(max(0, new_filled - new_invalid))

    assigned = int((assigned_s != "").sum())
    unassigned = int(total_rows - assigned)
    unique_assignees = int(assigned_s[assigned_s != ""].nunique()) if total_rows else 0

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

# ============================================================
# Modern Charts (Altair)
# ============================================================
def donut(df_: pd.DataFrame, title: str, colors: list[str]):
    d = df_.copy()
    total = float(d["count"].sum()) if len(d) else 0.0
    d["pct"] = d["count"] / max(1.0, total)

    chart = (
        alt.Chart(d)
        .mark_arc(innerRadius=85, outerRadius=135, cornerRadius=8)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("label:N", scale=alt.Scale(range=colors), legend=alt.Legend(title="")),
            tooltip=[
                alt.Tooltip("label:N", title=""),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("pct:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=340, title=title)
    )

    # center text (percent)
    pct = (d.iloc[0]["pct"] * 100.0) if len(d) else 0.0
    center = (
        alt.Chart(pd.DataFrame({"t": [f"{pct:.0f}%"]}))
        .mark_text(fontSize=28, fontWeight=900, color="#E2E8F0")
        .encode(text="t:N")
    )

    return chart + center

def barh(df_: pd.DataFrame, title: str, x_title="Count", max_rows=12):
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

def trend_line(df: pd.DataFrame, title: str):
    return (
        alt.Chart(df)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("day:T", title="Date"),
            y=alt.Y("count:Q", title="New issues"),
            tooltip=[alt.Tooltip("day:T", title="Date"), alt.Tooltip("count:Q", title="Count")],
            color=alt.value(PDM_COLORS["indigo"]),
        )
        .properties(height=260, title=title)
    )

def build_progress_charts(df: pd.DataFrame, metrics: dict):
    # --- Core breakdowns
    progress_df = pd.DataFrame(
        [
            {"label": "Completed", "count": metrics["new_filled"]},
            {"label": "Pending", "count": metrics["pending"]},
        ]
    )
    quality_df = pd.DataFrame(
        [
            {"label": "Valid (English)", "count": metrics["new_valid"]},
            {"label": "Invalid (Arabic-script)", "count": metrics["new_invalid"]},
        ]
    )
    assign_df = pd.DataFrame(
        [
            {"label": "Assigned", "count": metrics["assigned"]},
            {"label": "Unassigned", "count": metrics["unassigned"]},
        ]
    )

    # --- Per-assignee workload (top 12)
    assigned_s = safe_get(df, "Assigned_To")
    if len(df) and (assigned_s != "").any():
        per_assignee = (
            assigned_s[assigned_s != ""]
            .value_counts()
            .head(12)
            .rename_axis("label")
            .reset_index(name="count")
        )
    else:
        per_assignee = pd.DataFrame({"label": [], "count": []})

    # --- Optional trend: if a timestamp-like column exists
    ts_col = next(
        (c for c in ["timestamp", "Timestamp", "created_at", "Created_At", "date", "Date", "logged_at", "Logged_At"] if c in df.columns),
        None,
    )
    trend_df = None
    if ts_col and not df.empty:
        t = pd.to_datetime(df[ts_col], errors="coerce")
        if t.notna().any():
            tmp = pd.DataFrame({"day": t.dt.date}).dropna()
            trend_df = tmp.groupby("day").size().reset_index(name="count")
            trend_df["day"] = pd.to_datetime(trend_df["day"])

    # --- Layout


    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.altair_chart(
            donut(progress_df, "Completion", colors=[PDM_COLORS["green"], PDM_COLORS["amber"]]),
            use_container_width=True,
        )
    with c2:
        st.altair_chart(
            donut(assign_df, "Assignment coverage", colors=[PDM_COLORS["teal"], PDM_COLORS["slate"]]),
            use_container_width=True,
        )
    with c3:
        st.altair_chart(
            donut(quality_df, "new_value quality", colors=[PDM_COLORS["blue"], PDM_COLORS["red"]]),
            use_container_width=True,
        )

    left, right = st.columns([1.2, 1])
    with left:
        st.altair_chart(barh(per_assignee, "Top assignees", x_title="Assigned records", max_rows=12), use_container_width=True)
    with right:
        if trend_df is None or trend_df.empty:
            st.info("Trend chart skipped: no timestamp/date column detected in Correction_Log.")
        else:
            st.altair_chart(trend_line(trend_df.sort_values("day"), "Issues trend (by day)"), use_container_width=True)

# ============================================================
# Secrets
# ============================================================
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

# ============================================================
# Load Correction_Log analytics (before run)
# ============================================================
df_before = load_correction_log_df(corr_ws)
metrics_before = compute_progress_metrics(df_before)

last_scan = st.session_state.get("last_scan_ts", "—")
last_duration = st.session_state.get("last_scan_duration", None)
dur_txt = f"{last_duration:.2f}s" if isinstance(last_duration, (int, float)) else "—"

# ============================================================
# KPI row(s)
# ============================================================
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

# ============================================================
# Run scan
# ============================================================
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
    build_progress_charts(df_after, metrics_after)

    st.success("Correction_Log analytics updated successfully.")

else:
    if df_before.empty:
        st.info("Correction_Log is empty or missing headers. Run a scan first or verify the sheet structure.")
    else:
        build_progress_charts(df_before, metrics_before)

# ============================================================
# Sidebar footer (correct placement)
# ============================================================
with st.sidebar:
    st.markdown("---")
    st.markdown(
        """
        <div class="pdm-sidebar-footer">
          Made by Shabeer Ahmad Ahsaas
        </div>
        """,
        unsafe_allow_html=True
    )

# pages/Daily_Plan.py
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timezone

from design.theme import load_css, page_header
from design.components import kpi
from services.streamlit_gsheets import get_worksheet

# ============================================================
# Page Config + Styling
# ============================================================
st.set_page_config(page_title="Daily Plan | PDM", page_icon="🗓️", layout="wide")
load_css()
page_header("Daily Plan", "Clean, modern Daily_Plan dashboard + editor (read-only + export)")

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
# Sheet
# ============================================================
DAILY_PLAN_WS = "Daily_Plan"
try:
    ws = get_worksheet(sa, spreadsheet_id, DAILY_PLAN_WS)
except Exception:
    st.error(f"Worksheet '{DAILY_PLAN_WS}' was not found. Please create it in your spreadsheet.")
    st.stop()

# ============================================================
# Required Headers (exact)
# ============================================================
REQ_HEADERS = [
    "Timestamp",
    "Interview_Date",
    "Surveyor Name",
    "Province",
    "Beneficiary Full Name",
    "Phone Number",
    "Status",
    "Remark",
]

# ============================================================
# UI Theme (colors)
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

STATUS_PALETTE = [
    PDM_COLORS["teal"],
    PDM_COLORS["blue"],
    PDM_COLORS["violet"],
    PDM_COLORS["pink"],
    PDM_COLORS["amber"],
    PDM_COLORS["red"],
    PDM_COLORS["green"],
    PDM_COLORS["slate"],
]

# ============================================================
# Data Load
# ============================================================
@st.cache_data(ttl=30)
def load_sheet_as_df(_ws) -> pd.DataFrame:
    """
    Load with get_all_values to avoid duplicate header issues.
    Uses REQUIRED headers by position if row1 exists but has mismatch.
    """
    values = _ws.get_all_values()
    if not values:
        return pd.DataFrame(columns=REQ_HEADERS)

    header = [str(x).strip() for x in (values[0] or [])]
    data = values[1:] if len(values) > 1 else []

    # If header doesn't match, still map by position to REQ_HEADERS (safe mode)
    if header[: len(REQ_HEADERS)] != REQ_HEADERS:
        rows = []
        for r in data:
            row = {}
            for i, col in enumerate(REQ_HEADERS):
                row[col] = r[i].strip() if i < len(r) and r[i] is not None else ""
            rows.append(row)
        df = pd.DataFrame(rows, columns=REQ_HEADERS)
    else:
        idx_map = {h: i for i, h in enumerate(header)}
        rows = []
        for r in data:
            row = {}
            for col in REQ_HEADERS:
                i = idx_map.get(col, None)
                row[col] = r[i].strip() if (i is not None and i < len(r) and r[i] is not None) else ""
            rows.append(row)
        df = pd.DataFrame(rows, columns=REQ_HEADERS)

    # Normalize types
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Interview_Date"] = pd.to_datetime(df["Interview_Date"], errors="coerce")
    for c in ["Surveyor Name", "Province", "Beneficiary Full Name", "Phone Number", "Status", "Remark"]:
        df[c] = df[c].astype(str).fillna("").map(lambda x: str(x).strip()).replace({"nan": ""})

    # Soft phone cleanup
    df["Phone Number"] = df["Phone Number"].map(
        lambda s: "".join(ch for ch in str(s) if ch.isdigit() or ch == "+")
    )

    return df


def ensure_header_row(_ws):
    """
    Ensures row 1 contains the exact headers (overwrites row 1 only).
    """
    row1 = _ws.row_values(1)
    row1 = [str(x).strip() for x in (row1 or []) if str(x).strip()]
    if row1[: len(REQ_HEADERS)] != REQ_HEADERS:
        _ws.update("A1:H1", [REQ_HEADERS])


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def progress_color(pct: float) -> str:
    if pct < 35:
        return PDM_COLORS["red"]
    if pct < 70:
        return PDM_COLORS["amber"]
    return PDM_COLORS["green"]


def render_progress_bar(label: str, value: int, total: int):
    pct = (value / total * 100) if total else 0.0
    pct = max(0.0, min(100.0, pct))
    color = progress_color(pct)
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <div style="font-size:0.9rem; opacity:0.85;"><b>{label}</b></div>
          <div style="font-size:0.9rem; opacity:0.85;">{pct:.1f}%</div>
        </div>
        <div style="width:100%; height:10px; background:rgba(148,163,184,0.25); border-radius:999px; overflow:hidden;">
          <div style="width:{pct}%; height:100%; background:{color}; border-radius:999px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Boot
# ============================================================
try:
    ensure_header_row(ws)
except Exception:
    # don't block the page
    pass

df = load_sheet_as_df(ws)

# ============================================================
# Sidebar (minimal)
# ============================================================
st.sidebar.header("Filters")

if not df.empty and df["Interview_Date"].notna().any():
    min_d = df["Interview_Date"].min().date()
    max_d = df["Interview_Date"].max().date()
else:
    today = datetime.now().date()
    min_d, max_d = today, today

date_range = st.sidebar.date_input(
    "Interview date range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d1, d2 = date_range
else:
    d1, d2 = min_d, max_d

province_opts = sorted([x for x in df["Province"].dropna().unique().tolist() if str(x).strip()]) if not df.empty else []
status_opts = sorted([x for x in df["Status"].dropna().unique().tolist() if str(x).strip()]) if not df.empty else []
surveyor_opts = sorted([x for x in df["Surveyor Name"].dropna().unique().tolist() if str(x).strip()]) if not df.empty else []

prov_sel = st.sidebar.multiselect("Province", province_opts, default=[])
stat_sel = st.sidebar.multiselect("Status", status_opts, default=[])
surv_sel = st.sidebar.multiselect("Surveyor", surveyor_opts, default=[])
q = st.sidebar.text_input("Search (name/phone/remark)", value="").strip().lower()

# Apply filters
fdf = df.copy()
if not fdf.empty:
    s = fdf["Interview_Date"]
    fdf = fdf[(s.notna()) & (s.dt.date >= d1) & (s.dt.date <= d2)]
    if prov_sel:
        fdf = fdf[fdf["Province"].isin(prov_sel)]
    if stat_sel:
        fdf = fdf[fdf["Status"].isin(stat_sel)]
    if surv_sel:
        fdf = fdf[fdf["Surveyor Name"].isin(surv_sel)]
    if q:
        fdf = fdf[
            fdf["Beneficiary Full Name"].str.lower().str.contains(q, na=False)
            | fdf["Phone Number"].str.lower().str.contains(q, na=False)
            | fdf["Remark"].str.lower().str.contains(q, na=False)
        ]

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
# Tabs (clean)
# ============================================================
tab_overview, tab_analytics, tab_records = st.tabs(
    ["Overview", "Analytics", "Records (Read-only + Export)"]
)

# ============================================================
# OVERVIEW
# ============================================================
with tab_overview:
    if df.empty:
        st.warning("Daily_Plan has no data yet.")
        st.stop()

    total = int(len(df))
    filtered = int(len(fdf))

    today = datetime.now().date()
    today_count = int((df["Interview_Date"].dt.date == today).sum()) if df["Interview_Date"].notna().any() else 0

    missing_phone = int((df["Phone Number"].astype(str).str.strip() == "").sum())
    missing_status = int((df["Status"].astype(str).str.strip() == "").sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi("Total records", f"{total:,}", "")
    with k2:
        kpi("Filtered window", f"{filtered:,}", "After filters")
    with k3:
        kpi("Interviews today", f"{today_count:,}", "")
    with k4:
        kpi("Missing phone", f"{missing_phone:,}", "")

    st.markdown("### Data quality")
    q1, q2, q3 = st.columns([1, 1, 1])
    with q1:
        kpi("Missing status", f"{missing_status:,}", "")
    with q2:
        filled = total * 2 - missing_phone - missing_status
        possible = max(1, total * 2)
        completion_pct = filled / possible * 100
        kpi("Completion", f"{completion_pct:.1f}%", "Phone + Status")
    with q3:
        prov_n = int(df["Province"].astype(str).str.strip().replace("", pd.NA).dropna().nunique())
        kpi("Provinces", f"{prov_n:,}", "")

    st.markdown("### Quick progress (filtered)")
    render_progress_bar("Filtered vs Total", filtered, max(1, total))
    st.caption("Progress colors are custom themed for readability.")

# ============================================================
# ANALYTICS
# ============================================================
with tab_analytics:
    if fdf.empty:
        st.info("No data in the selected filter window.")
        st.stop()

    st.markdown("### Status + Province")

    status_df = fdf["Status"].replace({"": "—"}).fillna("—").value_counts().reset_index()
    status_df.columns = ["Status", "Count"]
    status_df["pct"] = status_df["Count"] / max(1, int(status_df["Count"].sum()))

    status_donut = (
        alt.Chart(status_df)
        .mark_arc(innerRadius=70, outerRadius=125, cornerRadius=6)
        .encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(range=STATUS_PALETTE),
                legend=alt.Legend(title="Status"),
            ),
            tooltip=[
                alt.Tooltip("Status:N"),
                alt.Tooltip("Count:Q"),
                alt.Tooltip("pct:Q", format=".1%"),
            ],
        )
        .properties(height=360)
    )

    prov_df = fdf["Province"].replace({"": "—"}).fillna("—").value_counts().reset_index()
    prov_df.columns = ["Province", "Count"]
    prov_df = prov_df.sort_values("Count", ascending=False).head(20)

    prov_bar = (
        alt.Chart(prov_df)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X("Count:Q", title="Count"),
            y=alt.Y("Province:N", sort="-x", title=""),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="turbo"), legend=None),
            tooltip=[alt.Tooltip("Province:N"), alt.Tooltip("Count:Q")],
        )
        .properties(height=360)
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.altair_chart(status_donut, use_container_width=True)
    with c2:
        st.altair_chart(prov_bar, use_container_width=True)

    st.markdown("### Trend (Interview_Date)")
    tdf = fdf[fdf["Interview_Date"].notna()].copy()
    if tdf.empty:
        st.info("No valid Interview_Date to plot.")
    else:
        tdf["day"] = tdf["Interview_Date"].dt.date
        trend = tdf.groupby("day").size().reset_index(name="Count").sort_values("day")

        base = alt.Chart(trend).encode(
            x=alt.X("day:T", title="Date"),
            y=alt.Y("Count:Q", title="Interviews"),
            tooltip=[alt.Tooltip("day:T", title="Date"), alt.Tooltip("Count:Q")],
        )
        area = base.mark_area(opacity=0.12).encode(color=alt.value(PDM_COLORS["indigo"]))
        line = base.mark_line(strokeWidth=3).encode(color=alt.value(PDM_COLORS["indigo"]))
        pts = base.mark_circle(size=80).encode(color=alt.value(PDM_COLORS["pink"]))
        st.altair_chart((area + line + pts).properties(height=320), use_container_width=True)

    st.markdown("### Surveyor performance (Top 15)")
    sv = fdf["Surveyor Name"].replace({"": "—"}).fillna("—").value_counts().reset_index()
    sv.columns = ["Surveyor", "Count"]
    sv = sv.head(15).sort_values("Count", ascending=True)

    sv_bar = (
        alt.Chart(sv)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X("Count:Q", title="Count"),
            y=alt.Y("Surveyor:N", sort=None, title=""),
            color=alt.Color("Count:Q", scale=alt.Scale(range=[PDM_COLORS["teal"], PDM_COLORS["violet"]]), legend=None),
            tooltip=[alt.Tooltip("Surveyor:N"), alt.Tooltip("Count:Q")],
        )
        .properties(height=360)
    )
    st.altair_chart(sv_bar, use_container_width=True)

# ============================================================
# RECORDS (READ-ONLY + EXPORTS)
# ============================================================
with tab_records:
    top_a, top_b, top_c, top_d = st.columns([1, 1, 1, 2])
    with top_a:
        refresh = st.button("🔄 Refresh", key="dp_refresh")
    with top_b:
        show_filtered = st.toggle("Show filtered only", value=True)
    with top_c:
        export_scope = st.selectbox("Export scope", ["Filtered", "All"], index=0)
    with top_d:
        st.caption("Table is read-only. Use downloads below (CSV / XLSX).")

    if refresh:
        load_sheet_as_df.clear()
        st.rerun()

    view_df = fdf if show_filtered else df
    if view_df.empty:
        st.info("No rows to display.")
        st.stop()

    # Read-only table (no edits)
    ro = view_df.copy()
    ro["Timestamp"] = ro["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    ro["Interview_Date"] = ro["Interview_Date"].dt.strftime("%Y-%m-%d").fillna("")
    st.dataframe(ro[REQ_HEADERS], use_container_width=True, hide_index=True)

    st.markdown("### Download")

    export_df = fdf if export_scope == "Filtered" else df
    exp = export_df.copy()
    exp["Timestamp"] = exp["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    exp["Interview_Date"] = exp["Interview_Date"].dt.strftime("%Y-%m-%d").fillna("")

    # CSV
    csv_bytes = exp[REQ_HEADERS].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV",
        data=csv_bytes,
        file_name="Daily_Plan.csv",
        mime="text/csv",
        use_container_width=False,
    )

    # XLSX
    xlsx_bytes = None
    try:
        import io

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            exp[REQ_HEADERS].to_excel(writer, index=False, sheet_name="Daily_Plan")
        xlsx_bytes = buf.getvalue()
    except Exception:
        xlsx_bytes = None

    if xlsx_bytes:
        st.download_button(
            "⬇️ Download Excel (XLSX)",
            data=xlsx_bytes,
            file_name="Daily_Plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False,
        )
    else:
        st.warning("Excel export is unavailable (openpyxl missing). CSV export is available.")

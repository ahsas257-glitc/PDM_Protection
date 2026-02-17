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
st.set_page_config(page_title="Dashboard | PDM", page_icon="📊", layout="wide")
load_css()
page_header("Dashboard", "Modern operational dashboard for monitoring and analysis")

# ============================================================
# Theme: Colors (PDM)
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
    "muted_bg": "#0B1220",   # deep background hint (for chart tooltip polish)
}

STATUS_PALETTE = [
    PDM_COLORS["teal"],
    PDM_COLORS["blue"],
    PDM_COLORS["indigo"],
    PDM_COLORS["violet"],
    PDM_COLORS["pink"],
    PDM_COLORS["amber"],
    PDM_COLORS["red"],
    PDM_COLORS["green"],
    PDM_COLORS["slate"],
]

# ============================================================
# Altair Global Theme (Modern)
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
                "titleFontWeight": 600,
            },
            "legend": {
                "labelColor": "#CBD5E1",
                "titleColor": "#E2E8F0",
                "labelFontSize": 12,
                "titleFontSize": 12,
                "titleFontWeight": 700,
                "symbolType": "circle",
                "symbolSize": 140,
            },
            "title": {
                "color": "#E2E8F0",
                "fontSize": 14,
                "fontWeight": 800,
                "anchor": "start",
                "offset": 10,
            },
            "tooltip": {
                "content": "data",
                "fill": "rgba(2,6,23,0.96)",
                "stroke": "rgba(148,163,184,0.25)",
                "color": "#E2E8F0",
            },
        }
    }

alt.themes.register("pdm_modern", _pdm_altair_theme)
alt.themes.enable("pdm_modern")

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
    rejection_ws = None

# ============================================================
# Targets (Hardcoded)
# ============================================================
TOTAL_TARGET = 974
MALE_TARGET = 596
FEMALE_TARGET = 378

PROVINCE_TARGETS = [
    {"province": "Badakhshan", "total": 70, "male": 46, "female": 24},
    {"province": "Baghlan", "total": 78, "male": 60, "female": 18},
    {"province": "Balkh", "total": 48, "male": 25, "female": 23},
    {"province": "Faryab", "total": 36, "male": 17, "female": 19},
    {"province": "Herat", "total": 123, "male": 78, "female": 45},
    {"province": "Kabul", "total": 83, "male": 62, "female": 21},
    {"province": "Kandahar", "total": 94, "male": 54, "female": 40},
    {"province": "Kunduz", "total": 61, "male": 26, "female": 35},
    {"province": "Nangarhar", "total": 119, "male": 69, "female": 50},
    {"province": "Nimroz", "total": 143, "male": 94, "female": 49},
    {"province": "Sar-e Pol", "total": 59, "male": 34, "female": 25},
    {"province": "Takhar", "total": 60, "male": 31, "female": 29},
]
prov_targets_df = pd.DataFrame(PROVINCE_TARGETS)

# ============================================================
# Raw Kobo Columns (exact header text)
# ============================================================
DATE_COL = "start"  # adjust if needed
INTERVIEWER_COL = "A.2. Interviewer name"
GENDER_COL = "A.5. Sex of Respondent"
PROVINCE_COL = "A.8. Province"
ASSIST_COL = "B.1. What kind of assistance did you receive from IOM Protection?"
DISABILITY_COL = "A.12. Is there any person with disability in your family?"

# ============================================================
# Data Load
# ============================================================
@st.cache_data(ttl=60)
def load_sheet_as_df(_ws) -> pd.DataFrame:
    values = _ws.get_all_values()
    if not values:
        return pd.DataFrame()

    headers = values[0]
    data = values[1:]

    seen = {}
    new_headers = []
    for h in headers:
        h = str(h).strip()
        if h in seen:
            seen[h] += 1
            new_headers.append(f"{h}__{seen[h]}")
        else:
            seen[h] = 0
            new_headers.append(h)

    return pd.DataFrame(data, columns=new_headers)

df = load_sheet_as_df(target_ws)

rej_df = None
if rejection_ws is not None:
    try:
        rej_df = load_sheet_as_df(rejection_ws)
    except Exception:
        rej_df = None

if df.empty:
    st.warning("The worksheet has no data yet.")
    st.stop()

# ============================================================
# Helpers
# ============================================================
def clean_text_series(s: pd.Series, unknown="Unknown") -> pd.Series:
    return s.astype(str).str.strip().replace({"": unknown, "nan": unknown, "None": unknown})

def norm_key(x: str) -> str:
    return str(x).strip().lower()

def norm_gender_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    gender_map = {
        "m": "male", "male": "male", "man": "male",
        "f": "female", "female": "female", "woman": "female",
    }
    return s.map(gender_map).fillna(s)

def norm_yes_no_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    yn = {
        "yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
        "no": "No", "n": "No", "false": "No", "0": "No",
    }
    out = s.map(yn).fillna("Unknown")
    out = out.replace({"": "Unknown"})
    return out

def progress_color(pct: float) -> str:
    if pct < 50:
        return PDM_COLORS["red"]
    if pct < 80:
        return PDM_COLORS["amber"]
    return PDM_COLORS["green"]

def render_progress_line(label: str, received: int, target: int):
    pct = (received / target * 100) if target else 0.0
    pct = max(0.0, min(100.0, pct))
    color = progress_color(pct)
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <div style="font-size:0.9rem; opacity:0.85;"><b>{label}</b></div>
          <div style="font-size:0.9rem; opacity:0.85;">{pct:.1f}%</div>
        </div>
        <div style="width:100%; height:10px; background:rgba(148,163,184,0.18); border-radius:999px; overflow:hidden;">
          <div style="width:{pct}%; height:100%; background:{color}; border-radius:999px;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def value_counts_df(s: pd.Series, top_n: int = 12, other_label: str = "Other"):
    vc = s.value_counts(dropna=False).reset_index()
    vc.columns = ["category", "count"]
    vc["category"] = vc["category"].astype(str)
    vc = vc.sort_values("count", ascending=False).reset_index(drop=True)
    if len(vc) > top_n:
        top = vc.head(top_n).copy()
        other_count = int(vc.iloc[top_n:]["count"].sum())
        top = pd.concat([top, pd.DataFrame([{"category": other_label, "count": other_count}])], ignore_index=True)
        vc = top
    total = int(vc["count"].sum())
    vc["pct"] = vc["count"] / max(1, total)
    return vc

def donut_chart(df_: pd.DataFrame, title: str, color_range=None):
    color_range = color_range or STATUS_PALETTE
    return (
        alt.Chart(df_)
        .mark_arc(innerRadius=80, outerRadius=135, cornerRadius=8)
        .encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(range=color_range),
                legend=alt.Legend(title="")
            ),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("pct:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=360, title=title)
    )

def modern_barh(df_: pd.DataFrame, title: str, x_title="Count", y_title="", max_rows=20):
    d = df_.copy()
    d = d.sort_values("count", ascending=False).head(max_rows)
    d = d.sort_values("count", ascending=True)
    return (
        alt.Chart(d)
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
        .encode(
            x=alt.X("count:Q", title=x_title),
            y=alt.Y("category:N", sort=None, title=y_title),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="turbo"), legend=None),
            tooltip=[
                alt.Tooltip("category:N", title="Category"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("pct:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=360, title=title)
    )

def trend_area_line(trend_df: pd.DataFrame, title="Trend"):
    base = alt.Chart(trend_df).encode(
        x=alt.X("day:T", title="Date"),
        y=alt.Y("count:Q", title="Interviews"),
        tooltip=[alt.Tooltip("day:T", title="Date"), alt.Tooltip("count:Q", title="Count")],
    )
    area = base.mark_area(opacity=0.14).encode(color=alt.value(PDM_COLORS["indigo"]))
    line = base.mark_line(strokeWidth=3).encode(color=alt.value(PDM_COLORS["indigo"]))
    pts = base.mark_circle(size=80).encode(color=alt.value(PDM_COLORS["pink"]))
    return (area + line + pts).properties(height=320, title=title)

# ============================================================
# Parse Dates
# ============================================================
if DATE_COL not in df.columns:
    st.error(f"Date column '{DATE_COL}' not found in '{TARGET_WORKSHEET}'.")
    st.stop()

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df[df[DATE_COL].notna()].copy()

# ============================================================
# Sidebar Filters
# ============================================================
st.sidebar.header("Filters")

min_date = df[DATE_COL].min().date()
max_date = df[DATE_COL].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_date, max_date

mask = (df[DATE_COL].dt.date >= start_d) & (df[DATE_COL].dt.date <= end_d)
fdf = df.loc[mask].copy()

if fdf.empty:
    st.warning("No data found in the selected date range.")
    st.stop()

last_update_dt = df[DATE_COL].max()
last_update_str = last_update_dt.strftime("%Y-%m-%d %H:%M")

# ============================================================
# Normalize required fields (once)
# ============================================================
# Gender
if GENDER_COL in fdf.columns:
    fdf["_gender"] = norm_gender_series(fdf[GENDER_COL])
else:
    fdf["_gender"] = "unknown"

# Province
if PROVINCE_COL in fdf.columns:
    fdf["_province_raw"] = clean_text_series(fdf[PROVINCE_COL], "Unknown")
    fdf["_province"] = fdf["_province_raw"].map(norm_key)
else:
    fdf["_province_raw"] = "Unknown"
    fdf["_province"] = "unknown"

# Interviewer
if INTERVIEWER_COL in fdf.columns:
    fdf["_interviewer"] = clean_text_series(fdf[INTERVIEWER_COL], "Unknown interviewer")
else:
    fdf["_interviewer"] = "Unknown interviewer"

# Assistance type
if ASSIST_COL in fdf.columns:
    fdf["_assist"] = clean_text_series(fdf[ASSIST_COL], "Unknown")
else:
    fdf["_assist"] = "Unknown"

# Disability
if DISABILITY_COL in fdf.columns:
    fdf["_disability"] = norm_yes_no_series(fdf[DISABILITY_COL])
else:
    fdf["_disability"] = "Unknown"

# ============================================================
# Global KPIs (filtered window)
# ============================================================
received_total = int(len(fdf))
received_male = int((fdf["_gender"] == "male").sum())
received_female = int((fdf["_gender"] == "female").sum())

remaining_total = max(0, TOTAL_TARGET - received_total)
remaining_male = max(0, MALE_TARGET - received_male)
remaining_female = max(0, FEMALE_TARGET - received_female)

progress_total = (received_total / TOTAL_TARGET * 100) if TOTAL_TARGET else 0.0
progress_male = (received_male / MALE_TARGET * 100) if MALE_TARGET else 0.0
progress_female = (received_female / FEMALE_TARGET * 100) if FEMALE_TARGET else 0.0

# Errors (optional)
recent_errors = 0
if rej_df is not None and not rej_df.empty:
    rej_time_col = next((c for c in ["timestamp", "time", "created_at", "created", "date", "logged_at"] if c in rej_df.columns), None)
    if rej_time_col:
        rej_df[rej_time_col] = pd.to_datetime(rej_df[rej_time_col], errors="coerce")
        cutoff = pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None) - pd.Timedelta(days=7)
        recent_errors = int((rej_df[rej_time_col] >= cutoff).sum())
    else:
        recent_errors = int(len(rej_df))

# ============================================================
# Province summary table (targets + received)
# ============================================================
prov_targets_lookup = {norm_key(r["province"]): r for r in PROVINCE_TARGETS}

prov_received = (
    fdf.groupby("_province")
       .size()
       .reset_index(name="received_total")
)

prov_base = prov_targets_df.copy()
prov_base["_province"] = prov_base["province"].map(norm_key)

prov_summary = prov_base.merge(prov_received, on="_province", how="left")
prov_summary["received_total"] = prov_summary["received_total"].fillna(0).astype(int)

prov_summary["remaining_total"] = (prov_summary["total"] - prov_summary["received_total"]).clip(lower=0).astype(int)
prov_summary["progress_total_pct"] = (prov_summary["received_total"] / prov_summary["total"].replace(0, pd.NA) * 100).fillna(0.0)

prov_summary_view = prov_summary[
    ["province", "total", "received_total", "remaining_total", "progress_total_pct"]
].sort_values("progress_total_pct", ascending=False)

prov_summary_view["progress_total_pct"] = prov_summary_view["progress_total_pct"].map(lambda x: round(float(x), 1))

# ============================================================
# Interviewer summary
# ============================================================
interviewer_summary = (
    fdf.groupby("_interviewer")
       .agg(
           received=("_interviewer", "size"),
           male=("_gender", lambda x: int((x == "male").sum())),
           female=("_gender", lambda x: int((x == "female").sum())),
           disability_yes=("_disability", lambda x: int((x == "Yes").sum())),
       )
       .reset_index()
       .rename(columns={"_interviewer": "interviewer"})
)

interviewer_summary["unknown_gender"] = (interviewer_summary["received"] - interviewer_summary["male"] - interviewer_summary["female"]).clip(lower=0).astype(int)
interviewer_summary["disability_yes_pct"] = (interviewer_summary["disability_yes"] / interviewer_summary["received"].replace(0, pd.NA) * 100).fillna(0.0)
interviewer_summary["disability_yes_pct"] = interviewer_summary["disability_yes_pct"].map(lambda x: round(float(x), 1))
interviewer_summary = interviewer_summary.sort_values("received", ascending=False)

# top province per interviewer
top_prov = (
    fdf.groupby(["_interviewer", "_province_raw"])
       .size()
       .reset_index(name="n")
       .sort_values(["_interviewer", "n"], ascending=[True, False])
       .drop_duplicates(subset=["_interviewer"])
       .rename(columns={"_interviewer": "interviewer", "_province_raw": "top_province", "n": "top_province_count"})
)
interviewer_summary = interviewer_summary.merge(top_prov, on="interviewer", how="left")

# ============================================================
# Tabs
# ============================================================
tab_overview, tab_assistance, tab_disability, tab_interviewers, tab_provinces = st.tabs(
    ["Overview", "Assistance", "Disability", "Interviewers", "Provinces"]
)

# ============================================================
# OVERVIEW (Modern)
# ============================================================
with tab_overview:
    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi("Received (Total)", f"{received_total:,}", "")
    with k2:
        kpi("Progress (Total)", f"{progress_total:.1f}%", "")
    with k3:
        kpi("Remaining (Total)", f"{remaining_total:,}", "")
    with k4:
        kpi("Errors (7d)", f"{recent_errors:,}", "")
    with k5:
        yes_count = int((fdf["_disability"] == "Yes").sum())
        yes_pct = (yes_count / max(1, len(fdf)) * 100)
        kpi("Disability (Yes)", f"{yes_pct:.1f}%", "")

    st.markdown("### Progress")
    a, b = st.columns([1, 1])
    with a:
        render_progress_line("Total", received_total, TOTAL_TARGET)
        render_progress_line("Male", received_male, MALE_TARGET)
        render_progress_line("Female", received_female, FEMALE_TARGET)
    with b:
        # Gender donut (modern)
        gdf = pd.DataFrame(
            [
                {"category": "Male", "count": received_male},
                {"category": "Female", "count": received_female},
                {"category": "Unknown", "count": max(0, received_total - received_male - received_female)},
            ]
        )
        gdf["pct"] = gdf["count"] / max(1, int(gdf["count"].sum()))
        gdonut = donut_chart(gdf, "Gender Split", color_range=[PDM_COLORS["blue"], PDM_COLORS["pink"], PDM_COLORS["slate"]])
        st.altair_chart(gdonut, use_container_width=True)

    st.markdown("### Trend (Daily interviews)")
    tdf = fdf[[DATE_COL]].copy()
    tdf["day"] = tdf[DATE_COL].dt.date
    trend = tdf.groupby("day").size().reset_index(name="count").sort_values("day")
    st.altair_chart(trend_area_line(trend, title="Daily Volume"), use_container_width=True)

    # Optional: Heatmap day/hour (if hour available)
    st.markdown("### Activity heatmap (Day x Hour)")
    hdf = fdf[[DATE_COL]].copy()
    hdf["hour"] = hdf[DATE_COL].dt.hour
    hdf["dow"] = hdf[DATE_COL].dt.day_name()
    if hdf["hour"].notna().any():
        heat = (
            alt.Chart(hdf)
            .mark_rect(cornerRadius=4)
            .encode(
                x=alt.X("hour:O", title="Hour"),
                y=alt.Y("dow:N", title="Day", sort=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]),
                color=alt.Color("count():Q", scale=alt.Scale(scheme="viridis"), legend=None),
                tooltip=[alt.Tooltip("dow:N", title="Day"), alt.Tooltip("hour:O", title="Hour"), alt.Tooltip("count():Q", title="Count")],
            )
            .properties(height=260, title="When interviews happen (density)")
        )
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("Heatmap skipped: no hour information in the date column.")

    st.caption(f"Last data timestamp: {last_update_str} | Source: {TARGET_WORKSHEET}")

# ============================================================
# ASSISTANCE (Modern)
# ============================================================
with tab_assistance:
    st.markdown("### Assistance Type Distribution")

    if ASSIST_COL not in fdf.columns:
        st.warning(f"Column not found: {ASSIST_COL}")
    else:
        top_n = st.slider("Top categories", min_value=5, max_value=25, value=12, step=1, key="assist_topn")
        dist_plot = value_counts_df(fdf["_assist"], top_n=top_n, other_label="Other")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.altair_chart(donut_chart(dist_plot, "Assistance Mix"), use_container_width=True)
        with c2:
            st.altair_chart(modern_barh(dist_plot, "Assistance (Ranked)", x_title="Count"), use_container_width=True)

# ============================================================
# DISABILITY (Modern)
# ============================================================
with tab_disability:
    st.markdown("### Disability in Family (Yes/No)")

    if DISABILITY_COL not in fdf.columns:
        st.warning(f"Column not found: {DISABILITY_COL}")
    else:
        dist = value_counts_df(fdf["_disability"], top_n=10, other_label="Other")
        # stable ordering if present
        order = {"Yes": 0, "No": 1, "Unknown": 2}
        dist["sort_key"] = dist["category"].map(lambda x: order.get(x, 99))
        dist = dist.sort_values(["sort_key", "count"], ascending=[True, False]).drop(columns=["sort_key"])

        c1, c2 = st.columns([1, 1])
        with c1:
            st.altair_chart(
                donut_chart(dist, "Disability (Yes/No)", color_range=[PDM_COLORS["amber"], PDM_COLORS["teal"], PDM_COLORS["slate"]]),
                use_container_width=True
            )
        with c2:
            # compact modern column bars
            bar = (
                alt.Chart(dist)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("category:N", title=""),
                    y=alt.Y("count:Q", title="Count"),
                    color=alt.Color("category:N",
                                    scale=alt.Scale(range=[PDM_COLORS["amber"], PDM_COLORS["teal"], PDM_COLORS["slate"]]),
                                    legend=None),
                    tooltip=["category", "count", alt.Tooltip("pct:Q", format=".1%")],
                )
                .properties(height=360, title="Distribution")
            )
            st.altair_chart(bar, use_container_width=True)

# ============================================================
# INTERVIEWERS (Modern)
# ============================================================
with tab_interviewers:
    st.markdown("### Interviewers (General Overview)")
    if INTERVIEWER_COL not in fdf.columns:
        st.warning(f"Column not found: {INTERVIEWER_COL}")
    else:
        # Top interviewers chart
        top15 = interviewer_summary.head(15).copy()
        top15_plot = top15.rename(columns={"interviewer": "category", "received": "count"})
        top15_plot["pct"] = top15_plot["count"] / max(1, int(top15_plot["count"].sum()))

        c1, c2 = st.columns([1, 1])
        with c1:
            st.altair_chart(modern_barh(top15_plot, "Top Interviewers (by volume)", x_title="Interviews", max_rows=15), use_container_width=True)
        with c2:
            st.dataframe(
                interviewer_summary[
                    ["interviewer", "received", "male", "female", "unknown_gender", "top_province", "top_province_count", "disability_yes_pct"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")
        st.markdown("### Interviewer Detail")

        interviewer_list = sorted(interviewer_summary["interviewer"].dropna().unique().tolist())
        selected_interviewer = st.selectbox("Select interviewer", interviewer_list)

        idf = fdf[fdf["_interviewer"] == selected_interviewer].copy()

        i_total = int(len(idf))
        i_male = int((idf["_gender"] == "male").sum())
        i_female = int((idf["_gender"] == "female").sum())
        i_unknown = max(0, i_total - i_male - i_female)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Received", f"{i_total:,}", "")
        with k2:
            kpi("Male", f"{i_male:,}", "")
        with k3:
            kpi("Female", f"{i_female:,}", "")
        with k4:
            kpi("Unknown", f"{i_unknown:,}", "")

        c1, c2 = st.columns([1, 1])
        with c1:
            gdf = pd.DataFrame(
                [{"category": "Male", "count": i_male},
                 {"category": "Female", "count": i_female},
                 {"category": "Unknown", "count": i_unknown}]
            )
            gdf["pct"] = gdf["count"] / max(1, int(gdf["count"].sum()))
            st.altair_chart(
                donut_chart(gdf, "Gender Split (This interviewer)", color_range=[PDM_COLORS["blue"], PDM_COLORS["pink"], PDM_COLORS["slate"]]),
                use_container_width=True
            )
        with c2:
            avc = value_counts_df(idf["_assist"], top_n=10, other_label="Other").sort_values("count", ascending=False)
            avc_plot = avc.rename(columns={"category": "category", "count": "count"})
            st.altair_chart(modern_barh(avc_plot, "Assistance Types (This interviewer)", x_title="Count", max_rows=12), use_container_width=True)

# ============================================================
# PROVINCES (Modern)
# ============================================================
with tab_provinces:
    st.markdown("### Provinces (General Overview)")

    # Province ranking chart
    pplot = prov_summary.copy()
    pplot = pplot.rename(columns={"province": "category", "received_total": "count"})
    pplot["pct"] = pplot["count"] / max(1, int(pplot["count"].sum()))
    st.altair_chart(modern_barh(pplot[["category","count","pct"]], "Province Volume (Received)", x_title="Received", max_rows=20), use_container_width=True)

    st.dataframe(
        prov_summary_view,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.markdown("### Province Detail")

    province_options = sorted(prov_targets_df["province"].tolist())
    selected_province = st.selectbox("Select province", province_options)

    pkey = norm_key(selected_province)
    trow = prov_targets_lookup.get(pkey, {"total": 0, "male": 0, "female": 0})

    p_target_total = int(trow.get("total", 0))
    p_target_male = int(trow.get("male", 0))
    p_target_female = int(trow.get("female", 0))

    pdf = fdf[fdf["_province"] == pkey].copy()

    p_received_total = int(len(pdf))
    p_received_male = int((pdf["_gender"] == "male").sum())
    p_received_female = int((pdf["_gender"] == "female").sum())

    p_remaining_total = max(0, p_target_total - p_received_total)
    p_remaining_male = max(0, p_target_male - p_received_male)
    p_remaining_female = max(0, p_target_female - p_received_female)

    p_progress_total = (p_received_total / p_target_total * 100) if p_target_total else 0.0
    p_progress_male = (p_received_male / p_target_male * 100) if p_target_male else 0.0
    p_progress_female = (p_received_female / p_target_female * 100) if p_target_female else 0.0

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        kpi("Target (Total)", f"{p_target_total:,}", "")
    with t2:
        kpi("Received (Total)", f"{p_received_total:,}", "")
    with t3:
        kpi("Remaining (Total)", f"{p_remaining_total:,}", "")
    with t4:
        kpi("Progress", f"{p_progress_total:.1f}%", "")

    st.markdown("#### Progress Lines")
    render_progress_line("Total", p_received_total, p_target_total)
    render_progress_line("Male", p_received_male, p_target_male)
    render_progress_line("Female", p_received_female, p_target_female)

    c1, c2 = st.columns([1, 1])
    with c1:
        avc = value_counts_df(pdf["_assist"], top_n=10, other_label="Other")
        st.altair_chart(modern_barh(avc, "Assistance Types (This Province)", x_title="Count", max_rows=12), use_container_width=True)
    with c2:
        dvc = value_counts_df(pdf["_disability"], top_n=10, other_label="Other")
        st.altair_chart(
            donut_chart(dvc, "Disability (This Province)", color_range=[PDM_COLORS["amber"], PDM_COLORS["teal"], PDM_COLORS["slate"]]),
            use_container_width=True
        )

# ============================================================
# Sidebar Footer
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

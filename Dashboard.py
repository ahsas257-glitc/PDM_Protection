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
st.set_page_config(page_title="Dashboard | PDM", page_icon="", layout="wide")
load_css()
page_header("Dashboard", "Modern operational dashboard for monitoring and analysis")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
      .stTabs [data-baseweb="tab-list"] { gap: 10px; }
      .stTabs [data-baseweb="tab"] { border-radius: 12px; padding: 8px 12px; }
      [data-testid="stVerticalBlock"] { gap: 0.75rem; }

      /* Progress line (custom) */
      .pwrap { width: 100%; background: #E9EDF3; border-radius: 999px; height: 10px; overflow: hidden; }
      .pbar  { height: 10px; border-radius: 999px; }
      .plabel { display:flex; justify-content:space-between; align-items:center; margin: 6px 0 6px 0; }
      .plabel span { font-size: 12px; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    """
    Uses get_all_values() to avoid duplicate-header errors from get_all_records().
    Auto-uniques duplicate header names by appending __1, __2, ...
    """
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
        return "#EF4444"  # red
    if pct < 80:
        return "#F59E0B"  # amber
    return "#10B981"      # green

def render_progress_line(label: str, received: int, target: int):
    pct = (received / target * 100) if target else 0.0
    pct = max(0.0, min(100.0, pct))
    color = progress_color(pct)
    st.markdown(
        f"""
        <div class="plabel">
          <span><b>{label}</b></span>
          <span>{pct:.1f}%</span>
        </div>
        <div class="pwrap">
          <div class="pbar" style="width:{pct}%; background:{color};"></div>
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
# Province summary table (general)
# ============================================================
prov_targets_lookup = {norm_key(r["province"]): r for r in PROVINCE_TARGETS}

prov_received = (
    fdf.groupby("_province")
       .agg(
           received_total=("__dummy__", "size") if "__dummy__" in fdf.columns else ("_province", "size"),
           received_male=("_gender", lambda x: int((x == "male").sum())),
           received_female=("_gender", lambda x: int((x == "female").sum())),
       )
       .reset_index()
)

# Ensure all provinces exist in summary (from target list)
prov_base = prov_targets_df.copy()
prov_base["_province"] = prov_base["province"].map(norm_key)

prov_summary = prov_base.merge(prov_received, on="_province", how="left")
prov_summary["received_total"] = prov_summary["received_total"].fillna(0).astype(int)
prov_summary["received_male"] = prov_summary["received_male"].fillna(0).astype(int)
prov_summary["received_female"] = prov_summary["received_female"].fillna(0).astype(int)

prov_summary["remaining_total"] = (prov_summary["total"] - prov_summary["received_total"]).clip(lower=0).astype(int)
prov_summary["remaining_male"] = (prov_summary["male"] - prov_summary["received_male"]).clip(lower=0).astype(int)
prov_summary["remaining_female"] = (prov_summary["female"] - prov_summary["received_female"]).clip(lower=0).astype(int)

prov_summary["progress_total_pct"] = (prov_summary["received_total"] / prov_summary["total"].replace(0, pd.NA) * 100).fillna(0.0)
prov_summary["progress_male_pct"] = (prov_summary["received_male"] / prov_summary["male"].replace(0, pd.NA) * 100).fillna(0.0)
prov_summary["progress_female_pct"] = (prov_summary["received_female"] / prov_summary["female"].replace(0, pd.NA) * 100).fillna(0.0)

prov_summary_view = prov_summary[
    ["province", "total", "male", "female",
     "received_total", "received_male", "received_female",
     "remaining_total", "remaining_male", "remaining_female",
     "progress_total_pct"]
].sort_values("progress_total_pct", ascending=False)

prov_summary_view["progress_total_pct"] = prov_summary_view["progress_total_pct"].map(lambda x: round(float(x), 1))


# ============================================================
# Interviewer summary table (general)
# ============================================================
interviewer_summary = (
    fdf.groupby("_interviewer")
       .agg(
           received=("__dummy__", "size") if "__dummy__" in fdf.columns else ("_interviewer", "size"),
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

# Add top province per interviewer (for quick analysis)
top_prov = (
    fdf.groupby(["_interviewer", "_province_raw"])
       .size()
       .reset_index(name="n")
       .sort_values(["_interviewer", "n"], ascending=[True, False])
       .drop_duplicates(subset=["_interviewer"])
       .rename(columns={"_interviewer": "interviewer", "_province_raw": "top_province", "n": "top_province_count"})
)

interviewer_summary = interviewer_summary.merge(top_prov, on="interviewer", how="left")
interviewer_summary = interviewer_summary.sort_values("received", ascending=False)


# ============================================================
# Tabs
# ============================================================
tab_overview, tab_assistance, tab_disability, tab_interviewers, tab_provinces = st.tabs(
    ["Overview", "Assistance", "Disability", "Interviewers", "Provinces"]
)

# ============================================================
# OVERVIEW
# ============================================================
with tab_overview:
    st.markdown("### Targets")
    a1, a2, a3 = st.columns(3)
    with a1:
        kpi("Total Target", f"{TOTAL_TARGET:,}", "")
    with a2:
        kpi("Male Target", f"{MALE_TARGET:,}", "")
    with a3:
        kpi("Female Target", f"{FEMALE_TARGET:,}", "")

    st.markdown("### Received")
    b1, b2, b3 = st.columns(3)
    with b1:
        kpi("Received (Total)", f"{received_total:,}", "")
    with b2:
        kpi("Received (Male)", f"{received_male:,}", "")
    with b3:
        kpi("Received (Female)", f"{received_female:,}", "")

    st.markdown("### Remaining")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Remaining (Total)", f"{remaining_total:,}", "")
    with c2:
        kpi("Remaining (Male)", f"{remaining_male:,}", "")
    with c3:
        kpi("Remaining (Female)", f"{remaining_female:,}", "")

    st.markdown("### Progress")
    d1, d2, d3 = st.columns(3)
    with d1:
        kpi("Progress (Total)", f"{progress_total:.1f}%", "")
    with d2:
        kpi("Progress (Male)", f"{progress_male:.1f}%", "")
    with d3:
        kpi("Progress (Female)", f"{progress_female:.1f}%", "")

    st.markdown("### Progress Lines")
    left, right = st.columns([1, 1])
    with left:
        render_progress_line("Total", received_total, TOTAL_TARGET)
        render_progress_line("Male", received_male, MALE_TARGET)
        render_progress_line("Female", received_female, FEMALE_TARGET)
    with right:
        kpi("Errors (7d)", f"{recent_errors:,}", "")
        # Optional quick disability insight
        yes_count = int((fdf["_disability"] == "Yes").sum())
        yes_pct = (yes_count / max(1, len(fdf)) * 100)
        kpi("Disability (Yes)", f"{yes_pct:.1f}%", "")

    st.caption(f"Last data timestamp: {last_update_str} | Source: {TARGET_WORKSHEET}")


# ============================================================
# ASSISTANCE
# ============================================================
with tab_assistance:
    st.markdown("### Assistance Type Distribution")

    if ASSIST_COL not in fdf.columns:
        st.warning(f"Column not found: {ASSIST_COL}")
    else:
        top_n = st.slider("Top categories", min_value=5, max_value=25, value=12, step=1, key="assist_topn")
        dist_plot = value_counts_df(fdf["_assist"], top_n=top_n, other_label="Other")

        donut = (
            alt.Chart(dist_plot)
            .mark_arc(innerRadius=75, outerRadius=125)
            .encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color("category:N", legend=alt.Legend(title="Assistance type")),
                tooltip=[
                    alt.Tooltip("category:N", title="Type"),
                    alt.Tooltip("count:Q", title="Count"),
                    alt.Tooltip("pct:Q", title="Share", format=".1%"),
                ],
            )
            .properties(height=380)
        )

        bars = (
            alt.Chart(dist_plot.sort_values("count", ascending=True))
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("category:N", sort=None, title=""),
                tooltip=[
                    alt.Tooltip("category:N", title="Type"),
                    alt.Tooltip("count:Q", title="Count"),
                    alt.Tooltip("pct:Q", title="Share", format=".1%"),
                ],
            )
            .properties(height=380)
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.altair_chart(donut, use_container_width=True)
        with c2:
            st.altair_chart(bars, use_container_width=True)


# ============================================================
# DISABILITY
# ============================================================
with tab_disability:
    st.markdown("### Disability in Family (Yes/No)")

    if DISABILITY_COL not in fdf.columns:
        st.warning(f"Column not found: {DISABILITY_COL}")
    else:
        dist = value_counts_df(fdf["_disability"], top_n=10, other_label="Other")
        # Keep stable order if present
        order = {"Yes": 0, "No": 1, "Unknown": 2}
        dist["sort_key"] = dist["category"].map(lambda x: order.get(x, 99))
        dist = dist.sort_values(["sort_key", "count"], ascending=[True, False]).drop(columns=["sort_key"])

        donut = (
            alt.Chart(dist)
            .mark_arc(innerRadius=75, outerRadius=125)
            .encode(
                theta="count:Q",
                color=alt.Color("category:N", legend=alt.Legend(title="")),
                tooltip=["category", "count", alt.Tooltip("pct:Q", format=".1%")],
            )
            .properties(height=380)
        )

        bar = (
            alt.Chart(dist)
            .mark_bar()
            .encode(
                x=alt.X("category:N", title=""),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["category", "count", alt.Tooltip("pct:Q", format=".1%")],
            )
            .properties(height=380)
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.altair_chart(donut, use_container_width=True)
        with c2:
            st.altair_chart(bar, use_container_width=True)


# ============================================================
# INTERVIEWERS (General table + selector detail)
# ============================================================
with tab_interviewers:
    st.markdown("### Interviewers (General Overview)")
    if INTERVIEWER_COL not in fdf.columns:
        st.warning(f"Column not found: {INTERVIEWER_COL}")
    else:
        st.dataframe(
            interviewer_summary[
                ["interviewer", "received", "male", "female", "unknown_gender", "top_province", "top_province_count", "disability_yes_pct"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.markdown("### Interviewer Detail")

        interviewer_list = interviewer_summary["interviewer"].dropna().unique().tolist()
        interviewer_list = sorted(interviewer_list)
        selected_interviewer = st.selectbox("Select interviewer", interviewer_list)

        idf = fdf[fdf["_interviewer"] == selected_interviewer].copy()

        i_total = int(len(idf))
        i_male = int((idf["_gender"] == "male").sum())
        i_female = int((idf["_gender"] == "female").sum())
        i_unknown = max(0, i_total - i_male - i_female)

        st.markdown("#### KPIs")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi("Received", f"{i_total:,}", "")
        with k2:
            kpi("Male", f"{i_male:,}", "")
        with k3:
            kpi("Female", f"{i_female:,}", "")
        with k4:
            kpi("Unknown", f"{i_unknown:,}", "")

        # Keep only modern, analysis-useful charts
        c1, c2 = st.columns([1, 1])

        with c1:
            # Gender donut (compact + clear)
            gdf = pd.DataFrame(
                [
                    {"category": "Male", "count": i_male},
                    {"category": "Female", "count": i_female},
                    {"category": "Unknown", "count": i_unknown},
                ]
            )
            gdf["pct"] = gdf["count"] / max(1, int(gdf["count"].sum()))
            gdonut = (
                alt.Chart(gdf)
                .mark_arc(innerRadius=75, outerRadius=125)
                .encode(
                    theta="count:Q",
                    color=alt.Color("category:N", legend=alt.Legend(title="Gender")),
                    tooltip=["category", "count", alt.Tooltip("pct:Q", format=".1%")],
                )
                .properties(height=360, title="Gender Split")
            )
            st.altair_chart(gdonut, use_container_width=True)

        with c2:
            # Assistance distribution for interviewer (ranked bar)
            avc = value_counts_df(idf["_assist"], top_n=10, other_label="Other").sort_values("count", ascending=True)
            abar = (
                alt.Chart(avc)
                .mark_bar()
                .encode(
                    x=alt.X("count:Q", title="Count"),
                    y=alt.Y("category:N", sort=None, title=""),
                    tooltip=["category", "count", alt.Tooltip("pct:Q", format=".1%")],
                )
                .properties(height=360, title="Assistance Types")
            )
            st.altair_chart(abar, use_container_width=True)


# ============================================================
# PROVINCES (General table + selector detail)
# ============================================================
with tab_provinces:
    st.markdown("### Provinces (General Overview)")
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

    # Province cards: Targets / Received / Remaining / Progress
    st.markdown("#### Targets")
    t1, t2, t3 = st.columns(3)
    with t1:
        kpi("Target (Total)", f"{p_target_total:,}", "")
    with t2:
        kpi("Target (Male)", f"{p_target_male:,}", "")
    with t3:
        kpi("Target (Female)", f"{p_target_female:,}", "")

    st.markdown("#### Received")
    r1, r2, r3 = st.columns(3)
    with r1:
        kpi("Received (Total)", f"{p_received_total:,}", "")
    with r2:
        kpi("Received (Male)", f"{p_received_male:,}", "")
    with r3:
        kpi("Received (Female)", f"{p_received_female:,}", "")

    st.markdown("#### Remaining")
    m1, m2, m3 = st.columns(3)
    with m1:
        kpi("Remaining (Total)", f"{p_remaining_total:,}", "")
    with m2:
        kpi("Remaining (Male)", f"{p_remaining_male:,}", "")
    with m3:
        kpi("Remaining (Female)", f"{p_remaining_female:,}", "")

    st.markdown("#### Progress")
    pr1, pr2, pr3 = st.columns(3)
    with pr1:
        kpi("Progress (Total)", f"{p_progress_total:.1f}%", "")
    with pr2:
        kpi("Progress (Male)", f"{p_progress_male:.1f}%", "")
    with pr3:
        kpi("Progress (Female)", f"{p_progress_female:.1f}%", "")

    st.markdown("#### Progress Lines")
    render_progress_line("Total", p_received_total, p_target_total)
    render_progress_line("Male", p_received_male, p_target_male)
    render_progress_line("Female", p_received_female, p_target_female)

    # Keep charts minimal and analysis-oriented
    c1, c2 = st.columns([1, 1])
    with c1:
        avc = value_counts_df(pdf["_assist"], top_n=10, other_label="Other").sort_values("count", ascending=True)
        abar = (
            alt.Chart(avc)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Count"),
                y=alt.Y("category:N", sort=None, title=""),
                tooltip=["category", "count", alt.Tooltip("pct:Q", format=".1%")],
            )
            .properties(height=360, title="Assistance Types (This Province)")
        )
        st.altair_chart(abar, use_container_width=True)

    with c2:
        dvc = value_counts_df(pdf["_disability"], top_n=10, other_label="Other")
        ddonut = (
            alt.Chart(dvc)
            .mark_arc(innerRadius=75, outerRadius=125)
            .encode(
                theta="count:Q",
                color=alt.Color("category:N", legend=alt.Legend(title="")),
                tooltip=["category", "count", alt.Tooltip("pct:Q", format=".1%")],
            )
            .properties(height=360, title="Disability (This Province)")
        )
        st.altair_chart(ddonut, use_container_width=True)

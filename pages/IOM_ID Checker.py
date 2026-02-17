import time
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
page_header("🆔 IOM ID Checker", "Validate Raw Kobo IOM IDs against the Sample list and log missing IDs")

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
# Altair Global Theme (modern, glass-friendly)
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

SAMPLE_IOM_COL = "IOMID"


# ============================================================
# Utilities (robust + fast)
# ============================================================
def normalize_id(x) -> str:
    """Trim, collapse spaces, uppercase. None/blank -> ''."""
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    s = " ".join(s.split())
    return s.upper()


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
    Fast + stable Google Sheet loader (cached):
    - Uses get_all_values() to preserve headers safely, then de-dupes duplicates.
    """
    ws = get_worksheet(sa, spreadsheet_id, sheet_name)
    values = retry_api(ws.get_all_values)
    if not values:
        return pd.DataFrame()

    headers = _dedupe_headers(values[0])
    data = values[1:]
    if not data:
        return pd.DataFrame(columns=headers)

    return pd.DataFrame(data, columns=headers)


def get_or_create_ws(title: str, rows: int = 5000, cols: int = 30):
    """Return worksheet handle (create if missing). Uses cached spreadsheet + retry."""
    _, sh = get_spreadsheet(sa, spreadsheet_id)
    try:
        return retry_api(sh.worksheet, title)
    except WorksheetNotFound:
        return retry_api(sh.add_worksheet, title=title, rows=rows, cols=cols)


def chunked(lst, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ============================================================
# Modern chart helpers
# ============================================================
def donut_chart(df_: pd.DataFrame, title: str, color_range: list[str]):
    """
    df_ columns: label, count
    """
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

    # center label (match%)
    match_pct = 0.0
    if len(d) and (d["label"] == "Matched").any():
        match_pct = float(d.loc[d["label"] == "Matched", "pct"].iloc[0]) * 100.0
    center = (
        alt.Chart(pd.DataFrame({"t": [f"{match_pct:.0f}%"]}))
        .mark_text(fontSize=28, fontWeight=900, color="#E2E8F0")
        .encode(text="t:N")
    )

    return donut + center


def modern_barh(df_: pd.DataFrame, title: str, x_title: str = "Count", max_rows: int = 20):
    """
    df_ columns: label, count
    """
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


def trend_area_line(trend_df: pd.DataFrame, title: str):
    base = alt.Chart(trend_df).encode(
        x=alt.X("day:T", title="Date"),
        y=alt.Y("count:Q", title="Records"),
        tooltip=[alt.Tooltip("day:T", title="Date"), alt.Tooltip("count:Q", title="Count")],
    )
    area = base.mark_area(opacity=0.14).encode(color=alt.value(PDM_COLORS["indigo"]))
    line = base.mark_line(strokeWidth=3).encode(color=alt.value(PDM_COLORS["indigo"]))
    pts = base.mark_circle(size=70).encode(color=alt.value(PDM_COLORS["pink"]))
    return (area + line + pts).properties(height=280, title=title)


# ============================================================
# Controls
# ============================================================
st.sidebar.header("Controls")

if st.sidebar.button("Refresh data (clear cache)"):
    st.cache_data.clear()
    st.rerun()

ignore_blank_ids = st.sidebar.checkbox("Ignore blank IOM IDs", value=True)
use_normalization = st.sidebar.checkbox("Normalize IDs (trim + uppercase)", value=True)


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

# Required columns
missing_cols = []
if RAW_IOM_COL not in raw_df.columns:
    missing_cols.append(f"{RAW_SHEET}: '{RAW_IOM_COL}'")
if SAMPLE_IOM_COL not in sample_df.columns:
    missing_cols.append(f"{SAMPLE_SHEET}: '{SAMPLE_IOM_COL}'")

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
# Build normalized columns (vectorized)
# ============================================================
raw_ids = filtered_raw[RAW_IOM_COL].astype(str).fillna("").str.strip()
sample_ids = sample_df[SAMPLE_IOM_COL].astype(str).fillna("").str.strip()

if use_normalization:
    raw_norm = raw_ids.map(normalize_id)
    sample_norm = sample_ids.map(normalize_id)
else:
    raw_norm = raw_ids
    sample_norm = sample_ids

filtered_raw["_iom_norm"] = raw_norm
sample_df["_iom_norm"] = sample_norm

if ignore_blank_ids:
    check_df = filtered_raw[filtered_raw["_iom_norm"] != ""].copy()
else:
    check_df = filtered_raw.copy()

sample_set = set(sample_df["_iom_norm"].dropna().astype(str).tolist())

check_df["_match"] = check_df["_iom_norm"].isin(sample_set)

matched_df = check_df[check_df["_match"]].copy()
not_found_df = check_df[~check_df["_match"]].copy()

not_found_ids = not_found_df["_iom_norm"].value_counts().reset_index()
not_found_ids.columns = ["iom_id", "occurrences"]

dup_df = check_df["_iom_norm"].value_counts()
raw_dups = dup_df[dup_df > 1].reset_index()
raw_dups.columns = ["iom_id", "count"]


# ============================================================
# Tabs
# ============================================================
tab_overview, tab_not_found, tab_matched, tab_sync = st.tabs(
    ["Overview", "Not Found", "Matched", "Sync to IOM_Not_found"]
)

# ------------------ Overview ------------------
with tab_overview:
    total_raw = int(len(raw_df))
    total_in_range = int(len(filtered_raw))
    total_checked = int(len(check_df))
    total_matched = int(len(matched_df))
    total_not_found = int(len(not_found_df))
    unique_not_found = int(len(not_found_ids))
    dup_count = int(len(raw_dups))

    st.markdown("### Summary")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi("Raw Records", f"{total_raw:,}", "")
    with c2:
        kpi("In Range", f"{total_in_range:,}", "Filtered window" if date_range_applied else "No date filter")
    with c3:
        kpi("Checked IDs", f"{total_checked:,}", "")
    with c4:
        kpi("Matched", f"{total_matched:,}", "")
    with c5:
        kpi("Not Found", f"{total_not_found:,}", f"Unique missing: {unique_not_found:,}")
    with c6:
        kpi("Duplicate IDs", f"{dup_count:,}", "")

    st.markdown("---")

    # Modern donut + center percent
    breakdown = pd.DataFrame(
        [{"label": "Matched", "count": total_matched}, {"label": "Not Found", "count": total_not_found}]
    )
    st.altair_chart(
        donut_chart(
            breakdown,
            title="Match Rate",
            color_range=[PDM_COLORS["green"], PDM_COLORS["red"]],
        ),
        use_container_width=True,
    )

    # Optional: trend by day (if date column exists)
    if RAW_DATE_COL in filtered_raw.columns and pd.api.types.is_datetime64_any_dtype(filtered_raw[RAW_DATE_COL]):
        st.markdown("### Trend (Matched vs Not Found)")
        tdf = check_df[[RAW_DATE_COL, "_match"]].copy()
        tdf = tdf[tdf[RAW_DATE_COL].notna()].copy()
        if not tdf.empty:
            tdf["day"] = tdf[RAW_DATE_COL].dt.date
            agg = tdf.groupby(["day", "_match"]).size().reset_index(name="count")
            agg["day"] = pd.to_datetime(agg["day"])
            agg["_match"] = agg["_match"].map({True: "Matched", False: "Not Found"})
            trend = (
                alt.Chart(agg)
                .mark_line(strokeWidth=3)
                .encode(
                    x=alt.X("day:T", title="Date"),
                    y=alt.Y("count:Q", title="Records"),
                    color=alt.Color(
                        "_match:N",
                        scale=alt.Scale(range=[PDM_COLORS["green"], PDM_COLORS["red"]]),
                        legend=alt.Legend(title=""),
                    ),
                    tooltip=[alt.Tooltip("day:T", title="Date"), alt.Tooltip("_match:N", title=""), alt.Tooltip("count:Q", title="Count")],
                )
                .properties(height=280, title="Daily match diagnostics")
            )
            st.altair_chart(trend, use_container_width=True)
        else:
            st.info("Trend skipped: no valid dates in selected range.")

    if dup_count:
        st.markdown("### Duplicate IDs in Raw Data (Top 25)")
        st.dataframe(raw_dups.sort_values("count", ascending=False).head(25), use_container_width=True, hide_index=True)

# ------------------ Not Found ------------------
with tab_not_found:
    st.markdown("### Missing IDs (Unique list)")

    # Show a modern bar for top missing IDs
    if not not_found_ids.empty:
        top_missing = not_found_ids.head(20).rename(columns={"iom_id": "label", "occurrences": "count"})
        st.altair_chart(modern_barh(top_missing, "Top missing IDs (by occurrences)", x_title="Occurrences", max_rows=20), use_container_width=True)

    st.dataframe(
        not_found_ids.sort_values("occurrences", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download missing IDs (CSV)",
        data=not_found_ids.to_csv(index=False).encode("utf-8"),
        file_name="iom_not_found_ids.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("### Raw rows containing missing IDs (Preview)")

    preview_cols = [RAW_IOM_COL]
    for c in [RAW_DATE_COL, RAW_INTERVIEWER_COL, RAW_PROVINCE_COL]:
        if c in not_found_df.columns:
            preview_cols.append(c)

    preview = not_found_df[preview_cols].copy()
    preview["iom_id"] = not_found_df["_iom_norm"]
    st.dataframe(preview.head(300), use_container_width=True, hide_index=True)

# ------------------ Matched ------------------
with tab_matched:
    st.markdown("### Matched IDs (Preview)")

    preview_cols = [RAW_IOM_COL]
    for c in [RAW_DATE_COL, RAW_INTERVIEWER_COL, RAW_PROVINCE_COL]:
        if c in matched_df.columns:
            preview_cols.append(c)

    m = matched_df[preview_cols].copy()
    m["iom_id"] = matched_df["_iom_norm"]
    st.dataframe(m.head(500), use_container_width=True, hide_index=True)

# ------------------ Sync ------------------
with tab_sync:
    st.markdown("### Sync missing IDs to `IOM_Not_found`")

    st.info(
        "This will append only NEW missing IDs (no duplicates) into `IOM_Not_found`. "
        "The sheet is created automatically if it does not exist."
    )

    include_context = st.checkbox("Include context columns (date/interviewer/province)", value=True)
    include_occurrences = st.checkbox("Include occurrence count", value=True)

    if st.button("Sync now", type="primary"):
        try:
            nf_ws = get_or_create_ws(NOT_FOUND_SHEET, rows=5000, cols=30)

            # Load existing not-found sheet quickly (no cache here, we need latest)
            existing_vals = _retry(lambda: nf_ws.get_all_values())
            if existing_vals:
                ex_headers = _dedupe_headers(existing_vals[0])
                ex_data = existing_vals[1:]
                ex_df = pd.DataFrame(ex_data, columns=ex_headers) if ex_data else pd.DataFrame(columns=ex_headers)
            else:
                ex_df = pd.DataFrame()

            # Detect existing ID column
            existing_ids = set()
            if not ex_df.empty:
                id_col = None
                for cand in ["iom_id", "IOMID", "IOM_ID", RAW_IOM_COL]:
                    if cand in ex_df.columns:
                        id_col = cand
                        break
                if id_col:
                    existing_ids = set(ex_df[id_col].map(normalize_id).tolist())

            # Build rows to add (one row per missing ID)
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            ctx_cols = []
            if include_context:
                for c in [RAW_DATE_COL, RAW_INTERVIEWER_COL, RAW_PROVINCE_COL]:
                    if c in not_found_df.columns:
                        ctx_cols.append(c)

            ctx_first = not_found_df.copy()
            if RAW_DATE_COL in ctx_first.columns:
                ctx_first[RAW_DATE_COL] = pd.to_datetime(ctx_first[RAW_DATE_COL], errors="coerce")
                ctx_first = ctx_first.sort_values(RAW_DATE_COL)

            ctx_first = ctx_first.assign(iom_id=ctx_first["_iom_norm"]).drop_duplicates(subset=["iom_id"], keep="first")

            out_df = ctx_first[["iom_id"] + ctx_cols].copy()

            if include_occurrences:
                out_df = out_df.merge(not_found_ids, left_on="iom_id", right_on="iom_id", how="left")

            out_df.insert(0, "timestamp_utc", ts)

            out_df["_iom_norm"] = out_df["iom_id"].map(normalize_id)
            to_add = out_df[~out_df["_iom_norm"].isin(existing_ids)].drop(columns=["_iom_norm"])

            if to_add.empty:
                st.success("No new missing IDs to add. `IOM_Not_found` is already up to date.")
            else:
                # Ensure header exists
                if not existing_vals:
                    _retry(lambda: nf_ws.append_row(list(to_add.columns), value_input_option="RAW"))

                # Append in chunks (safer for Google API limits)
                rows = to_add.fillna("").astype(str).values.tolist()
                for part in chunked(rows, 300):
                    _retry(lambda p=part: nf_ws.append_rows(p, value_input_option="RAW"))

                st.success(f"Synced {len(rows):,} new missing IDs to `{NOT_FOUND_SHEET}`.")

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
        st.info("`IOM_Not_found` does not exist yet. Click **Sync now** to create it.")
    except Exception as e:
        st.warning(f"Could not load `{NOT_FOUND_SHEET}` preview: {e}")

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

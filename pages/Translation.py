import time
import streamlit as st
import pandas as pd

from design.theme import load_css, page_header
from services.sheets import (
    get_client_from_secrets,
    looks_non_english,
    batch_update_new_values,
)

st.set_page_config(page_title="Translation | PDM", page_icon="🌐", layout="wide")
load_css()
page_header("🌐 Translation", "Translate pending items and build Clean_Data row on Save")


# -----------------------------
# Retry wrapper (429-safe)
# -----------------------------

def retry_api(fn, *args, **kwargs):
    """
    Retries Google Sheets calls on 429/quota errors with exponential backoff.
    """
    delays = [0.6, 1.2, 2.4, 4.8, 9.6]
    last_err = None
    for d in delays:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            msg = str(e)
            if "429" in msg or "Quota exceeded" in msg or "Read requests" in msg:
                time.sleep(d)
                continue
            raise
    raise last_err


# -----------------------------
# UUID normalization (critical fix)
# -----------------------------

def normalize_uuid(s: str) -> str:
    """
    Normalizes UUID strings to avoid mismatches caused by invisible/BOM chars.
    """
    if s is None:
        return ""
    s = str(s)
    # Remove common invisible chars / BOM / RTL marks
    s = s.replace("\ufeff", "")  # BOM
    s = s.replace("\u200c", "")  # ZWNJ
    s = s.replace("\u200d", "")  # ZWJ
    s = s.replace("\u200e", "")  # LRM
    s = s.replace("\u200f", "")  # RLM
    s = s.replace("\u202a", "").replace("\u202b", "").replace("\u202c", "")  # bidi embedding
    return s.strip()


# -----------------------------
# Cached resources (NO repeated open_by_key)
# -----------------------------

@st.cache_resource
def get_spreadsheet(sa: dict, spreadsheet_id: str):
    """
    Opens the spreadsheet ONCE per Streamlit process (cached resource).
    Prevents repeated metadata reads that cause 429 errors.
    """
    gc = get_client_from_secrets(sa)
    sh = retry_api(gc.open_by_key, spreadsheet_id)
    return gc, sh


@st.cache_resource
def get_worksheets(sa: dict, spreadsheet_id: str):
    """
    Returns worksheet objects, cached.
    """
    _, sh = get_spreadsheet(sa, spreadsheet_id)
    return {
        "Correction_Log": retry_api(sh.worksheet, "Correction_Log"),
        "Raw_Kobo_Data": retry_api(sh.worksheet, "Raw_Kobo_Data"),
        "Clean_Data": retry_api(sh.worksheet, "Clean_Data"),
    }


# -----------------------------
# UI components
# -----------------------------

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


# -----------------------------
# Helpers
# -----------------------------

def safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df))
    return df[col].astype(str).fillna("").map(lambda x: str(x).strip())


def _col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


# -----------------------------
# Cached data reads (unhashable-safe)
# -----------------------------

@st.cache_data(show_spinner=False, ttl=120)
def load_correction_log_df_cached(_corr_ws) -> pd.DataFrame:
    """
    Cached load of Correction_Log as dataframe with _row.
    _corr_ws underscore avoids hashing Worksheet object.
    """
    values = retry_api(_corr_ws.get_all_values)
    if not values or len(values) < 2:
        return pd.DataFrame()

    headers = [str(h).strip() for h in values[0]]
    rows = values[1:]

    data = []
    for sheet_row_num, r in enumerate(rows, start=2):
        row_dict = {"_row": sheet_row_num}
        for i, h in enumerate(headers):
            row_dict[h] = r[i].strip() if i < len(r) else ""
        data.append(row_dict)

    return pd.DataFrame(data)


@st.cache_data(show_spinner=False, ttl=300)
def build_pending_uuid_list_cached(df_corr: pd.DataFrame) -> list[str]:
    """
    Only UUIDs with pending translations (new_value empty).
    Uses normalize_uuid to avoid invisible-char mismatches.
    """
    if df_corr.empty or "_uuid" not in df_corr.columns or "new_value" not in df_corr.columns:
        return []

    uu_raw = safe_col(df_corr, "_uuid").map(normalize_uuid)
    newv = safe_col(df_corr, "new_value")

    pending = df_corr[(uu_raw != "") & (newv == "")]
    if pending.empty:
        return []

    uuids = safe_col(pending, "_uuid").map(normalize_uuid)
    uuids = uuids[uuids != ""].unique().tolist()
    uuids.sort()
    return uuids


@st.cache_data(show_spinner=False, ttl=300)
def get_headers_cached(_ws) -> list[str]:
    headers = retry_api(_ws.row_values, 1)
    return [str(h).strip() for h in headers if str(h).strip()]


@st.cache_data(show_spinner=False, ttl=300)
def get_uuid_column_cached(_ws, uuid_col: str = "_uuid") -> list[str]:
    """
    Reads only the UUID column (data rows only), normalized.
    """
    headers = retry_api(_ws.row_values, 1)
    headers = [str(h).strip() for h in headers]
    if uuid_col not in headers:
        return []
    idx = headers.index(uuid_col) + 1
    vals = retry_api(_ws.col_values, idx)
    # normalize values
    return [normalize_uuid(v) for v in vals[1:]]


# -----------------------------
# Clean_Data dedupe in session_state
# -----------------------------

def ensure_clean_uuid_set(clean_ws):
    if "clean_uuid_set" in st.session_state:
        return
    vals = get_uuid_column_cached(clean_ws, "_uuid")
    st.session_state["clean_uuid_set"] = set(v for v in vals if v)


def clean_contains_uuid(clean_ws, uuid_value: str) -> bool:
    ensure_clean_uuid_set(clean_ws)
    return normalize_uuid(uuid_value) in st.session_state["clean_uuid_set"]


def add_uuid_to_clean_set(clean_ws, uuid_value: str):
    ensure_clean_uuid_set(clean_ws)
    st.session_state["clean_uuid_set"].add(normalize_uuid(uuid_value))


# -----------------------------
# Build Clean_Data row on Save
# -----------------------------

def build_replacements(df_uuid_all: pd.DataFrame) -> dict:
    """
    {Question: new_value} for the UUID (non-empty only).
    """
    repl = {}
    q = safe_col(df_uuid_all, "Question")
    nv = safe_col(df_uuid_all, "new_value")
    for qq, vv in zip(q.tolist(), nv.tolist()):
        if qq and vv:
            repl[qq] = vv
    return repl


def fetch_raw_rows_for_uuid(raw_ws, uuid_value: str, raw_headers: list[str], uuid_col: str = "_uuid") -> list[list[str]]:
    """
    Finds row numbers using normalized cached _uuid column, then fetches only those rows by range.
    """
    target = normalize_uuid(uuid_value)
    uuid_vals = get_uuid_column_cached(raw_ws, uuid_col)  # already normalized

    matches = [i + 2 for i, v in enumerate(uuid_vals) if v == target]  # data starts at row 2
    if not matches:
        return []

    last_col = _col_letter(len(raw_headers))
    out = []
    for rnum in matches:
        rng = f"A{rnum}:{last_col}{rnum}"
        row = retry_api(raw_ws.get, rng)
        out.append(row[0] if row else [""] * len(raw_headers))
    return out


def ensure_clean_headers_like_raw(clean_ws, raw_headers: list[str]) -> list[str]:
    existing = retry_api(clean_ws.row_values, 1)
    if not any(existing):
        retry_api(clean_ws.update, "A1", [raw_headers])
        return raw_headers
    headers = [str(h).strip() for h in existing if str(h).strip()]
    return headers if headers else raw_headers


def apply_replacements(raw_headers: list[str], raw_row: list[str], repl: dict) -> list[str]:
    """
    Replace only columns named in repl; keep all other values unchanged.
    """
    row_map = {raw_headers[i]: (raw_row[i] if i < len(raw_row) else "") for i in range(len(raw_headers))}
    for col_name, new_val in repl.items():
        if col_name in row_map:
            row_map[col_name] = new_val
    return [row_map.get(h, "") for h in raw_headers]


# -----------------------------
# Secrets / worksheets
# -----------------------------

try:
    sa = dict(st.secrets["gcp_service_account"])
    spreadsheet_id = st.secrets["app"]["spreadsheet_id"]
except Exception:
    st.error("Secrets are not configured. Please set .streamlit/secrets.toml (service account + spreadsheet_id).")
    st.stop()

ws_map = get_worksheets(sa, spreadsheet_id)
corr_ws = ws_map["Correction_Log"]
raw_ws = ws_map["Raw_Kobo_Data"]
clean_ws = ws_map["Clean_Data"]

# Load Correction_Log (cached)
df_corr = load_correction_log_df_cached(corr_ws)
if df_corr.empty:
    st.warning("Correction_Log is empty or missing headers.")
    st.stop()

required_cols = {"_row", "_uuid", "Question", "old_value", "new_value"}
missing = [c for c in required_cols if c not in df_corr.columns]
if missing:
    st.error(f"Correction_Log is missing required columns: {missing}")
    st.stop()

# UUID list: ONLY UUIDs with pending items (normalized)
pending_uuid_list = build_pending_uuid_list_cached(df_corr)
if not pending_uuid_list:
    st.success("No pending translations found.")
    st.stop()


# -----------------------------
# UUID selection
# -----------------------------

st.markdown("### Select UUID")

uuid_value = st.selectbox(
    "UUID (pending translations only)",
    options=[""] + pending_uuid_list,
    index=0,
)

uuid_value = normalize_uuid(uuid_value)

if not uuid_value:
    st.info("Select a UUID to load pending items.")
    st.stop()

# Pending rows for this UUID (new_value empty) using normalized match
df_corr_norm = df_corr.copy()
df_corr_norm["_uuid_norm"] = safe_col(df_corr_norm, "_uuid").map(normalize_uuid)

df_uuid_pending = df_corr_norm[
    (df_corr_norm["_uuid_norm"] == uuid_value) &
    (safe_col(df_corr_norm, "new_value") == "")
].copy()

if df_uuid_pending.empty:
    st.success("No pending items for this UUID.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
with c1:
    mini_card("Pending items", f"{len(df_uuid_pending):,}", "new_value is empty")
with c2:
    mini_card("UUID", uuid_value, "Selected key")
with c3:
    mini_card("Validation", "English-only", "Arabic-script blocked in new_value")
with c4:
    mini_card("Clean_Data", "On Save", "Appends updated row(s)")

st.markdown("---")


# -----------------------------
# Editor
# -----------------------------

st.markdown("### Translate and Save")

edit_df = df_uuid_pending[["_row", "Question", "old_value", "new_value"]].copy()

edited = st.data_editor(
    edit_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "_row": st.column_config.NumberColumn("_row", disabled=True, help="Internal sheet row number"),
        "Question": st.column_config.TextColumn("Question", disabled=True),
        "old_value": st.column_config.TextColumn("old_value", disabled=True),
        "new_value": st.column_config.TextColumn("new_value", help="Enter English translation"),
    },
    disabled=["_row", "Question", "old_value"],
    column_order=["Question", "old_value", "new_value"],
)

row_nums = edited["_row"].astype(int).tolist()
new_vals = edited["new_value"].astype(str).fillna("").str.strip().tolist()

invalid_rows = []
empty_rows = []
updates = []

for rnum, new_val in zip(row_nums, new_vals):
    if not new_val:
        empty_rows.append(rnum)
        continue
    if looks_non_english(new_val):
        invalid_rows.append(rnum)
        continue
    updates.append((rnum, new_val))

a1, a2 = st.columns([1, 1.4])
with a1:
    save = st.button("Save", type="primary")
with a2:
    st.markdown(
        f"<div style='opacity:0.72; padding-top:8px;'>Ready: <b>{len(updates)}</b> | Empty: <b>{len(empty_rows)}</b> | Invalid: <b>{len(invalid_rows)}</b></div>",
        unsafe_allow_html=True,
    )

if invalid_rows:
    st.error(
        "Some new_value entries contain Arabic-script characters and cannot be saved. "
        f"Fix these rows first: {', '.join(map(str, invalid_rows[:30]))}"
        + (" ..." if len(invalid_rows) > 30 else "")
    )

if save:
    if empty_rows:
        st.warning("Fill all pending new_value cells before saving.")
        st.stop()
    if invalid_rows:
        st.stop()

    # 1) Update Correction_Log
    with st.spinner("Updating Correction_Log..."):
        try:
            retry_api(batch_update_new_values, corr_ws, updates=updates, new_value_col="new_value")
        except Exception as e:
            st.exception(e)
            st.stop()

    # Clear relevant caches
    load_correction_log_df_cached.clear()
    build_pending_uuid_list_cached.clear()

    # 2) Build + append Clean_Data row(s)
    with st.spinner("Building Clean_Data row..."):
        try:
            # Reload Correction_Log once (fresh) to build replacements
            df_corr2 = load_correction_log_df_cached(corr_ws)
            df_corr2["_uuid_norm"] = safe_col(df_corr2, "_uuid").map(normalize_uuid)
            df_uuid_all = df_corr2[df_corr2["_uuid_norm"] == uuid_value].copy()

            if df_uuid_all.empty:
                st.error("UUID not found in Correction_Log after save. Clean_Data not updated.")
                st.stop()

            if (safe_col(df_uuid_all, "new_value") == "").any():
                st.warning("Some items for this UUID are still pending. Clean_Data was not updated.")
                st.stop()

            if safe_col(df_uuid_all, "new_value").apply(looks_non_english).any():
                st.warning("Some new_value entries are not English-only. Clean_Data was not updated.")
                st.stop()

            ensure_clean_uuid_set(clean_ws)
            if clean_contains_uuid(clean_ws, uuid_value):
                st.info("Clean_Data already contains this UUID. No new row was appended.")
            else:
                raw_headers = get_headers_cached(raw_ws)
                if not raw_headers:
                    st.error("Raw_Kobo_Data headers not found.")
                    st.stop()

                raw_rows = fetch_raw_rows_for_uuid(raw_ws, uuid_value, raw_headers, uuid_col="_uuid")
                if not raw_rows:
                    # Diagnostics (no extra heavy reads): we already have normalized uuid column cached
                    raw_uuid_vals = get_uuid_column_cached(raw_ws, "_uuid")
                    st.error("Raw_Kobo_Data row not found for this UUID.")
                    st.caption(f"Raw UUID count (cached): {len(raw_uuid_vals):,}")
                    st.caption("This usually indicates an invisible character mismatch or the UUID is not present in Raw_Kobo_Data.")
                    st.stop()

                clean_headers = ensure_clean_headers_like_raw(clean_ws, raw_headers)
                repl = build_replacements(df_uuid_all)

                out_rows = []
                for rr in raw_rows:
                    updated_row_raw_order = apply_replacements(raw_headers, rr, repl)

                    if clean_headers == raw_headers:
                        out_rows.append(updated_row_raw_order)
                    else:
                        row_map = {raw_headers[i]: (updated_row_raw_order[i] if i < len(updated_row_raw_order) else "") for i in range(len(raw_headers))}
                        out_rows.append([row_map.get(h, "") for h in clean_headers])

                retry_api(clean_ws.append_rows, out_rows, value_input_option="USER_ENTERED")

                add_uuid_to_clean_set(clean_ws, uuid_value)
                st.success(f"Saved translations and appended {len(out_rows)} row(s) into Clean_Data.")

        except Exception as e:
            st.exception(e)
            st.stop()

    st.rerun()

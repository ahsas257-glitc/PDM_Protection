import time
import uuid as uuidlib
from datetime import datetime, timezone, timedelta

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

LOCK_SHEET_NAME = "Locks"
LOCK_TTL_MINUTES = 15  # قفل بعد از 15 دقیقه خودکار منقضی می‌شود


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
    Ensures Locks worksheet exists.
    """
    _, sh = get_spreadsheet(sa, spreadsheet_id)

    def _get_or_create(name: str, rows: int = 2000, cols: int = 10):
        try:
            return retry_api(sh.worksheet, name)
        except Exception:
            # create sheet if not exists
            ws = retry_api(sh.add_worksheet, title=name, rows=rows, cols=cols)
            return ws

    corr = _get_or_create("Correction_Log")
    raw = _get_or_create("Raw_Kobo_Data")
    clean = _get_or_create("Clean_Data")
    locks = _get_or_create(LOCK_SHEET_NAME)

    # ensure Locks header
    header = retry_api(locks.row_values, 1)
    header = [str(h).strip() for h in header]
    required = ["_uuid", "locked_by", "locked_at", "expires_at"]
    if header != required:
        # overwrite header only if sheet is empty or header incorrect
        vals = retry_api(locks.get_all_values)
        if not vals:
            retry_api(locks.update, "A1", [required])
        else:
            # اگر شیت Locks قبلاً داده دارد ولی header متفاوت است، همینجا متوقف می‌کنیم
            # تا دیتای قبلی خراب نشود.
            if set(required).issubset(set(header)) is False:
                st.error(
                    f"Worksheet '{LOCK_SHEET_NAME}' exists but header is not compatible.\n"
                    f"Expected: {required}\nFound: {header}\n"
                    f"Please fix the header manually."
                )
                st.stop()

    return {
        "Correction_Log": corr,
        "Raw_Kobo_Data": raw,
        "Clean_Data": clean,
        "Locks": locks,
    }


# -----------------------------
# UI components
# -----------------------------
def mini_card(
    title: str,
    value: str,
    subtitle: str = "",
    *,
    title_size: str = "0.78rem",
    value_size: str = "1.35rem",
    subtitle_size: str = "0.78rem",
    value_wrap: bool = True,
    card_min_height: str = "78px",
):
    """
    value_wrap=False => keep value in one line with ellipsis
    """
    value_css = (
        "white-space: normal; overflow-wrap: anywhere;"
        if value_wrap
        else "white-space: nowrap; overflow: hidden; text-overflow: ellipsis;"
    )

    st.markdown(
        f"""
        <div class="pdm-card" style="padding:10px 12px; min-height:{card_min_height};">
          <div style="font-size:{title_size}; opacity:0.72; line-height:1.1;">{title}</div>
          <div style="font-size:{value_size}; font-weight:800; margin-top:6px; line-height:1.1; {value_css}">{value}</div>
          <div style="font-size:{subtitle_size}; opacity:0.62; margin-top:6px; line-height:1.2;">{subtitle}</div>
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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


# -----------------------------
# User identity (for locking)
# -----------------------------
def get_user_id() -> str:
    """
    Best-effort user identifier.
    If Streamlit auth provides st.experimental_user, use it; else use a per-session UUID.
    """
    # Streamlit Cloud / auth may provide this (depends on deployment)
    u = getattr(st, "experimental_user", None)
    if u and getattr(u, "email", None):
        return str(u.email)

    # fallback: stable within session
    if "session_user_id" not in st.session_state:
        st.session_state["session_user_id"] = f"session-{uuidlib.uuid4().hex[:12]}"
    return st.session_state["session_user_id"]


# -----------------------------
# Locks (shared across users via Google Sheet)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=10)
def load_locks_df_cached(_locks_ws) -> pd.DataFrame:
    values = retry_api(_locks_ws.get_all_values)
    if not values or len(values) < 2:
        return pd.DataFrame(columns=["_row", "_uuid", "locked_by", "locked_at", "expires_at"])

    headers = [str(h).strip() for h in values[0]]
    rows = values[1:]
    data = []
    for sheet_row_num, r in enumerate(rows, start=2):
        row_dict = {"_row": sheet_row_num}
        for i, h in enumerate(headers):
            row_dict[h] = r[i].strip() if i < len(r) else ""
        data.append(row_dict)

    df = pd.DataFrame(data)
    if "_uuid" in df.columns:
        df["_uuid"] = df["_uuid"].map(normalize_uuid)
    return df


def parse_dt(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        # expects ISO like 2026-02-17T10:00:00+00:00
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def cleanup_expired_locks(locks_ws):
    """
    Deletes expired rows from Locks to avoid stale locks.
    Best-effort; runs on page usage.
    """
    df = load_locks_df_cached(locks_ws)
    if df.empty:
        return

    now = utc_now()
    expired_rows = []
    for _, row in df.iterrows():
        exp = parse_dt(row.get("expires_at", ""))
        if exp and exp <= now:
            expired_rows.append(int(row["_row"]))

    # delete bottom-up to keep row numbers stable
    if expired_rows:
        for r in sorted(expired_rows, reverse=True):
            try:
                retry_api(locks_ws.delete_rows, r)
            except Exception:
                # Best effort; ignore
                pass
        load_locks_df_cached.clear()


def is_uuid_locked_by_other(locks_ws, uuid_value: str, user_id: str) -> bool:
    df = load_locks_df_cached(locks_ws)
    if df.empty:
        return False

    now = utc_now()
    target = normalize_uuid(uuid_value)

    # keep only non-expired
    df2 = df.copy()
    df2["exp_dt"] = df2["expires_at"].map(parse_dt)
    df2 = df2[df2["exp_dt"].notna() & (df2["exp_dt"] > now)]

    hit = df2[df2["_uuid"] == target]
    if hit.empty:
        return False

    # if any active lock not by me => locked by other
    return any(safe_col(hit, "locked_by") != user_id)


def acquire_lock(locks_ws, uuid_value: str, user_id: str) -> bool:
    """
    Try to lock UUID. Returns True if acquired or already locked by me.
    """
    cleanup_expired_locks(locks_ws)

    df = load_locks_df_cached(locks_ws)
    target = normalize_uuid(uuid_value)
    now = utc_now()
    exp = now + timedelta(minutes=LOCK_TTL_MINUTES)

    if df.empty:
        # append new lock
        retry_api(
            locks_ws.append_rows,
            [[target, user_id, iso(now), iso(exp)]],
            value_input_option="USER_ENTERED",
        )
        load_locks_df_cached.clear()
        return True

    # check existing active lock
    df2 = df.copy()
    df2["exp_dt"] = df2["expires_at"].map(parse_dt)
    df2_active = df2[df2["exp_dt"].notna() & (df2["exp_dt"] > now)]
    hit = df2_active[df2_active["_uuid"] == target]

    if not hit.empty:
        # already locked; if mine -> refresh TTL
        locked_by = str(hit.iloc[0].get("locked_by", "")).strip()
        rownum = int(hit.iloc[0]["_row"])
        if locked_by == user_id:
            # refresh expiry (and locked_at)
            retry_api(locks_ws.update, f"C{rownum}", [[iso(now)]])   # locked_at
            retry_api(locks_ws.update, f"D{rownum}", [[iso(exp)]])   # expires_at
            load_locks_df_cached.clear()
            return True
        return False

    # no active lock: add lock
    retry_api(
        locks_ws.append_rows,
        [[target, user_id, iso(now), iso(exp)]],
        value_input_option="USER_ENTERED",
    )
    load_locks_df_cached.clear()
    return True


def release_lock(locks_ws, uuid_value: str, user_id: str):
    """
    Release lock if owned by user.
    """
    df = load_locks_df_cached(locks_ws)
    if df.empty:
        return
    target = normalize_uuid(uuid_value)

    # find any row with this uuid and locked_by == me (delete)
    hit = df[(safe_col(df, "_uuid") == target) & (safe_col(df, "locked_by") == user_id)]
    if hit.empty:
        return

    # delete bottom-up
    for r in sorted(hit["_row"].astype(int).tolist(), reverse=True):
        try:
            retry_api(locks_ws.delete_rows, r)
        except Exception:
            pass
    load_locks_df_cached.clear()


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
locks_ws = ws_map["Locks"]

user_id = get_user_id()

# clean expired locks occasionally
cleanup_expired_locks(locks_ws)

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
pending_uuid_list_all = build_pending_uuid_list_cached(df_corr)
if not pending_uuid_list_all:
    st.success("No pending translations found.")
    st.stop()

# Filter out UUIDs locked by other users
pending_uuid_list = []
for u in pending_uuid_list_all:
    if not is_uuid_locked_by_other(locks_ws, u, user_id):
        pending_uuid_list.append(u)

# If everything is locked
if not pending_uuid_list:
    st.info("All pending UUIDs are currently being processed by other users (locked). Please refresh later.")
    st.stop()


# -----------------------------
# UUID selection (with locking)
# -----------------------------
st.markdown("### Select UUID")

# if user had a lock and refresh happens, keep it visible
my_locked_uuid = st.session_state.get("locked_uuid", "")
if my_locked_uuid and my_locked_uuid not in pending_uuid_list:
    # keep my lock in list so user can continue
    pending_uuid_list = [my_locked_uuid] + pending_uuid_list

selected = st.selectbox(
    "UUID (pending translations only, excludes locked items)",
    options=[""] + pending_uuid_list,
    index=0 if not my_locked_uuid else (0 if my_locked_uuid == "" else 1),
    key="uuid_select",
)

uuid_value = normalize_uuid(selected)

# release previous lock if user changed UUID
prev_uuid = normalize_uuid(st.session_state.get("locked_uuid", ""))
if prev_uuid and uuid_value and prev_uuid != uuid_value:
    release_lock(locks_ws, prev_uuid, user_id)
    st.session_state["locked_uuid"] = ""

if not uuid_value:
    st.info("Select a UUID to load pending items.")
    st.stop()

# acquire lock for this uuid
ok = acquire_lock(locks_ws, uuid_value, user_id)
if not ok:
    st.error("This UUID is currently selected by another user. Choose another UUID.")
    st.session_state["uuid_select"] = ""
    st.rerun()

# record lock in session
st.session_state["locked_uuid"] = uuid_value

# Pending rows for this UUID (new_value empty) using normalized match
df_corr_norm = df_corr.copy()
df_corr_norm["_uuid_norm"] = safe_col(df_corr_norm, "_uuid").map(normalize_uuid)

df_uuid_pending = df_corr_norm[
    (df_corr_norm["_uuid_norm"] == uuid_value) &
    (safe_col(df_corr_norm, "new_value") == "")
].copy()

if df_uuid_pending.empty:
    st.success("No pending items for this UUID.")
    # release lock since nothing to do
    release_lock(locks_ws, uuid_value, user_id)
    st.session_state["locked_uuid"] = ""
    st.stop()

# ✅ کارت UUID بزرگتر، بقیه کوچک‌تر
c1, c2, c3, c4 = st.columns([1, 2.4, 1, 1])
with c1:
    mini_card("Pending items", f"{len(df_uuid_pending):,}", title_size="0.72rem", value_size="1.05rem", subtitle="")
with c2:
    mini_card("UUID (locked)", uuid_value, f"Locked by: {user_id}",
              title_size="0.8rem", value_size="1.45rem", subtitle_size="0.76rem",
              value_wrap=False, card_min_height="90px")
with c3:
    mini_card("Validation", "English-only", title_size="0.72rem", value_size="1.05rem", subtitle="")
with c4:
    mini_card("Clean_Data", "On Save", title_size="0.72rem", value_size="1.05rem", subtitle="")

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
        finally:
            # Always release lock on save attempt (success path already handled)
            release_lock(locks_ws, uuid_value, user_id)
            st.session_state["locked_uuid"] = ""
            load_locks_df_cached.clear()

    st.session_state["uuid_select"] = ""
    st.rerun()

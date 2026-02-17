import time
import uuid as uuidlib
import random
import socket
from datetime import datetime, timezone, timedelta

import streamlit as st
import pandas as pd
import bcrypt
import requests
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

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
LOCK_TTL_MINUTES = 15

USERS_SHEET_NAME = "Users"
AUDIT_SHEET_NAME = "Audit_Log"


# -----------------------------
# Retry wrapper (429 + network-safe)
# -----------------------------
def retry_api(fn, *args, **kwargs):
    """
    Retries Google Sheets calls on 429/quota AND transient network errors.
    """
    delays = [0.6, 1.2, 2.4, 4.8, 9.6]
    last_err = None
    for d in delays:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            msg = str(e)

            transient = (
                "429" in msg
                or "Quota exceeded" in msg
                or "Read requests" in msg
                or "ConnectionResetError" in msg
                or "Connection aborted" in msg
                or "RemoteDisconnected" in msg
                or "SSLError" in msg
                or "TLS" in msg
                or isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout, socket.timeout))
            )

            if transient:
                time.sleep(d)
                continue

            raise
    raise last_err


# -----------------------------
# UUID normalization
# -----------------------------
def normalize_uuid(s: str) -> str:
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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def parse_dt(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df))
    return df[col].astype(str).fillna("").map(lambda x: str(x).strip())


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def get_spreadsheet(sa: dict, spreadsheet_id: str):
    gc = get_client_from_secrets(sa)
    sh = retry_api(gc.open_by_key, spreadsheet_id)
    return gc, sh


@st.cache_resource
def get_worksheets(sa: dict, spreadsheet_id: str):
    _, sh = get_spreadsheet(sa, spreadsheet_id)

    def _get_or_create(name: str, rows: int = 2000, cols: int = 20):
        try:
            return retry_api(sh.worksheet, name)
        except Exception:
            return retry_api(sh.add_worksheet, title=name, rows=rows, cols=cols)

    corr = _get_or_create("Correction_Log")
    raw = _get_or_create("Raw_Kobo_Data")
    clean = _get_or_create("Clean_Data")
    locks = _get_or_create(LOCK_SHEET_NAME, rows=3000, cols=10)
    users = _get_or_create(USERS_SHEET_NAME, rows=2000, cols=10)
    audit = _get_or_create(AUDIT_SHEET_NAME, rows=10000, cols=10)

    # ---- ensure Locks header (ROBUST: fixes Found: [])
    required_locks = ["_uuid", "locked_by", "locked_at", "expires_at"]
    header = retry_api(locks.row_values, 1)
    header = [str(h).strip() for h in header if str(h).strip()]

    if header != required_locks:
        # Always set header row; safe even if sheet has stray blanks
        retry_api(locks.update, "A1:D1", [required_locks])

        header2 = retry_api(locks.row_values, 1)
        header2 = [str(h).strip() for h in header2 if str(h).strip()]
        if header2 != required_locks:
            st.error(
                f"Worksheet '{LOCK_SHEET_NAME}' header could not be set automatically.\n"
                f"Expected: {required_locks}\nFound: {header2}"
            )
            st.stop()

    # ---- ensure Users header
    required_users = ["email", "password_hash", "is_active", "reset_code_hash", "reset_expires_at", "last_login_at"]
    uheader = retry_api(users.row_values, 1)
    uheader = [str(h).strip() for h in uheader if str(h).strip()]

    if uheader != required_users:
        uvals = retry_api(users.get_all_values)
        if not uvals or len(uvals) == 0:
            retry_api(users.update, "A1:F1", [required_users])
        else:
            if not (("email" in uheader) and ("password_hash" in uheader) and ("is_active" in uheader)):
                st.error(f"Users sheet header invalid. Expected at least email/password_hash/is_active. Found: {uheader}")
                st.stop()
            # If partially compatible, do not auto-overwrite to avoid damaging existing layout.

    # ---- ensure Audit_Log header
    required_audit = ["ts_utc", "action", "_uuid", "email", "items_count", "note"]
    aheader = retry_api(audit.row_values, 1)
    aheader = [str(h).strip() for h in aheader if str(h).strip()]
    if aheader != required_audit:
        avals = retry_api(audit.get_all_values)
        if not avals or len(avals) == 0:
            retry_api(audit.update, "A1:F1", [required_audit])

    return {
        "Correction_Log": corr,
        "Raw_Kobo_Data": raw,
        "Clean_Data": clean,
        "Locks": locks,
        "Users": users,
        "Audit_Log": audit,
    }


# -----------------------------
# Audit log
# -----------------------------
def audit_log(audit_ws, action: str, uuid_value: str, email: str, items_count: int = 0, note: str = ""):
    try:
        retry_api(
            audit_ws.append_rows,
            [[iso(utc_now()), action, normalize_uuid(uuid_value), email, str(items_count), note]],
            value_input_option="USER_ENTERED",
        )
    except Exception:
        pass


# -----------------------------
# Auth helpers (OTP + password set + forgot)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=120)
def load_users_df_cached(_users_ws) -> pd.DataFrame:
    vals = retry_api(_users_ws.get_all_values)
    if not vals or len(vals) < 2:
        return pd.DataFrame(
            columns=[
                "_row",
                "email",
                "password_hash",
                "is_active",
                "reset_code_hash",
                "reset_expires_at",
                "last_login_at",
            ]
        )

    headers = [str(h).strip() for h in vals[0]]
    rows = vals[1:]
    data = []
    for sheet_row_num, r in enumerate(rows, start=2):
        row = {"_row": sheet_row_num}
        for i, h in enumerate(headers):
            row[h] = r[i].strip() if i < len(r) else ""
        data.append(row)

    df = pd.DataFrame(data)
    if "email" in df.columns:
        df["email"] = df["email"].astype(str).str.strip().str.lower()
    return df


def _bcrypt_hash(s: str) -> str:
    return bcrypt.hashpw(s.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _bcrypt_check(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def send_otp_email(to_email: str, code: str) -> int:
    """
    Sends OTP via SendGrid and returns HTTP status code.
    Shows diagnostic status in UI to avoid "silent" failures.
    """
    api_key = st.secrets["auth"]["sendgrid_api_key"]
    from_email = st.secrets["auth"]["from_email"]

    subject = "PDM Translation - Verification Code"
    body = (
        f"Your verification code is: {code}\n\n"
        f"This code will expire soon. If you did not request this, ignore this email."
    )

    msg = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject=subject,
        plain_text_content=body,
    )

    sg = SendGridAPIClient(api_key)
    resp = sg.send(msg)

    # Diagnostic: 202 means accepted by SendGrid
    st.info(f"SendGrid response: {resp.status_code} (202 means accepted)")
    return int(resp.status_code)


def upsert_user_reset_code(users_ws, email: str, code: str, expires_at_iso: str) -> bool:
    dfu = load_users_df_cached(users_ws)
    hit = dfu[dfu["email"] == email]
    if hit.empty:
        return False

    row = hit.iloc[0]
    is_active = str(row.get("is_active", "TRUE")).strip().lower()
    if is_active in ("false", "0", "no", "inactive"):
        return False

    rownum = int(row["_row"])
    code_hash = _bcrypt_hash(code)

    # Users columns expected: A email, B password_hash, C is_active, D reset_code_hash, E reset_expires_at, F last_login_at
    retry_api(users_ws.update, f"D{rownum}", [[code_hash]])
    retry_api(users_ws.update, f"E{rownum}", [[expires_at_iso]])
    load_users_df_cached.clear()
    return True


def verify_otp(users_ws, email: str, code: str) -> bool:
    dfu = load_users_df_cached(users_ws)
    hit = dfu[dfu["email"] == email]
    if hit.empty:
        return False

    row = hit.iloc[0]
    is_active = str(row.get("is_active", "TRUE")).strip().lower()
    if is_active in ("false", "0", "no", "inactive"):
        return False

    code_hash = str(row.get("reset_code_hash", "")).strip()
    exp = str(row.get("reset_expires_at", "")).strip()
    if not code_hash or not exp:
        return False

    exp_dt = parse_dt(exp)
    if not exp_dt or exp_dt <= utc_now():
        return False

    return _bcrypt_check(code, code_hash)


def set_user_password(users_ws, email: str, new_password: str) -> bool:
    dfu = load_users_df_cached(users_ws)
    hit = dfu[dfu["email"] == email]
    if hit.empty:
        return False

    rownum = int(hit.iloc[0]["_row"])
    pw_hash = _bcrypt_hash(new_password)

    retry_api(users_ws.update, f"B{rownum}", [[pw_hash]])  # password_hash
    retry_api(users_ws.update, f"D{rownum}", [[""]])  # clear reset_code_hash
    retry_api(users_ws.update, f"E{rownum}", [[""]])  # clear reset_expires_at
    load_users_df_cached.clear()
    return True


def verify_login(users_ws, email: str, password: str) -> bool:
    dfu = load_users_df_cached(users_ws)
    hit = dfu[dfu["email"] == email]
    if hit.empty:
        return False

    row = hit.iloc[0]
    is_active = str(row.get("is_active", "TRUE")).strip().lower()
    if is_active in ("false", "0", "no", "inactive"):
        return False

    pw_hash = str(row.get("password_hash", "")).strip()
    if not pw_hash:
        return False

    return _bcrypt_check(password, pw_hash)


def update_last_login(users_ws, email: str):
    dfu = load_users_df_cached(users_ws)
    hit = dfu[dfu["email"] == email]
    if hit.empty:
        return
    rownum = int(hit.iloc[0]["_row"])
    retry_api(users_ws.update, f"F{rownum}", [[iso(utc_now())]])
    load_users_df_cached.clear()


def auth_gate(users_ws):
    """
    - Login with email+password
    - First time / Forgot password: send OTP to registered email, then set/reset your password.
    - Only emails already present in Users are allowed.
    """
    if st.session_state.get("auth_ok") and st.session_state.get("auth_email"):
        return

    st.markdown("## Sign in")
    tabs = st.tabs(["Login", "First time / Forgot password"])

    with tabs[0]:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")

        if submit:
            email = (email or "").strip().lower()
            if not email or not password:
                st.error("Email and password required.")
                st.stop()

            if not verify_login(users_ws, email, password):
                st.error("Invalid credentials, account inactive, or password not set.")
                st.stop()

            st.session_state["auth_ok"] = True
            st.session_state["auth_email"] = email
            update_last_login(users_ws, email)
            st.rerun()

    with tabs[1]:
        st.caption("Enter your registered email, receive a 6-digit code, then set/reset your password.")

        with st.form("request_code_form"):
            email2 = st.text_input("Email (registered)")
            req = st.form_submit_button("Send code")

        if req:
            email2 = (email2 or "").strip().lower()
            if not email2:
                st.error("Email required.")
                st.stop()

            dfu = load_users_df_cached(users_ws)
            if dfu[dfu["email"] == email2].empty:
                st.error("This email is not registered in Users sheet.")
                st.stop()

            ttl = int(st.secrets["auth"].get("otp_ttl_minutes", 15))
            code = f"{random.randint(0, 999999):06d}"
            exp = utc_now() + timedelta(minutes=ttl)

            if not upsert_user_reset_code(users_ws, email2, code, iso(exp)):
                st.error("Account inactive or could not create code.")
                st.stop()

            try:
                status = send_otp_email(email2, code)
                if status != 202:
                    st.warning(f"SendGrid did not accept the email (status={status}). Check SendGrid settings.")
                    st.stop()
            except Exception as e:
                st.error("Failed to send email. Check SendGrid API key, sender verification, and network.")
                st.exception(e)
                st.stop()

            st.success("Code accepted by SendGrid. Check your inbox/spam/promotions.")

        st.markdown("### Verify code and set password")
        with st.form("set_password_form"):
            email3 = st.text_input("Email")
            code3 = st.text_input("6-digit code")
            p1 = st.text_input("New password", type="password")
            p2 = st.text_input("Confirm password", type="password")
            setp = st.form_submit_button("Set password")

        if setp:
            email3 = (email3 or "").strip().lower()
            code3 = (code3 or "").strip()

            if not email3 or not code3 or not p1 or not p2:
                st.error("All fields required.")
                st.stop()
            if p1 != p2:
                st.error("Passwords do not match.")
                st.stop()
            if len(p1) < 8:
                st.error("Password must be at least 8 characters.")
                st.stop()

            if not verify_otp(users_ws, email3, code3):
                st.error("Invalid or expired code.")
                st.stop()

            if not set_user_password(users_ws, email3, p1):
                st.error("Failed to set password.")
                st.stop()

            st.session_state["auth_ok"] = True
            st.session_state["auth_email"] = email3
            update_last_login(users_ws, email3)
            st.success("Password set. Logged in.")
            st.rerun()

    st.stop()


def logout_button():
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("Logout"):
            for k in ["auth_ok", "auth_email", "locked_uuid", "uuid_select"]:
                st.session_state.pop(k, None)
            st.rerun()
    with c2:
        st.caption(f"Signed in as: {st.session_state.get('auth_email','')}")


def get_user_id() -> str:
    email = st.session_state.get("auth_email")
    if email:
        return str(email)
    if "session_user_id" not in st.session_state:
        st.session_state["session_user_id"] = f"session-{uuidlib.uuid4().hex[:12]}"
    return st.session_state["session_user_id"]


# -----------------------------
# Locks (shared via Google Sheet)
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


def cleanup_expired_locks(locks_ws):
    df = load_locks_df_cached(locks_ws)
    if df.empty:
        return

    now = utc_now()
    expired_rows = []
    for _, row in df.iterrows():
        exp = parse_dt(row.get("expires_at", ""))
        if exp and exp <= now:
            expired_rows.append(int(row["_row"]))

    if expired_rows:
        for r in sorted(expired_rows, reverse=True):
            try:
                retry_api(locks_ws.delete_rows, r)
            except Exception:
                pass
        load_locks_df_cached.clear()


def is_uuid_locked_by_other(locks_ws, uuid_value: str, user_id: str) -> bool:
    df = load_locks_df_cached(locks_ws)
    if df.empty:
        return False

    now = utc_now()
    target = normalize_uuid(uuid_value)

    df2 = df.copy()
    df2["exp_dt"] = df2["expires_at"].map(parse_dt)
    df2 = df2[df2["exp_dt"].notna() & (df2["exp_dt"] > now)]

    hit = df2[df2["_uuid"] == target]
    if hit.empty:
        return False

    return any(safe_col(hit, "locked_by") != user_id)


def acquire_lock(locks_ws, uuid_value: str, user_id: str) -> bool:
    cleanup_expired_locks(locks_ws)

    df = load_locks_df_cached(locks_ws)
    target = normalize_uuid(uuid_value)
    now = utc_now()
    exp = now + timedelta(minutes=LOCK_TTL_MINUTES)

    if df.empty:
        retry_api(
            locks_ws.append_rows,
            [[target, user_id, iso(now), iso(exp)]],
            value_input_option="USER_ENTERED",
        )
        load_locks_df_cached.clear()
        return True

    df2 = df.copy()
    df2["exp_dt"] = df2["expires_at"].map(parse_dt)
    df2_active = df2[df2["exp_dt"].notna() & (df2["exp_dt"] > now)]
    hit = df2_active[df2_active["_uuid"] == target]

    if not hit.empty:
        locked_by = str(hit.iloc[0].get("locked_by", "")).strip()
        rownum = int(hit.iloc[0]["_row"])
        if locked_by == user_id:
            retry_api(locks_ws.update, f"C{rownum}", [[iso(now)]])
            retry_api(locks_ws.update, f"D{rownum}", [[iso(exp)]])
            load_locks_df_cached.clear()
            return True
        return False

    retry_api(
        locks_ws.append_rows,
        [[target, user_id, iso(now), iso(exp)]],
        value_input_option="USER_ENTERED",
    )
    load_locks_df_cached.clear()
    return True


def release_lock(locks_ws, uuid_value: str, user_id: str):
    df = load_locks_df_cached(locks_ws)
    if df.empty:
        return
    target = normalize_uuid(uuid_value)
    hit = df[(safe_col(df, "_uuid") == target) & (safe_col(df, "locked_by") == user_id)]
    if hit.empty:
        return

    for r in sorted(hit["_row"].astype(int).tolist(), reverse=True):
        try:
            retry_api(locks_ws.delete_rows, r)
        except Exception:
            pass
    load_locks_df_cached.clear()


# -----------------------------
# Cached data reads
# -----------------------------
@st.cache_data(show_spinner=False, ttl=120)
def load_correction_log_df_cached(_corr_ws) -> pd.DataFrame:
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
    headers = retry_api(_ws.row_values, 1)
    headers = [str(h).strip() for h in headers]
    if uuid_col not in headers:
        return []
    idx = headers.index(uuid_col) + 1
    vals = retry_api(_ws.col_values, idx)
    return [normalize_uuid(v) for v in vals[1:]]


# -----------------------------
# Ensure Correction_Log has translated_by / translated_at columns
# -----------------------------
@st.cache_data(show_spinner=False, ttl=300)
def get_corr_headers_cached(_corr_ws) -> list[str]:
    hdr = retry_api(_corr_ws.row_values, 1)
    return [str(h).strip() for h in hdr]


def ensure_corr_translation_cols(corr_ws) -> tuple[int, int]:
    headers = get_corr_headers_cached(corr_ws)
    headers_clean = [str(h).strip() for h in headers]

    changed = False
    if "translated_by" not in headers_clean:
        headers_clean.append("translated_by")
        changed = True
    if "translated_at" not in headers_clean:
        headers_clean.append("translated_at")
        changed = True

    if changed:
        retry_api(corr_ws.update, "A1", [headers_clean])
        get_corr_headers_cached.clear()
        headers_clean = get_corr_headers_cached(corr_ws)

    by_idx = headers_clean.index("translated_by") + 1
    at_idx = headers_clean.index("translated_at") + 1
    return by_idx, at_idx


def batch_update_translated_meta(corr_ws, row_nums: list[int], email: str):
    if not row_nums:
        return

    by_idx, at_idx = ensure_corr_translation_cols(corr_ws)
    by_col = _col_letter(by_idx)
    at_col = _col_letter(at_idx)
    ts = iso(utc_now())

    data = []
    for r in row_nums:
        data.append({"range": f"{by_col}{r}", "values": [[email]]})
        data.append({"range": f"{at_col}{r}", "values": [[ts]]})

    retry_api(corr_ws.batch_update, data)


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
    repl = {}
    q = safe_col(df_uuid_all, "Question")
    nv = safe_col(df_uuid_all, "new_value")
    for qq, vv in zip(q.tolist(), nv.tolist()):
        if qq and vv:
            repl[qq] = vv
    return repl


def fetch_raw_rows_for_uuid(raw_ws, uuid_value: str, raw_headers: list[str], uuid_col: str = "_uuid") -> list[list[str]]:
    target = normalize_uuid(uuid_value)
    uuid_vals = get_uuid_column_cached(raw_ws, uuid_col)

    matches = [i + 2 for i, v in enumerate(uuid_vals) if v == target]
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
    row_map = {raw_headers[i]: (raw_row[i] if i < len(raw_row) else "") for i in range(len(raw_headers))}
    for col_name, new_val in repl.items():
        if col_name in row_map:
            row_map[col_name] = new_val
    return [row_map.get(h, "") for h in raw_headers]


# -----------------------------
# UI components (cards)
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
users_ws = ws_map["Users"]
audit_ws = ws_map["Audit_Log"]

# ✅ Auth gate
auth_gate(users_ws)
logout_button()

user_id = get_user_id()

# clean expired locks
cleanup_expired_locks(locks_ws)

# Load Correction_Log
df_corr = load_correction_log_df_cached(corr_ws)
if df_corr.empty:
    st.warning("Correction_Log is empty or missing headers.")
    st.stop()

required_cols = {"_row", "_uuid", "Question", "old_value", "new_value"}
missing = [c for c in required_cols if c not in df_corr.columns]
if missing:
    st.error(f"Correction_Log is missing required columns: {missing}")
    st.stop()

pending_uuid_list_all = build_pending_uuid_list_cached(df_corr)
if not pending_uuid_list_all:
    st.success("No pending translations found.")
    st.stop()

# Filter out UUIDs locked by other users
pending_uuid_list = [u for u in pending_uuid_list_all if not is_uuid_locked_by_other(locks_ws, u, user_id)]
if not pending_uuid_list:
    st.info("All pending UUIDs are currently being processed by other users (locked). Please refresh later.")
    st.stop()

# -----------------------------
# UUID selection (with locking)
# -----------------------------
st.markdown("### Select UUID")

my_locked_uuid = st.session_state.get("locked_uuid", "")
if my_locked_uuid and my_locked_uuid not in pending_uuid_list:
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
    audit_log(audit_ws, "RELEASE", prev_uuid, user_id, note="changed selection")
    st.session_state["locked_uuid"] = ""

if not uuid_value:
    st.info("Select a UUID to load pending items.")
    st.stop()

# acquire lock
ok = acquire_lock(locks_ws, uuid_value, user_id)
if not ok:
    st.error("This UUID is currently selected by another user. Choose another UUID.")
    st.session_state["uuid_select"] = ""
    st.rerun()

st.session_state["locked_uuid"] = uuid_value

# Pending rows for this UUID
df_corr_norm = df_corr.copy()
df_corr_norm["_uuid_norm"] = safe_col(df_corr_norm, "_uuid").map(normalize_uuid)

df_uuid_pending = df_corr_norm[
    (df_corr_norm["_uuid_norm"] == uuid_value) &
    (safe_col(df_corr_norm, "new_value") == "")
].copy()

if df_uuid_pending.empty:
    st.success("No pending items for this UUID.")
    release_lock(locks_ws, uuid_value, user_id)
    audit_log(audit_ws, "RELEASE", uuid_value, user_id, note="no pending items")
    st.session_state["locked_uuid"] = ""
    st.stop()

# Audit lock
audit_log(audit_ws, "LOCK", uuid_value, user_id, items_count=len(df_uuid_pending))

# Cards layout
c1, c2, c3, c4 = st.columns([1, 2.4, 1, 1])
with c1:
    mini_card("Pending items", f"{len(df_uuid_pending):,}", title_size="0.72rem", value_size="1.05rem", subtitle="")
with c2:
    mini_card(
        "UUID (locked)", uuid_value, f"Locked by: {user_id}",
        title_size="0.8rem", value_size="1.45rem", subtitle_size="0.76rem",
        value_wrap=False, card_min_height="90px"
    )
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

    # 1) Update Correction_Log new_value
    with st.spinner("Updating Correction_Log..."):
        try:
            retry_api(batch_update_new_values, corr_ws, updates=updates, new_value_col="new_value")
            translated_rows = [r for r, _v in updates]
            batch_update_translated_meta(corr_ws, translated_rows, user_id)
        except Exception as e:
            st.exception(e)
            st.stop()

    # Audit save
    audit_log(audit_ws, "SAVE", uuid_value, user_id, items_count=len(updates), note="updated new_value + translated_meta")

    # Clear relevant caches
    load_correction_log_df_cached.clear()
    build_pending_uuid_list_cached.clear()

    # 2) Build + append Clean_Data row(s)
    with st.spinner("Building Clean_Data row..."):
        try:
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
            # release lock after save attempt
            release_lock(locks_ws, uuid_value, user_id)
            audit_log(audit_ws, "RELEASE", uuid_value, user_id, note="released after save")
            st.session_state["locked_uuid"] = ""
            load_locks_df_cached.clear()

    st.session_state["uuid_select"] = ""
    st.rerun()

import re
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Detect Arabic-script characters (covers Dari/Farsi/Arabic)
ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")


# -----------------------------
# Core Google Sheets helpers
# -----------------------------

def get_client_from_secrets(service_account_info: dict):
    creds = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
    return gspread.authorize(creds)


def open_ws(gc, spreadsheet_id: str, worksheet_name: str):
    sh = gc.open_by_key(spreadsheet_id)
    return sh.worksheet(worksheet_name)


def get_headers(ws) -> list[str]:
    headers = ws.row_values(1)
    return [str(h).strip() for h in headers if str(h).strip() != ""]


def ensure_headers(ws, headers: list[str]):
    """
    If the worksheet is empty, write headers into row 1.
    """
    existing = ws.row_values(1)
    if not any(existing):
        ws.update("A1", [headers])


def _normalize_cell(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def get_existing_values_set(ws, col_name: str) -> set[str]:
    """
    Returns all non-empty values from the specified column, excluding the header row.
    """
    headers = ws.row_values(1)
    headers = [str(h).strip() for h in headers]

    if col_name not in headers:
        raise ValueError(f"Column '{col_name}' was not found in sheet '{ws.title}' headers (row 1).")

    col_idx = headers.index(col_name) + 1  # 1-based
    values = ws.col_values(col_idx)
    return set(str(v).strip() for v in values[1:] if str(v).strip())


def _col_letter(n: int) -> str:
    """
    1 -> A, 2 -> B, ... 27 -> AA
    """
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


# ---------------------------------------------
# Raw_Kobo_Data uploader (Rejection + Dedupe)
# ---------------------------------------------

def append_new_rows_with_rejection_and_dedupe(
    target_ws,
    rejection_ws,
    rows: list[dict],
    key_col_name: str = "_uuid",
    skip_if_key_missing: bool = True,
) -> dict:
    """
    Append-only uploader with two filters:
      1) Hard blocklist: if key exists in rejection_ws[key_col_name], skip it.
      2) Dedupe: if key exists in target_ws[key_col_name], skip it.

    Returns:
      total_input
      inserted
      skipped_duplicates
      skipped_rejected
      skipped_missing_key
    """
    if not rows:
        return {
            "total_input": 0,
            "inserted": 0,
            "skipped_duplicates": 0,
            "skipped_rejected": 0,
            "skipped_missing_key": 0,
        }

    # Ensure target has headers (if empty, build from incoming rows)
    headers = get_headers(target_ws)
    if not headers:
        keys = set().union(*[r.keys() for r in rows])
        keys = [k for k in keys if k is not None]

        if key_col_name in keys:
            headers = [key_col_name] + sorted([k for k in keys if k != key_col_name])
        else:
            headers = sorted([k for k in keys])

        ensure_headers(target_ws, headers)

    if key_col_name not in headers:
        raise ValueError(f"Key column '{key_col_name}' must exist in '{target_ws.title}' headers (row 1).")

    # Load existing keys from target and rejection worksheets
    existing_keys = get_existing_values_set(target_ws, key_col_name)
    rejected_keys = get_existing_values_set(rejection_ws, key_col_name)

    to_append = []
    skipped_duplicates = 0
    skipped_rejected = 0
    skipped_missing = 0

    for r in rows:
        key_val = _normalize_cell(r.get(key_col_name, ""))

        if not key_val:
            if skip_if_key_missing:
                skipped_missing += 1
                continue
            skipped_missing += 1
            continue

        if key_val in rejected_keys:
            skipped_rejected += 1
            continue

        if key_val in existing_keys:
            skipped_duplicates += 1
            continue

        existing_keys.add(key_val)
        to_append.append([r.get(h, "") for h in headers])

    if to_append:
        target_ws.append_rows(to_append, value_input_option="USER_ENTERED")

    return {
        "total_input": len(rows),
        "inserted": len(to_append),
        "skipped_duplicates": skipped_duplicates,
        "skipped_rejected": skipped_rejected,
        "skipped_missing_key": skipped_missing,
    }


# --------------------------------
# Non-English detection (shared)
# --------------------------------

def looks_non_english(value: str) -> bool:
    """
    Returns True if the value contains Arabic-script characters (Dari/Farsi/Arabic).
    """
    if not value:
        return False
    return bool(ARABIC_SCRIPT_RE.search(value))


# -----------------------------------------
# Correction_Log schema + scan & append
# -----------------------------------------

def ensure_correction_log_headers(corr_ws) -> list[str]:
    """
    Ensures Correction_Log contains at least these columns:
      _uuid, Question, old_value, new_value, Assigned_To

    If the sheet has no header row, it will be created.
    If required columns are missing, they will be appended to the header row.
    """
    required = ["_uuid", "Question", "old_value", "new_value", "Assigned_To"]

    headers = corr_ws.row_values(1)
    headers = [str(h).strip() for h in headers if str(h).strip()]

    if not headers:
        corr_ws.update("A1", [required])
        return required

    missing = [c for c in required if c not in headers]
    if missing:
        corr_ws.update("A1", [headers + missing])
        return headers + missing

    return headers


def build_existing_correction_keyset(corr_ws, headers: list[str]) -> set[str]:
    """
    Builds a de-duplication key set for Correction_Log entries.
    Key format: _uuid||Question||old_value
    """
    for c in ["_uuid", "Question", "old_value"]:
        if c not in headers:
            raise ValueError(f"Correction_Log must contain column '{c}'.")

    idx_uuid = headers.index("_uuid")
    idx_q = headers.index("Question")
    idx_old = headers.index("old_value")

    values = corr_ws.get_all_values()  # includes header row
    if len(values) <= 1:
        return set()

    existing = set()
    for row in values[1:]:
        u = _normalize_cell(row[idx_uuid]) if idx_uuid < len(row) else ""
        q = _normalize_cell(row[idx_q]) if idx_q < len(row) else ""
        ov = _normalize_cell(row[idx_old]) if idx_old < len(row) else ""
        if u and q and ov:
            existing.add(f"{u}||{q}||{ov}")

    return existing


def scan_raw_kobo_and_log_corrections(
    raw_ws,
    corr_ws,
    uuid_col: str = "_uuid",
    max_rows: int | None = None,
) -> dict:
    """
    Scans Raw_Kobo_Data for non-English values (Arabic-script characters).
    For each detected cell, appends a row to Correction_Log with:
      _uuid     = uuid from the same row
      Question  = column header name where the value was found
      old_value = the detected non-English value

    De-duplication:
      Does not re-add identical (_uuid, Question, old_value) already present in Correction_Log.

    Append behavior:
      Always appends after existing data using append_rows.
      If Correction_Log has extra columns, only (_uuid, Question, old_value) are populated
      and all other columns are left blank.
    """
    corr_headers = ensure_correction_log_headers(corr_ws)
    existing_keys = build_existing_correction_keyset(corr_ws, corr_headers)

    raw_values = raw_ws.get_all_values()
    if not raw_values or len(raw_values) < 2:
        return {"scanned_rows": 0, "found": 0, "inserted": 0, "skipped_existing": 0}

    raw_headers = [str(h).strip() for h in raw_values[0]]
    if uuid_col not in raw_headers:
        raise ValueError(f"Raw_Kobo_Data must contain column '{uuid_col}' in row 1.")

    uuid_idx = raw_headers.index(uuid_col)

    data_rows = raw_values[1:]
    if max_rows is not None:
        data_rows = data_rows[:max_rows]

    idx_uuid = corr_headers.index("_uuid")
    idx_q = corr_headers.index("Question")
    idx_old = corr_headers.index("old_value")
    corr_width = len(corr_headers)

    found = 0
    skipped_existing = 0
    to_append = []

    for row in data_rows:
        row_uuid = _normalize_cell(row[uuid_idx]) if uuid_idx < len(row) else ""
        if not row_uuid:
            continue

        for col_idx, header in enumerate(raw_headers):
            if not header or header == uuid_col:
                continue

            cell = _normalize_cell(row[col_idx]) if col_idx < len(row) else ""
            if not cell:
                continue

            if looks_non_english(cell):
                found += 1
                key = f"{row_uuid}||{header}||{cell}"
                if key in existing_keys:
                    skipped_existing += 1
                    continue

                existing_keys.add(key)

                new_row = [""] * corr_width
                new_row[idx_uuid] = row_uuid
                new_row[idx_q] = header
                new_row[idx_old] = cell
                to_append.append(new_row)

    if to_append:
        corr_ws.append_rows(to_append, value_input_option="USER_ENTERED")

    return {
        "scanned_rows": len(data_rows),
        "found": found,
        "inserted": len(to_append),
        "skipped_existing": skipped_existing,
    }


# -----------------------------------------
# Translation page helpers (UUID + updates)
# -----------------------------------------

def get_correction_log_table(corr_ws) -> tuple[list[str], list[list[str]]]:
    """
    Returns (headers, rows) where rows exclude the header row.
    Uses get_all_values to preserve row positions for updates.
    """
    values = corr_ws.get_all_values()
    if not values:
        return [], []
    headers = [str(h).strip() for h in values[0]]
    rows = values[1:]
    return headers, rows


def list_uuids_from_correction_log(corr_ws, uuid_col: str = "_uuid", limit: int = 5000) -> list[str]:
    """
    Returns a sorted list of unique uuids from Correction_Log.
    limit prevents loading extremely large lists into the UI.
    """
    headers, rows = get_correction_log_table(corr_ws)
    if not headers or uuid_col not in headers:
        return []

    idx = headers.index(uuid_col)
    uuids = []
    seen = set()

    for r in rows[:limit]:
        v = _normalize_cell(r[idx]) if idx < len(r) else ""
        if v and v not in seen:
            seen.add(v)
            uuids.append(v)

    uuids.sort()
    return uuids


def get_corrections_by_uuid_df(
    corr_ws,
    uuid_value: str,
    uuid_col: str = "_uuid",
) -> pd.DataFrame:
    """
    Returns a DataFrame with sheet row numbers for all Correction_Log entries matching uuid_value.
    Columns include: _row, _uuid, Question, old_value, new_value, Assigned_To (if present).
    """
    headers, rows = get_correction_log_table(corr_ws)
    if not headers:
        raise ValueError("Correction_Log is empty or missing headers.")
    if uuid_col not in headers:
        raise ValueError(f"Correction_Log must contain column '{uuid_col}'.")

    def col_idx(name: str):
        return headers.index(name) if name in headers else None

    idx_uuid = col_idx(uuid_col)
    idx_q = col_idx("Question")
    idx_old = col_idx("old_value")
    idx_new = col_idx("new_value")
    idx_assigned = col_idx("Assigned_To")

    data = []
    for sheet_row_num, r in enumerate(rows, start=2):  # header is row 1
        u = _normalize_cell(r[idx_uuid]) if idx_uuid is not None and idx_uuid < len(r) else ""
        if u != uuid_value:
            continue

        data.append(
            {
                "_row": sheet_row_num,
                "_uuid": u,
                "Question": _normalize_cell(r[idx_q]) if idx_q is not None and idx_q < len(r) else "",
                "old_value": _normalize_cell(r[idx_old]) if idx_old is not None and idx_old < len(r) else "",
                "new_value": _normalize_cell(r[idx_new]) if idx_new is not None and idx_new < len(r) else "",
                "Assigned_To": _normalize_cell(r[idx_assigned]) if idx_assigned is not None and idx_assigned < len(r) else "",
            }
        )

    return pd.DataFrame(data)


def batch_update_new_values(
    corr_ws,
    updates: list[tuple[int, str]],
    new_value_col: str = "new_value",
) -> int:
    """
    updates: list of (sheet_row_number, new_value)
    Writes values into the 'new_value' column for the given rows using a single batch update.
    Returns count of updated rows.
    """
    if not updates:
        return 0

    headers = corr_ws.row_values(1)
    headers = [str(h).strip() for h in headers]
    if new_value_col not in headers:
        raise ValueError(f"Correction_Log must contain column '{new_value_col}'.")

    col_idx = headers.index(new_value_col) + 1  # 1-based
    col_letter = _col_letter(col_idx)

    data = []
    for row_num, val in updates:
        a1 = f"{col_letter}{int(row_num)}"
        data.append({"range": a1, "values": [[val]]})

    corr_ws.batch_update(data)
    return len(updates)

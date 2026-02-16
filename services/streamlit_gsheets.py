import time
import streamlit as st

from services.sheets import get_client_from_secrets

# ---------------------------------------------------------
# Streamlit-cached Google Sheets helpers
# ---------------------------------------------------------
# Rationale:
# - gspread open_by_key / worksheet metadata calls are slow and can trigger 429 quotas.
# - st.cache_resource keeps connections + spreadsheet object alive per process.
# - retry wrapper provides basic resilience on transient quota/network errors.
# ---------------------------------------------------------

def retry_api(fn, *args, **kwargs):
    """Retries Google Sheets calls on 429/quota errors with exponential backoff."""
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


@st.cache_resource
def get_spreadsheet(sa: dict, spreadsheet_id: str):
    """Return (gc, sh) cached per Streamlit process."""
    gc = get_client_from_secrets(sa)
    sh = retry_api(gc.open_by_key, spreadsheet_id)
    return gc, sh


@st.cache_resource
def get_worksheet(sa: dict, spreadsheet_id: str, worksheet_name: str):
    """Return a cached worksheet handle."""
    _, sh = get_spreadsheet(sa, spreadsheet_id)
    return retry_api(sh.worksheet, worksheet_name)

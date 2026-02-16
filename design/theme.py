import streamlit as st
from pathlib import Path

def load_css() -> None:
    """Load global CSS from design/styles.css.

    All visual styling should be controlled from the /design folder.
    """
    css_path = Path(__file__).parent / "styles.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="pdm-card">
          <div class="pdm-title">{title}</div>
          <div class="pdm-sub">{subtitle}</div>
        </div>
        <div class="pdm-spacer"></div>
        """,
        unsafe_allow_html=True
    )

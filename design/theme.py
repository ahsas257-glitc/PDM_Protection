import streamlit as st
from pathlib import Path

def load_css():
    css_path = Path(__file__).parent / "styles.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="pdm-card">
          <div class="pdm-title">{title}</div>
          <div class="pdm-sub">{subtitle}</div>
        </div>
        <div style="height:14px;"></div>
        """,
        unsafe_allow_html=True
    )

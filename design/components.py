import streamlit as st

def kpi(label: str, value: str, help_text: str = ""):
    st.markdown(
        f"""
        <div class="pdm-card">
          <div style="font-size:0.85rem; opacity:0.75;">{label}</div>
          <div style="font-size:1.6rem; font-weight:800; margin-top:6px;">{value}</div>
          <div style="font-size:0.85rem; opacity:0.65; margin-top:6px;">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

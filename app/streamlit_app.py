import streamlit as st

st.set_page_config(
    page_title="Intent Classifier",
    page_icon=":material/psychology:",
    layout="wide",
)

predict_page = st.Page("pages/predict.py", title="Predict", icon=":material/chat:")
compare_page = st.Page(
    "pages/compare.py", title="Compare Models", icon=":material/bar_chart:"
)
monitoring_page = st.Page(
    "pages/monitoring.py", title="Monitoring", icon=":material/monitoring:"
)

pg = st.navigation([predict_page, compare_page, monitoring_page])
pg.run()

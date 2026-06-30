import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from app.api_client import api_predict, require_api

require_api()

st.title("Predict Intent")
st.caption("Test the intent classifier live, switching between models or letting the A/B router decide.")

col1, col2 = st.columns([3, 1])

with col1:
    text = st.text_input("Enter a query", placeholder="e.g. what is my account balance")

with col2:
    model_choice = st.selectbox(
        "Model",
        options=["A/B Router", "Classical (LogReg)", "SVM", "Transformer (DistilBERT)"],
    )

model_map = {
    "A/B Router": None,
    "Classical (LogReg)": "classical",
    "SVM": "svm",
    "Transformer (DistilBERT)": "transformer",
}

if st.button("Predict", type="primary", disabled=not text):
    with st.spinner("predicting..."):
        result = api_predict(text, model_map[model_choice])

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Intent", result["intent"])
    col_b.metric("Confidence", f"{result['confidence']:.2%}")
    col_c.metric("Latency", f"{result['latency_ms']:.1f} ms")

    if result["is_oos"]:
        st.warning("This query was flagged as out-of-scope (low confidence).")

    if result.get("ab_variant"):
        st.info(f"Served by A/B variant **{result['ab_variant']}** using model `{result['model_used']}`")
    else:
        st.info(f"Served by model `{result['model_used']}`")

    st.subheader("Top 5 Predictions")
    df = pd.DataFrame(result["top5"])
    df["confidence"] = df["confidence"].astype(float)
    st.bar_chart(df.set_index("intent")["confidence"], horizontal=True)

st.divider()
st.caption("Sample queries to try:")
samples = [
    "what is my account balance",
    "book a flight to new york",
    "set an alarm for 7am",
    "tell me about quantum physics",
    "write me a poem about the ocean",
]
cols = st.columns(len(samples))
for col, sample in zip(cols, samples):
    col.code(sample, language=None)
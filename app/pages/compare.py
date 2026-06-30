import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from app.api_client import api_predict, require_api

require_api()

st.title("Compare Models")
st.caption("Side-by-side comparison of all trained models on CLINC150 test set.")

RESULTS = {
    "logreg":      {"accuracy": 0.8058, "macro_f1": 0.8416, "weighted_f1": 0.7985, "latency_p50_ms": 0.12},
    "svm":         {"accuracy": 0.8002, "macro_f1": 0.8474, "weighted_f1": 0.7795, "latency_p50_ms": 0.14},
    "textcnn":     {"accuracy": 0.7740, "macro_f1": 0.8244, "weighted_f1": 0.7646, "latency_p50_ms": 97.6},
    "rnn":         {"accuracy": 0.1818, "macro_f1": 0.0020, "weighted_f1": 0.0560, "latency_p50_ms": 124.5},
    "lstm":        {"accuracy": 0.7669, "macro_f1": 0.8113, "weighted_f1": 0.7555, "latency_p50_ms": 249.2},
    "distilbert":  {"accuracy": 0.8765, "macro_f1": 0.9019, "weighted_f1": 0.8722, "latency_p50_ms": 9.6},
}

df = pd.DataFrame(RESULTS).T
df.index.name = "model"

st.subheader("Metrics Table")
st.dataframe(
    df.style.format({
        "accuracy": "{:.2%}",
        "macro_f1": "{:.2%}",
        "weighted_f1": "{:.2%}",
        "latency_p50_ms": "{:.2f} ms",
    }).background_gradient(subset=["accuracy", "macro_f1", "weighted_f1"], cmap="Greens"),
    use_container_width=True,
)

st.subheader("Accuracy & F1 Comparison")
chart_df = df[["accuracy", "macro_f1", "weighted_f1"]]
st.bar_chart(chart_df)

st.subheader("Latency Comparison")
st.bar_chart(df[["latency_p50_ms"]])

st.divider()
st.subheader("Side-by-Side Prediction")
st.caption("Run the same input through multiple models at once.")

text = st.text_input("Enter a query to compare", placeholder="e.g. cancel my hotel reservation")
selected_models = st.multiselect(
    "Models to compare",
    options=["classical", "svm", "transformer"],
    default=["classical", "transformer"],
)

if st.button("Compare", type="primary", disabled=not (text and selected_models)):
    cols = st.columns(len(selected_models))
    for col, model_type in zip(cols, selected_models):
        with col:
            with st.spinner(f"running {model_type}..."):
                result = api_predict(text, model_type)
            st.markdown(f"**{model_type}**")
            st.metric("Intent", result["intent"])
            st.metric("Confidence", f"{result['confidence']:.2%}")
            st.metric("Latency", f"{result['latency_ms']:.1f} ms")
            if result["is_oos"]:
                st.warning("flagged OOS")
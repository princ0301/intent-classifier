import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app.api_client import api_ab_stats, api_drift, api_reset_monitoring, require_api

require_api()

st.title("Monitoring Dashboard")
st.caption("Live A/B test stats and data drift detection.")

st.subheader("A/B Test Stats")

if st.button("Refresh stats"):
    st.rerun()

try:
    stats = api_ab_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Model A — `{stats['model_a']['model_name']}`")
        st.metric("Requests", stats["model_a"]["request_count"])
        st.metric("Avg Latency", f"{stats['model_a']['avg_latency_ms']:.2f} ms")
        st.metric("Avg Confidence", f"{stats['model_a']['avg_confidence']:.2%}")
        st.metric("OOS Rate", f"{stats['model_a']['oos_rate']:.2%}")

    with col2:
        st.markdown(f"### Model B — `{stats['model_b']['model_name']}`")
        st.metric("Requests", stats["model_b"]["request_count"])
        st.metric("Avg Latency", f"{stats['model_b']['avg_latency_ms']:.2f} ms")
        st.metric("Avg Confidence", f"{stats['model_b']['avg_confidence']:.2%}")
        st.metric("OOS Rate", f"{stats['model_b']['oos_rate']:.2%}")

    st.caption(f"Configured split: {stats['split']:.0%} to model B")

except Exception as e:
    st.warning(
        f"No A/B stats yet. Make some predictions on the Predict page first. ({e})"
    )

if st.button("Reset A/B stats"):
    api_reset_monitoring()
    st.success("Stats reset.")
    st.rerun()

st.divider()
st.subheader("Data Drift Report")
st.caption(
    "Compares reference traffic (initial baseline) against recent production traffic."
)

if st.button("Run drift check"):
    try:
        with st.spinner("computing drift..."):
            drift = api_drift()

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Drifted Columns",
            int(drift["drift_summary"].get("drifted_columns_count", 0)),
        )
        col2.metric(
            "Confidence Drop",
            f"{drift['confidence_drift']['confidence_drop']:.2%}",
            delta=f"{-drift['confidence_drift']['confidence_drop']:.2%}",
        )
        col3.metric(
            "OOS Rate Increase",
            f"{drift['oos_rate_drift']['oos_rate_increase']:.2%}",
        )

        if drift["confidence_drift"]["is_degraded"]:
            st.error("Confidence has degraded significantly vs reference traffic.")
        else:
            st.success("Confidence is stable vs reference traffic.")

        if drift["oos_rate_drift"]["is_anomalous"]:
            st.error(
                "OOS rate has increased anomalously — possible topic drift in incoming queries."
            )
        else:
            st.success("OOS rate is within normal range.")

        with st.expander("Raw drift details"):
            st.json(drift)

    except Exception as e:
        st.warning(
            "No reference/current data available yet. Run `verify_phase11.py` first to "
            f"generate baseline monitoring data. ({e})"
        )

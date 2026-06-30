import random

from src.monitoring.drift import (
    append_current_data,
    get_confidence_drift,
    get_drift_summary,
    get_oos_rate_drift,
    load_current_data,
    load_reference_data,
    run_drift_report,
    save_drift_report_html,
    save_reference_data,
)
from src.serving.predictor import Predictor


NORMAL_TEXTS = [
    "what is my account balance",
    "book a flight to new york",
    "set an alarm for 7am",
    "transfer 100 dollars to savings",
    "what is the weather like today",
    "cancel my hotel reservation",
    "what is my credit score",
    "play some music",
    "remind me to call mom",
    "what time is it",
]

DRIFTED_TEXTS = [
    "yo can u like book me a flight or whatever lol asap pls thx",
    "URGENT need balance info RIGHT NOW account frozen help",
    "vibes check what's my acc status fr fr no cap",
    "bruh my card declined again smh fix this",
    "lowkey need to know my balance rn no joke",
    "literally just tell me the weather already ugh",
    "ayo set alarm for tmrw 7 yk the usual",
    "ngl i forgot my pin can u reset it asap",
    "fr need to cancel this reservation its giving regret",
    "tbh just play whatever music u got rn",
]


def simulate_traffic(predictor: Predictor, texts: list[str]) -> tuple[list, list, list]:
    predictions = []
    confidences = []
    for text in texts:
        result = predictor.predict(text)
        predictions.append(result.intent)
        confidences.append(result.confidence)
    return texts, predictions, confidences


def main():
    print("loading predictor...")
    predictor = Predictor("classical")

    print("\nsimulating reference traffic (normal queries)...")
    texts, predictions, confidences = simulate_traffic(predictor, NORMAL_TEXTS * 10)
    save_reference_data(texts, predictions, confidences)
    print(f"  saved {len(texts)} reference samples")

    print("\nsimulating current traffic (normal, no drift)...")
    texts, predictions, confidences = simulate_traffic(predictor, NORMAL_TEXTS * 10)
    append_current_data(texts, predictions, confidences)

    reference_df = load_reference_data()
    current_df = load_current_data()

    print("\nrunning drift report (no drift expected)...")
    snapshot = run_drift_report(reference_df, current_df)
    summary = get_drift_summary(snapshot)
    print(f"  drift summary: {summary}")

    conf_drift = get_confidence_drift(reference_df, current_df)
    print(f"  confidence drift: {conf_drift}")

    oos_drift = get_oos_rate_drift(reference_df, current_df)
    print(f"  oos rate drift: {oos_drift}")

    print("\n\nsimulating DRIFTED traffic (slang, casual phrasing)...")
    texts, predictions, confidences = simulate_traffic(predictor, DRIFTED_TEXTS * 10)
    append_current_data(texts, predictions, confidences, window_size=100)

    current_df = load_current_data()

    print("running drift report (drift expected)...")
    snapshot = run_drift_report(reference_df, current_df)
    summary = get_drift_summary(snapshot)
    print(f"  drift summary: {summary}")

    conf_drift = get_confidence_drift(reference_df, current_df)
    print(f"  confidence drift: {conf_drift}")

    oos_drift = get_oos_rate_drift(reference_df, current_df)
    print(f"  oos rate drift: {oos_drift}")

    print("\nsaving html report...")
    save_drift_report_html(snapshot)
    print("  saved: artifacts/monitoring/drift_report.html")


if __name__ == "__main__":
    main()
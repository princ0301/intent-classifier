import pytest

from src.monitoring.drift import (
    append_current_data,
    get_confidence_drift,
    get_oos_rate_drift,
    load_current_data,
    load_reference_data,
    save_reference_data,
)


@pytest.fixture
def sample_data():
    texts = ["hello there", "what time is it", "book a flight"]
    predictions = ["greeting", "time", "book_flight"]
    confidences = [0.9, 0.85, 0.92]
    return texts, predictions, confidences


def test_save_reference_data(tmp_path, sample_data, monkeypatch):
    monkeypatch.chdir(tmp_path)
    texts, predictions, confidences = sample_data
    save_reference_data(texts, predictions, confidences)
    df = load_reference_data()
    assert len(df) == len(texts)


def test_reference_data_has_required_columns(tmp_path, sample_data, monkeypatch):
    monkeypatch.chdir(tmp_path)
    texts, predictions, confidences = sample_data
    save_reference_data(texts, predictions, confidences)
    df = load_reference_data()
    assert set(["text", "text_length", "prediction", "confidence"]).issubset(df.columns)


def test_load_reference_data_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_reference_data()


def test_append_current_data_creates_file(tmp_path, sample_data, monkeypatch):
    monkeypatch.chdir(tmp_path)
    texts, predictions, confidences = sample_data
    append_current_data(texts, predictions, confidences)
    df = load_current_data()
    assert len(df) == len(texts)


def test_append_current_data_accumulates(tmp_path, sample_data, monkeypatch):
    monkeypatch.chdir(tmp_path)
    texts, predictions, confidences = sample_data
    append_current_data(texts, predictions, confidences)
    append_current_data(texts, predictions, confidences)
    df = load_current_data()
    assert len(df) == len(texts) * 2


def test_append_current_data_respects_window_size(tmp_path, sample_data, monkeypatch):
    monkeypatch.chdir(tmp_path)
    texts, predictions, confidences = sample_data
    for _ in range(5):
        append_current_data(texts, predictions, confidences, window_size=5)
    df = load_current_data()
    assert len(df) == 5


def test_confidence_drift_detects_no_drop():
    import pandas as pd
    ref = pd.DataFrame({"confidence": [0.9, 0.9, 0.9]})
    cur = pd.DataFrame({"confidence": [0.89, 0.91, 0.9]})
    result = get_confidence_drift(ref, cur)
    assert result["is_degraded"] is False


def test_confidence_drift_detects_drop():
    import pandas as pd
    ref = pd.DataFrame({"confidence": [0.9, 0.9, 0.9]})
    cur = pd.DataFrame({"confidence": [0.5, 0.5, 0.5]})
    result = get_confidence_drift(ref, cur)
    assert result["is_degraded"] is True
    assert result["confidence_drop"] > 0.1


def test_oos_rate_drift_detects_increase():
    import pandas as pd
    ref = pd.DataFrame({"confidence": [0.9, 0.9, 0.9, 0.9]})
    cur = pd.DataFrame({"confidence": [0.3, 0.3, 0.3, 0.9]})
    result = get_oos_rate_drift(ref, cur, oos_threshold=0.5)
    assert result["is_anomalous"] is True


def test_oos_rate_drift_no_change():
    import pandas as pd
    ref = pd.DataFrame({"confidence": [0.9, 0.9, 0.3, 0.9]})
    cur = pd.DataFrame({"confidence": [0.9, 0.3, 0.9, 0.9]})
    result = get_oos_rate_drift(ref, cur, oos_threshold=0.5)
    assert result["is_anomalous"] is False
import pytest
from src.serving.ab_router import ABRouter, ABStats
from src.serving.predictor import PredictionResult, Predictor


@pytest.fixture(scope="module")
def classical_predictor():
    return Predictor("classical")


def test_predictor_invalid_model_type():
    with pytest.raises(ValueError):
        Predictor("not_a_real_model")


def test_predictor_returns_prediction_result(classical_predictor):
    result = classical_predictor.predict("what is my account balance")
    assert isinstance(result, PredictionResult)


def test_predictor_intent_is_known_label(classical_predictor):
    result = classical_predictor.predict("book a flight to paris")
    assert result.intent in classical_predictor.label_map


def test_predictor_confidence_in_range(classical_predictor):
    result = classical_predictor.predict("set an alarm for tomorrow")
    assert 0.0 <= result.confidence <= 1.0


def test_predictor_top5_has_five_items(classical_predictor):
    result = classical_predictor.predict("transfer money to savings")
    assert len(result.top5) == 5


def test_predictor_top5_confidences_descending(classical_predictor):
    result = classical_predictor.predict("what is the weather today")
    confidences = [item["confidence"] for item in result.top5]
    assert confidences == sorted(confidences, reverse=True)


def test_predictor_batch_returns_correct_count(classical_predictor):
    texts = ["hello", "what time is it", "cancel my reservation"]
    results = classical_predictor.predict_batch(texts)
    assert len(results) == len(texts)


def test_predictor_latency_is_positive(classical_predictor):
    result = classical_predictor.predict("play some music")
    assert result.latency_ms >= 0


def test_ab_stats_starts_empty():
    stats = ABStats(model_name="test")
    summary = stats.summary()
    assert summary["request_count"] == 0
    assert summary["avg_latency_ms"] == 0.0


def test_ab_stats_records_correctly():
    stats = ABStats(model_name="test")
    result = PredictionResult(intent="balance", confidence=0.9, latency_ms=5.0, is_oos=False)
    stats.record(result)
    summary = stats.summary()
    assert summary["request_count"] == 1
    assert summary["avg_latency_ms"] == 5.0
    assert summary["avg_confidence"] == 0.9
    assert summary["oos_rate"] == 0.0


def test_ab_stats_tracks_oos_rate():
    stats = ABStats(model_name="test")
    stats.record(PredictionResult(intent="oos", confidence=0.3, latency_ms=1.0, is_oos=True))
    stats.record(PredictionResult(intent="balance", confidence=0.9, latency_ms=1.0, is_oos=False))
    summary = stats.summary()
    assert summary["oos_rate"] == 0.5


def test_ab_router_split_zero_always_routes_a():
    router = ABRouter(model_a="classical", model_b="classical", split=0.0)
    for _ in range(10):
        _, variant = router.route("what is my balance")
        assert variant == "A"


def test_ab_router_split_one_always_routes_b():
    router = ABRouter(model_a="classical", model_b="classical", split=1.0)
    for _ in range(10):
        _, variant = router.route("what is my balance")
        assert variant == "B"


def test_ab_router_predict_returns_dict():
    router = ABRouter(model_a="classical", model_b="classical", split=0.5)
    result = router.predict("book a hotel")
    assert "intent" in result
    assert "ab_variant" in result
    assert result["ab_variant"] in ("A", "B")


def test_ab_router_reset_stats():
    router = ABRouter(model_a="classical", model_b="classical", split=0.5)
    router.predict("hello")
    router.predict("goodbye")
    router.reset_stats()
    stats = router.get_stats()
    assert stats["model_a"]["request_count"] == 0
    assert stats["model_b"]["request_count"] == 0
from src.serving.ab_router import ABRouter
from src.serving.predictor import Predictor


SAMPLE_TEXTS = [
    "what is my account balance",
    "book a flight to new york",
    "set an alarm for 7am",
    "transfer 100 dollars to savings",
    "what is the weather like today",
]


def test_single_predictor(model_type: str) -> None:
    print(f"\ntesting predictor: {model_type}")
    predictor = Predictor(model_type)

    for text in SAMPLE_TEXTS[:3]:
        result = predictor.predict(text)
        print(f"  text       : {text}")
        print(f"  intent     : {result.intent}")
        print(f"  confidence : {result.confidence}")
        print(f"  is_oos     : {result.is_oos}")
        print(f"  latency    : {result.latency_ms}ms")
        print()


def test_batch_prediction(model_type: str) -> None:
    print(f"\ntesting batch prediction: {model_type}")
    predictor = Predictor(model_type)
    results = predictor.predict_batch(SAMPLE_TEXTS)
    for text, result in zip(SAMPLE_TEXTS, results):
        print(f"  {text:<40} -> {result.intent} ({result.confidence})")


def test_ab_router() -> None:
    print("\ntesting AB router (classical vs transformer, split=0.3)")
    router = ABRouter(model_a="classical", model_b="transformer", split=0.3)

    for _ in range(20):
        for text in SAMPLE_TEXTS:
            router.predict(text)

    stats = router.get_stats()
    print(f"\n  split configured : {stats['split']}")
    print(f"  model A ({stats['model_a']['model_name']}):")
    print(f"    requests        : {stats['model_a']['request_count']}")
    print(f"    avg_latency_ms  : {stats['model_a']['avg_latency_ms']}")
    print(f"    avg_confidence  : {stats['model_a']['avg_confidence']}")
    print(f"  model B ({stats['model_b']['model_name']}):")
    print(f"    requests        : {stats['model_b']['request_count']}")
    print(f"    avg_latency_ms  : {stats['model_b']['avg_latency_ms']}")
    print(f"    avg_confidence  : {stats['model_b']['avg_confidence']}")

    total = stats["model_a"]["request_count"] + stats["model_b"]["request_count"]
    actual_split = stats["model_b"]["request_count"] / total
    print(f"\n  expected split ~0.3, actual split: {actual_split:.3f}")


def main():
    test_single_predictor("classical")
    test_batch_prediction("transformer")
    test_ab_router()


if __name__ == "__main__":
    main()
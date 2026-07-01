import pytest
import pandas as pd
import torch


@pytest.fixture(scope="session")
def sample_texts():
    return [
        "what is my account balance",
        "book a flight to new york",
        "set an alarm for 7am",
        "transfer money to savings",
        "what is the weather today",
    ]


@pytest.fixture(scope="session")
def sample_labels():
    return [0, 1, 2, 3, 4]


@pytest.fixture(scope="session")
def sample_df(sample_texts, sample_labels):
    return pd.DataFrame({"text": sample_texts, "label_id": sample_labels})


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def num_classes():
    return 151
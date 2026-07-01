import pytest
import torch
import numpy as np

from src.models.neural import TextCNN, RNNModel, LSTMModel, Vocabulary, IntentDatasetNN
from src.models.classical import LogisticRegressionModel, SVMModel
from src.models.transformer import TransformerModel, IntentDatasetHF


VOCAB_SIZE = 100
EMBEDDING_DIM = 32
NUM_FILTERS = 64
KERNEL_SIZES = [2, 3, 4]
HIDDEN_DIM = 64
NUM_LAYERS = 1
DROPOUT = 0.0
BATCH_SIZE = 4
SEQ_LEN = 16


@pytest.fixture
def textcnn(num_classes):
    return TextCNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        num_classes=num_classes,
        dropout=DROPOUT,
    )


@pytest.fixture
def rnn_model(num_classes):
    return RNNModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    )


@pytest.fixture
def lstm_model(num_classes):
    return LSTMModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    )


@pytest.fixture
def dummy_input():
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))


def test_textcnn_output_shape(textcnn, dummy_input, num_classes):
    out = textcnn(dummy_input)
    assert out.shape == (BATCH_SIZE, num_classes)


def test_rnn_output_shape(rnn_model, dummy_input, num_classes):
    out = rnn_model(dummy_input)
    assert out.shape == (BATCH_SIZE, num_classes)


def test_lstm_output_shape(lstm_model, dummy_input, num_classes):
    out = lstm_model(dummy_input)
    assert out.shape == (BATCH_SIZE, num_classes)


def test_textcnn_predict_proba_sums_to_one(textcnn, dummy_input):
    probs = textcnn.predict_proba(dummy_input)
    assert probs.shape[0] == BATCH_SIZE
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(BATCH_SIZE), atol=1e-5)


def test_rnn_predict_proba_sums_to_one(rnn_model, dummy_input):
    probs = rnn_model.predict_proba(dummy_input)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(BATCH_SIZE), atol=1e-5)


def test_lstm_predict_proba_sums_to_one(lstm_model, dummy_input):
    probs = lstm_model.predict_proba(dummy_input)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(BATCH_SIZE), atol=1e-5)


def test_textcnn_save_and_load(textcnn, dummy_input, tmp_path):
    save_path = str(tmp_path / "textcnn.pt")
    out_before = textcnn(dummy_input).detach().numpy()
    textcnn.save(save_path)

    loaded = TextCNN(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        num_classes=151,
        dropout=DROPOUT,
    )
    loaded.load(save_path)
    out_after = loaded(dummy_input).detach().numpy()
    np.testing.assert_allclose(out_before, out_after, atol=1e-5)


def test_lstm_save_and_load(lstm_model, dummy_input, tmp_path):
    save_path = str(tmp_path / "lstm.pt")
    out_before = lstm_model(dummy_input).detach().numpy()
    lstm_model.save(save_path)

    loaded = LSTMModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=151,
        dropout=DROPOUT,
    )
    loaded.load(save_path)
    out_after = loaded(dummy_input).detach().numpy()
    np.testing.assert_allclose(out_before, out_after, atol=1e-5)


def test_vocabulary_build_and_encode():
    texts = ["hello world", "book a flight", "what is my balance"]
    vocab = Vocabulary()
    vocab.build(texts)
    assert len(vocab) > 2
    encoded = vocab.encode("hello world", max_length=5)
    assert len(encoded) == 5


def test_vocabulary_pads_to_max_length():
    vocab = Vocabulary()
    vocab.build(["hi"])
    encoded = vocab.encode("hi", max_length=10)
    assert len(encoded) == 10
    assert encoded[-1] == 0


def test_vocabulary_handles_unk():
    vocab = Vocabulary()
    vocab.build(["hello world"])
    encoded = vocab.encode("unknown_word_xyz", max_length=4)
    assert encoded[0] == 1


def test_intent_dataset_nn_length():
    vocab = Vocabulary()
    texts = ["hello", "book a flight", "set alarm"]
    vocab.build(texts)
    dataset = IntentDatasetNN(texts, [0, 1, 2], vocab, max_length=8)
    assert len(dataset) == 3


def test_intent_dataset_nn_item_shape():
    vocab = Vocabulary()
    texts = ["hello world"]
    vocab.build(texts)
    dataset = IntentDatasetNN(texts, [0], vocab, max_length=8)
    x, y = dataset[0]
    assert x.shape == (8,)
    assert y.item() == 0


def test_logreg_predict_output_shape(sample_texts, num_classes):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    vec = TfidfVectorizer(max_features=100)
    X = vec.fit_transform(sample_texts)
    y = np.array([0, 1, 2, 3, 4])

    model = LogisticRegressionModel(C=1.0, max_iter=200)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(sample_texts),)


def test_svm_predict_output_shape(sample_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    vec = TfidfVectorizer(max_features=100)
    X = vec.fit_transform(sample_texts)
    y = np.array([0, 1, 2, 3, 4])

    model = SVMModel(C=1.0, max_iter=200)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(sample_texts),)
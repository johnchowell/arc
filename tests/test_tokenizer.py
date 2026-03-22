import pytest
from semantic_searcher.services.tokenizer import BPETokenizerService


def test_tokenizer_not_trained():
    tok = BPETokenizerService()
    assert not tok.is_trained()
    assert tok.vocab_size is None


def test_tokenizer_train_and_encode():
    tok = BPETokenizerService()
    corpus = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning and artificial intelligence are transforming technology",
        "Semantic search uses vector embeddings to find similar content",
    ] * 100  # Repeat to give trainer enough data
    tok.train(corpus, vocab_size=500)
    assert tok.is_trained()
    assert tok.vocab_size <= 500
    assert tok.vocab_size > 0

    ids = tok.encode("the quick fox")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert len(ids) > 0


def test_tokenizer_roundtrip():
    tok = BPETokenizerService()
    corpus = ["hello world this is a test"] * 200
    tok.train(corpus, vocab_size=100)

    text = "hello world"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert text in decoded or decoded.strip() == text


def test_tokenizer_encode_raises_if_not_trained():
    tok = BPETokenizerService()
    with pytest.raises(RuntimeError):
        tok.encode("test")


def test_tokenizer_decode_raises_if_not_trained():
    tok = BPETokenizerService()
    with pytest.raises(RuntimeError):
        tok.decode([1, 2, 3])

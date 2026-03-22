from semantic_searcher.utils.content_extractor import extract_content, chunk_text
from semantic_searcher.utils.url_utils import normalize_url, url_hash


def test_normalize_url_strips_fragment():
    assert normalize_url("https://example.com/page#section") == "https://example.com/page"


def test_normalize_url_strips_tracking_params():
    url = "https://example.com/page?q=test&utm_source=google&utm_medium=cpc"
    assert normalize_url(url) == "https://example.com/page?q=test"


def test_normalize_url_sorts_params():
    url = "https://example.com/page?z=1&a=2"
    assert normalize_url(url) == "https://example.com/page?a=2&z=1"


def test_normalize_url_lowercase_host():
    assert normalize_url("https://EXAMPLE.COM/Page") == "https://example.com/Page"


def test_url_hash_deterministic():
    h1 = url_hash("https://example.com/page")
    h2 = url_hash("https://example.com/page")
    assert h1 == h2
    assert len(h1) == 64


def test_url_hash_different_urls():
    assert url_hash("https://a.com") != url_hash("https://b.com")


def test_extract_content(sample_html):
    content = extract_content(sample_html, "https://example.com")
    assert "Test" in content.title
    assert "machine learning" in content.text
    assert content.meta_description == "A test page for unit testing"


def test_extract_content_images(sample_html):
    content = extract_content(sample_html, "https://example.com")
    assert any("image.jpg" in u for u in content.image_urls)


def test_chunk_text_splits():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, max_words=55)
    assert len(chunks) == 5
    for chunk in chunks:
        assert len(chunk.split()) <= 55


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short():
    chunks = chunk_text("this is a short sentence with enough words to pass")
    assert len(chunks) == 1

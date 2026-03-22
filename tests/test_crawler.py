from semantic_searcher.utils.url_utils import normalize_url, url_hash, is_same_domain


def test_normalize_strips_trailing_slash():
    assert normalize_url("https://example.com/page/") == "https://example.com/page"


def test_normalize_preserves_root():
    assert normalize_url("https://example.com") == "https://example.com/"
    assert normalize_url("https://example.com/") == "https://example.com/"


def test_normalize_removes_default_port():
    assert normalize_url("https://example.com:443/page") == "https://example.com/page"
    assert normalize_url("http://example.com:80/page") == "http://example.com/page"


def test_normalize_keeps_non_default_port():
    assert "8080" in normalize_url("https://example.com:8080/page")


def test_is_same_domain():
    assert is_same_domain("https://example.com/a", "https://example.com/b")
    assert not is_same_domain("https://a.com/x", "https://b.com/x")


def test_url_hash_normalized():
    # Same URL with different tracking params should hash the same
    h1 = url_hash("https://example.com/page?q=test&utm_source=x")
    h2 = url_hash("https://example.com/page?q=test&utm_medium=y")
    assert h1 == h2

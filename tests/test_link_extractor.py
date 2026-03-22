from semantic_searcher.utils.link_extractor import (
    extract_all_links,
    extract_html_links,
    extract_js_urls,
)


class TestExtractHtmlLinks:
    BASE = "https://example.com/page"

    def test_extracts_a_href(self):
        html = '<html><body><a href="https://other.com/foo">link</a></body></html>'
        result = extract_html_links(html, self.BASE)
        assert "https://other.com/foo" in result

    def test_extracts_link_href(self):
        html = '<html><head><link rel="canonical" href="https://example.com/canonical"></head></html>'
        result = extract_html_links(html, self.BASE)
        assert "https://example.com/canonical" in result

    def test_extracts_iframe_src(self):
        html = '<html><body><iframe src="https://embed.com/video"></iframe></body></html>'
        result = extract_html_links(html, self.BASE)
        assert "https://embed.com/video" in result

    def test_extracts_form_action(self):
        html = '<html><body><form action="https://example.com/submit"></form></body></html>'
        result = extract_html_links(html, self.BASE)
        assert "https://example.com/submit" in result

    def test_extracts_area_href(self):
        html = '<html><body><area href="https://example.com/region"></body></html>'
        result = extract_html_links(html, self.BASE)
        assert "https://example.com/region" in result

    def test_resolves_relative_urls(self):
        html = '<html><body><a href="/about">About</a><a href="sub/page">Sub</a></body></html>'
        result = extract_html_links(html, self.BASE)
        assert "https://example.com/about" in result
        assert "https://example.com/sub/page" in result

    def test_skips_javascript_scheme(self):
        html = '<html><body><a href="javascript:void(0)">click</a></body></html>'
        result = extract_html_links(html, self.BASE)
        assert len(result) == 0

    def test_skips_mailto(self):
        html = '<html><body><a href="mailto:test@example.com">email</a></body></html>'
        result = extract_html_links(html, self.BASE)
        assert len(result) == 0

    def test_skips_tel(self):
        html = '<html><body><a href="tel:+1234567890">call</a></body></html>'
        result = extract_html_links(html, self.BASE)
        assert len(result) == 0

    def test_skips_data_scheme(self):
        html = '<html><body><a href="data:text/html,hello">data</a></body></html>'
        result = extract_html_links(html, self.BASE)
        assert len(result) == 0

    def test_skips_fragment_only(self):
        html = '<html><body><a href="#section">jump</a></body></html>'
        result = extract_html_links(html, self.BASE)
        assert len(result) == 0

    def test_strips_fragment_from_url(self):
        html = '<html><body><a href="https://example.com/page#section">link</a></body></html>'
        result = extract_html_links(html, self.BASE)
        assert "https://example.com/page" in result
        assert not any("#" in u for u in result)

    def test_deduplicates(self):
        html = """<html><body>
            <a href="https://example.com/dup">one</a>
            <a href="https://example.com/dup">two</a>
        </body></html>"""
        result = extract_html_links(html, self.BASE)
        assert len(result) == 1

    def test_multiple_tags(self):
        html = """<html>
        <head><link rel="next" href="https://example.com/page2"></head>
        <body>
            <a href="https://other.com">other</a>
            <iframe src="https://embed.com/v"></iframe>
        </body></html>"""
        result = extract_html_links(html, self.BASE)
        assert len(result) == 3


class TestExtractJsUrls:
    def test_extracts_double_quoted_url(self):
        html = '<html><body><script>var u = "https://api.example.com/data";</script></body></html>'
        result = extract_js_urls(html)
        assert "https://api.example.com/data" in result

    def test_extracts_single_quoted_url(self):
        html = "<html><body><script>var u = 'https://api.example.com/data';</script></body></html>"
        result = extract_js_urls(html)
        assert "https://api.example.com/data" in result

    def test_extracts_backtick_quoted_url(self):
        html = "<html><body><script>var u = `https://api.example.com/data`;</script></body></html>"
        result = extract_js_urls(html)
        assert "https://api.example.com/data" in result

    def test_filters_google_analytics(self):
        html = '<html><body><script>var u = "https://www.google-analytics.com/analytics.js";</script></body></html>'
        result = extract_js_urls(html)
        assert len(result) == 0

    def test_filters_googleapis(self):
        html = '<html><body><script>var u = "https://fonts.googleapis.com/css?family=Roboto";</script></body></html>'
        result = extract_js_urls(html)
        assert len(result) == 0

    def test_filters_w3_org(self):
        html = '<html><body><script>var ns = "https://www.w3.org/2000/svg";</script></body></html>'
        result = extract_js_urls(html)
        assert len(result) == 0

    def test_filters_schema_org(self):
        html = '<html><body><script>var ctx = "https://schema.org/Article";</script></body></html>'
        result = extract_js_urls(html)
        assert len(result) == 0

    def test_filters_static_png(self):
        html = '<html><body><script>var img = "https://example.com/logo.png";</script></body></html>'
        result = extract_js_urls(html)
        assert len(result) == 0

    def test_filters_static_css(self):
        html = '<html><body><script>var s = "https://example.com/style.css";</script></body></html>'
        result = extract_js_urls(html)
        assert len(result) == 0

    def test_filters_font_woff2(self):
        html = '<html><body><script>var f = "https://example.com/font.woff2";</script></body></html>'
        result = extract_js_urls(html)
        assert len(result) == 0

    def test_filters_minified_bundle(self):
        html = '<html><body><script>var s = "https://example.com/app.min.js";</script></body></html>'
        result = extract_js_urls(html)
        assert len(result) == 0

    def test_filters_bundle_file(self):
        html = '<html><body><script>var s = "https://example.com/vendor.bundle.js";</script></body></html>'
        result = extract_js_urls(html)
        assert len(result) == 0

    def test_ignores_external_script_src(self):
        """External <script src="..."> tags have no inline text to extract from."""
        html = '<html><body><script src="https://cdn.example.com/lib.js"></script></body></html>'
        result = extract_js_urls(html)
        assert len(result) == 0

    def test_handles_multiple_urls(self):
        html = """<html><body><script>
            var a = "https://api.example.com/v1";
            var b = 'https://app.example.com/dashboard';
        </script></body></html>"""
        result = extract_js_urls(html)
        assert "https://api.example.com/v1" in result
        assert "https://app.example.com/dashboard" in result
        assert len(result) == 2

    def test_extracts_http_url(self):
        html = '<html><body><script>var u = "http://legacy.example.com/api";</script></body></html>'
        result = extract_js_urls(html)
        assert "http://legacy.example.com/api" in result


class TestExtractAllLinks:
    BASE = "https://example.com/page"

    def test_merges_html_and_js(self):
        html = """<html><body>
            <a href="https://link.example.com">link</a>
            <script>var u = "https://api.example.com/data";</script>
        </body></html>"""
        result = extract_all_links(html, self.BASE)
        assert "https://link.example.com" in result
        assert "https://api.example.com/data" in result

    def test_deduplicates_across_html_and_js(self):
        html = """<html><body>
            <a href="https://example.com/shared">link</a>
            <script>var u = "https://example.com/shared";</script>
        </body></html>"""
        result = extract_all_links(html, self.BASE)
        shared = [u for u in result if "shared" in u]
        assert len(shared) == 1

    def test_returns_set(self):
        html = "<html><body></body></html>"
        result = extract_all_links(html, self.BASE)
        assert isinstance(result, set)

    def test_empty_html(self):
        result = extract_all_links("", self.BASE)
        assert len(result) == 0

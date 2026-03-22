from semantic_searcher.services.ct_watcher import CTWatcherService


class TestFilterDomain:
    def test_strips_wildcard(self):
        assert CTWatcherService._filter_domain("*.example.com") == "example.com"

    def test_normal_domain_passes(self):
        assert CTWatcherService._filter_domain("www.example.com") == "www.example.com"

    def test_rejects_ip(self):
        assert CTWatcherService._filter_domain("192.168.1.1") is None

    def test_rejects_ip_like(self):
        assert CTWatcherService._filter_domain("10.0.0.1") is None

    def test_rejects_private_tld_local(self):
        assert CTWatcherService._filter_domain("host.local") is None

    def test_rejects_private_tld_internal(self):
        assert CTWatcherService._filter_domain("app.internal") is None

    def test_rejects_private_tld_test(self):
        assert CTWatcherService._filter_domain("thing.test") is None

    def test_rejects_private_tld_localhost(self):
        assert CTWatcherService._filter_domain("something.localhost") is None

    def test_rejects_no_dot(self):
        assert CTWatcherService._filter_domain("localhost") is None

    def test_lowercases(self):
        assert CTWatcherService._filter_domain("WWW.EXAMPLE.COM") == "www.example.com"

    def test_strips_trailing_dot(self):
        assert CTWatcherService._filter_domain("example.com.") == "example.com"

    def test_empty_returns_none(self):
        assert CTWatcherService._filter_domain("") is None


class TestProcessCert:
    def _make_msg(self, domains: list[str]) -> dict:
        return {
            "message_type": "certificate_update",
            "data": {
                "leaf_cert": {
                    "all_domains": domains,
                }
            },
        }

    def test_extracts_domains(self):
        watcher = CTWatcherService()
        msg = self._make_msg(["example.com", "*.example.com", "sub.example.com"])
        result = watcher._process_cert(msg)
        assert "example.com" in result
        assert "sub.example.com" in result
        # Wildcard stripped — both *.example.com and example.com yield "example.com"
        assert all(d == "example.com" or d == "sub.example.com" for d in result)

    def test_filters_bad_domains(self):
        watcher = CTWatcherService()
        msg = self._make_msg(["192.168.1.1", "host.local", "good.com"])
        result = watcher._process_cert(msg)
        assert result == ["good.com"]

    def test_ignores_non_certificate_update(self):
        watcher = CTWatcherService()
        msg = {"message_type": "heartbeat", "data": {}}
        assert watcher._process_cert(msg) == []

    def test_handles_missing_fields(self):
        watcher = CTWatcherService()
        msg = {"message_type": "certificate_update", "data": {}}
        assert watcher._process_cert(msg) == []

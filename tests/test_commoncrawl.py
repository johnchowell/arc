from pathlib import Path

from semantic_searcher.utils.commoncrawl import CommonCrawlSeeder


class TestReverseSurt:
    def test_simple_two_part(self):
        assert CommonCrawlSeeder._reverse_surt("com.google") == "google.com"

    def test_three_part(self):
        assert CommonCrawlSeeder._reverse_surt("org.wikipedia.en") == "en.wikipedia.org"

    def test_four_part(self):
        assert CommonCrawlSeeder._reverse_surt("uk.co.bbc.www") == "www.bbc.co.uk"

    def test_single_segment_returns_none(self):
        assert CommonCrawlSeeder._reverse_surt("com") is None

    def test_empty_returns_none(self):
        assert CommonCrawlSeeder._reverse_surt("") is None


class TestDomainsToUrls:
    def test_basic(self):
        urls = CommonCrawlSeeder.domains_to_urls(["google.com", "example.org"])
        assert urls == ["https://google.com/", "https://example.org/"]

    def test_empty(self):
        assert CommonCrawlSeeder.domains_to_urls([]) == []


class TestSample:
    def test_sample_respects_n(self, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Write fake domain cache
        domains = [f"domain{i}.com" for i in range(100)]
        (cache_dir / "cc_domains.txt").write_text("\n".join(domains) + "\n")

        seeder = CommonCrawlSeeder(cache_dir=cache_dir)
        sample = seeder.sample(n=10)
        assert len(sample) == 10
        assert all(d in domains for d in sample)

    def test_sample_clamps_to_available(self, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        domains = ["a.com", "b.com", "c.com"]
        (cache_dir / "cc_domains.txt").write_text("\n".join(domains) + "\n")

        seeder = CommonCrawlSeeder(cache_dir=cache_dir)
        sample = seeder.sample(n=1000)
        assert len(sample) == 3


class TestLoadFromCache:
    def test_reads_existing_cache(self, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "cc_domains.txt").write_text("google.com\nexample.org\n")

        seeder = CommonCrawlSeeder(cache_dir=cache_dir)
        loaded = seeder.load()
        assert loaded == ["google.com", "example.org"]

    def test_skips_blank_lines(self, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "cc_domains.txt").write_text("google.com\n\nexample.org\n\n")

        seeder = CommonCrawlSeeder(cache_dir=cache_dir)
        loaded = seeder.load()
        assert loaded == ["google.com", "example.org"]

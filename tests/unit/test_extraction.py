"""HTML extraction and source classification."""

from __future__ import annotations

import pytest

from research.models.common import SourceType
from research.normalize.html import extract_page, parse_date
from research.sources.classify import classify_url

ARTICLE = """<!doctype html><html lang="en"><head>
<title>NuScale cost estimate rises | Example News</title>
<link rel="canonical" href="https://apnews.example/article/costs"/>
<meta property="og:site_name" content="Example News"/>
<meta property="article:published_time" content="2025-03-04T10:00:00Z"/>
<meta name="author" content="Jane Roe, Sam Poe"/>
<meta name="description" content="Costs rose by 53 percent."/>
<script type="application/ld+json">
{"@type":"NewsArticle","headline":"NuScale cost estimate rises",
 "datePublished":"2025-03-04","author":[{"name":"Jane Roe"}],
 "publisher":{"name":"Example News"}}
</script>
<script>var tracker = "should never appear"; document.cookie = "x";</script>
<style>.ad { display: none }</style>
</head><body>
<nav>Home | World | Business | Subscribe now</nav>
<header>Breaking news banner</header>
<article>
  <h1>NuScale cost estimate rises</h1>
  <p>The target price rose to $89 per megawatt hour.</p>
  <p>See the <a href="https://www.nrc.gov/docs/filing.html">regulatory filing</a> for details.</p>
</article>
<aside>Related: five things to know</aside>
<footer>&copy; 2025 Example News. All rights reserved.</footer>
</body></html>"""


class TestExtraction:
    @pytest.fixture
    def page(self):
        return extract_page(ARTICLE)

    def test_body_text_excludes_chrome_and_scripts(self, page) -> None:
        assert "89 per megawatt hour" in page.text
        assert "should never appear" not in page.text
        assert "Subscribe now" not in page.text
        assert "five things to know" not in page.text
        assert "All rights reserved" not in page.text

    def test_outlet_suffix_is_stripped_from_the_headline(self, page) -> None:
        assert page.title == "NuScale cost estimate rises"

    def test_publisher_declared_canonical_is_captured(self, page) -> None:
        assert page.canonical_url == "https://apnews.example/article/costs"

    def test_metadata_is_read_from_meta_tags_and_json_ld(self, page) -> None:
        assert page.site_name == "Example News"
        assert page.language == "en"
        assert page.published_at is not None
        assert page.published_at.date().isoformat() == "2025-03-04"
        assert page.authors[:2] == ["Jane Roe", "Sam Poe"]

    def test_outbound_links_are_kept_for_source_chasing(self, page) -> None:
        assert ("https://www.nrc.gov/docs/filing.html", "regulatory filing") in page.links

    def test_malformed_markup_does_not_raise(self) -> None:
        page = extract_page("<html><body><p>unclosed <b>text<script>x=1")
        assert "text" in page.text

    def test_empty_input_is_handled(self) -> None:
        assert extract_page("").text == ""


class TestDateParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2025-03-04T10:00:00Z", "2025-03-04"),
            ("2025-03-04", "2025-03-04"),
            ("March 4, 2025", "2025-03-04"),
            ("4 March 2025", "2025-03-04"),
            ("20250304", "2025-03-04"),
        ],
    )
    def test_parses_publisher_formats(self, raw: str, expected: str) -> None:
        parsed = parse_date(raw)
        assert parsed is not None and parsed.date().isoformat() == expected

    @pytest.mark.parametrize("raw", ["", None, "sometime last year", "not a date"])
    def test_returns_none_rather_than_inventing_a_date(self, raw) -> None:
        assert parse_date(raw) is None


class TestClassification:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.sec.gov/Archives/edgar/data/1/0001.htm", SourceType.CORPORATE_FILING),
            ("https://www.nrc.gov/reading-rm/doc.html", SourceType.REGULATORY_DOCUMENT),
            ("https://www.energy.gov/articles/report", SourceType.REGULATORY_DOCUMENT),
            ("https://www.whitehouse.gov/briefing/x", SourceType.GOVERNMENT_DOCUMENT),
            ("https://acme.example/newsroom/launch", SourceType.PRESS_RELEASE),
            ("https://www.prnewswire.com/news-releases/x.html", SourceType.PRESS_RELEASE),
            ("https://www.youtube.com/watch?v=abc", SourceType.VIDEO),
            ("https://www.reddit.com/r/energy/comments/x", SourceType.FORUM_POST),
            ("https://bsky.app/profile/a/post/1", SourceType.SOCIAL_POST),
            ("https://example.com/some/page", SourceType.WEB_PAGE),
        ],
    )
    def test_maps_urls_to_source_types(self, url: str, expected: SourceType) -> None:
        assert classify_url(url) == expected

    def test_news_results_are_secondary_until_shown_otherwise(self) -> None:
        assert (
            classify_url("https://unknownpaper.example/story", is_news_result=True)
            is SourceType.SECONDARY_NEWS_REPORTING
        )

    def test_classification_is_metadata_not_a_credibility_score(self) -> None:
        # A filing outranks an article only in distance to the record.
        from research.models.common import PRIMARY_SOURCE_DISTANCE

        assert (
            PRIMARY_SOURCE_DISTANCE[SourceType.CORPORATE_FILING]
            < PRIMARY_SOURCE_DISTANCE[SourceType.SECONDARY_NEWS_REPORTING]
        )

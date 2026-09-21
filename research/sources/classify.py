"""Classifying retrieved web material by what it is.

These rules assign a :class:`~research.models.common.SourceType` from the
URL and page metadata. The label says where a document sits relative to the
underlying record - a filing is closer to the event than an article about the
filing - and nothing about whether it is true.

Where the evidence is thin, the answer is ``WEB_PAGE``. An honest 'unknown'
is more useful downstream than a confident guess.
"""

from __future__ import annotations

from research.models.common import SourceType
from research.normalize.urls import registrable_domain, url_host

#: Regulators whose documents are the primary record for their domain.
REGULATOR_HOSTS = frozenset(
    {
        "nrc.gov", "ferc.gov", "sec.gov", "epa.gov", "fda.gov", "ftc.gov",
        "cftc.gov", "occ.gov", "federalreserve.gov", "eia.gov", "doe.gov",
        "energy.gov", "esma.europa.eu", "ec.europa.eu", "ofgem.gov.uk",
        "fca.org.uk", "onr.org.uk", "iaea.org", "cnsc-ccsn.gc.ca",
    }
)

GOVERNMENT_SUFFIXES = (".gov", ".mil", ".gov.uk", ".gc.ca", ".gov.au", ".govt.nz", ".europa.eu")

VIDEO_HOSTS = frozenset({"youtube.com", "youtu.be", "vimeo.com", "dailymotion.com", "c-span.org"})

FORUM_HOSTS = frozenset({"reddit.com", "news.ycombinator.com", "stackexchange.com", "quora.com"})

SOCIAL_HOSTS = frozenset(
    {"x.com", "twitter.com", "bsky.app", "mastodon.social", "threads.net", "linkedin.com"}
)

PRESS_RELEASE_HOSTS = frozenset(
    {"prnewswire.com", "businesswire.com", "globenewswire.com", "newswire.ca", "eqs-news.com"}
)

_PRESS_RELEASE_PATHS = (
    "/press-release", "/press-releases", "/news-release", "/newsroom", "/media-release",
    "/press/", "/pressroom",
)

_FILING_MARKERS = ("/archives/edgar", "/cgi-bin/browse-edgar", "sec.gov/ix?doc=")


def classify_url(
    url: str | None,
    *,
    title: str | None = None,
    is_news_result: bool = False,
) -> SourceType:
    """Best available source type for a URL."""
    if not url:
        return SourceType.WEB_PAGE
    host = url_host(url) or ""
    domain = registrable_domain(host) or host
    path = url.split("://", 1)[-1][len(host):].lower() if host else url.lower()
    lowered_title = (title or "").strip().lower()

    if any(marker in url.lower() for marker in _FILING_MARKERS):
        return SourceType.CORPORATE_FILING
    if domain in VIDEO_HOSTS:
        return SourceType.VIDEO
    if domain in FORUM_HOSTS:
        return SourceType.FORUM_POST
    if domain in SOCIAL_HOSTS:
        return SourceType.SOCIAL_POST
    if domain in PRESS_RELEASE_HOSTS:
        return SourceType.PRESS_RELEASE
    if domain in REGULATOR_HOSTS or host in REGULATOR_HOSTS:
        return SourceType.REGULATORY_DOCUMENT
    if host.endswith(GOVERNMENT_SUFFIXES):
        return SourceType.GOVERNMENT_DOCUMENT
    if any(marker in path for marker in _PRESS_RELEASE_PATHS) or lowered_title.startswith(
        ("press release", "news release", "media release")
    ):
        return SourceType.PRESS_RELEASE
    if is_news_result:
        # 'Original reporting' is a finding, not a default. News results start
        # as secondary; the news researcher promotes an item only after
        # establishing that it is where the reporting originated.
        return SourceType.SECONDARY_NEWS_REPORTING
    return SourceType.WEB_PAGE

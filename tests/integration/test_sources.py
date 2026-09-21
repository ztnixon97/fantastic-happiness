"""Provider adapters against recorded payloads.

No live API is contacted. Each adapter is asked to translate a real response
shape into the common evidence vocabulary.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote

import httpx
import pytest

from research.config import AcquisitionPolicy
from research.errors import SourceNotConfigured, SourceRejected, SourceUnavailable
from research.models.common import RetrievalMethod, SourceFamily, SourceType
from research.models.query import ResearchQuery, SearchHit
from research.sources.arxiv import ArxivSource
from research.sources.crossref import CrossrefSource
from research.sources.http import SafeHttpClient
from research.sources.news import GdeltNewsSource, NewsSearchSource
from research.sources.openalex import OpenAlexSource
from research.sources.semantic_scholar import SemanticScholarSource
from research.sources.web import BraveSearchSource, TavilySearchSource

FIXTURES = Path(__file__).parent.parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def json_fixture(name: str):
    return json.loads(fixture(name))


def client_for(routes: dict[str, object], *, record: list | None = None) -> SafeHttpClient:
    """Serve payloads by URL substring; anything unrouted is a 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request)
        url = str(request.url)
        for marker, payload in routes.items():
            if marker in url:
                if isinstance(payload, httpx.Response):
                    return payload
                if isinstance(payload, str):
                    return httpx.Response(
                        200, text=payload, headers={"content-type": "application/atom+xml"}
                    )
                return httpx.Response(200, json=payload)
        return httpx.Response(404, json={"error": "no route"})

    return SafeHttpClient(
        AcquisitionPolicy(per_host_min_interval_seconds=0.0, retry_backoff_seconds=0.0),
        transport=httpx.MockTransport(handler),
    )


def query(text: str = "small modular reactor cost", **kwargs) -> ResearchQuery:
    return ResearchQuery(text=text, **kwargs)


class TestOpenAlex:
    @pytest.fixture
    def source(self):
        return OpenAlexSource(
            client_for({"/works": json_fixture("openalex_search.json")}),
            contact_email="research@example.org",
        )

    async def test_search_normalises_records(self, source) -> None:
        hits = await source.search(query())
        assert [hit.external_id for hit in hits] == ["W2741809807", "W999"]
        first = hits[0]
        assert first.doi == "10.1016/j.enpol.2024.114001"
        assert first.title.startswith("Levelized cost")
        assert first.authors == ["R. Okafor", "L. Meng"]
        assert first.published_at.date().isoformat() == "2024-04-12"
        assert first.publisher == "Energy Policy"
        assert first.source_type is SourceType.ACADEMIC_PEER_REVIEWED
        assert first.source_family is SourceFamily.ACADEMIC
        assert first.raw["pmid"] == "12345678"

    async def test_inverted_abstract_is_rebuilt(self, source) -> None:
        hits = await source.search(query())
        assert hits[0].snippet == "We review 42 cost estimates"

    async def test_preprints_are_not_labelled_peer_reviewed(self, source) -> None:
        hits = await source.search(query())
        assert hits[1].source_type is SourceType.ACADEMIC_PREPRINT
        assert hits[1].raw["arxiv_id"] == "2501.11234"

    async def test_contact_address_is_sent_for_the_polite_pool(self) -> None:
        requests: list[httpx.Request] = []
        source = OpenAlexSource(
            client_for({"/works": json_fixture("openalex_search.json")}, record=requests),
            contact_email="research@example.org",
        )
        await source.search(query())
        assert "mailto=research%40example.org" in str(requests[0].url)

    async def test_date_filters_become_provider_filters(self) -> None:
        from datetime import datetime, timezone

        requests: list[httpx.Request] = []
        source = OpenAlexSource(client_for({"/works": json_fixture("openalex_search.json")}, record=requests))
        await source.search(
            query(published_after=datetime(2020, 1, 1, tzinfo=timezone.utc))
        )
        assert "from_publication_date%3A2020-01-01" in str(requests[0].url)

    async def test_references_resolve_in_one_batched_request(self) -> None:
        requests: list[httpx.Request] = []
        source = OpenAlexSource(
            client_for(
                {
                    "filter=openalex": json_fixture("openalex_references.json"),
                    "/works": json_fixture("openalex_search.json"),
                },
                record=requests,
            )
        )
        seed = (await source.search(query()))[0]
        references = await source.get_references(seed, limit=10)
        assert [hit.external_id for hit in references] == ["W111", "W222"]
        assert references[0].raw["_retrieval_method"] == str(RetrievalMethod.REFERENCE_EXPANSION)
        assert len(requests) == 2, "one search plus one batched reference lookup"

    async def test_citing_papers_travel_forward(self) -> None:
        source = OpenAlexSource(
            client_for(
                {
                    "filter=cites": json_fixture("openalex_citing.json"),
                    "/works": json_fixture("openalex_search.json"),
                }
            )
        )
        seed = (await source.search(query()))[0]
        citing = await source.get_citing_papers(seed, limit=10)
        assert citing[0].external_id == "W333"
        assert citing[0].raw["_retrieval_method"] == str(RetrievalMethod.CITATION_EXPANSION)

    async def test_fetch_promotes_a_hit_without_another_request(self) -> None:
        requests: list[httpx.Request] = []
        source = OpenAlexSource(client_for({"/works": json_fixture("openalex_search.json")}, record=requests))
        hit = (await source.search(query()))[0]
        document = await source.fetch(hit)
        assert len(requests) == 1
        assert document.id == "", "identity belongs to the store, not the source"
        assert document.doi == "10.1016/j.enpol.2024.114001"
        assert document.provenance.provider == "openalex"


class TestCrossref:
    @pytest.fixture
    def source(self):
        return CrossrefSource(
            client_for(
                {
                    "/works/10.1016/j.joule": json_fixture("crossref_work.json"),
                    "/works": json_fixture("crossref_search.json"),
                }
            ),
            contact_email="research@example.org",
        )

    async def test_search_normalises_records(self, source) -> None:
        hits = await source.search(query())
        assert hits[0].doi == "10.1016/j.enpol.2024.114001"  # case-folded
        assert hits[0].authors == ["Rachel Okafor", "Li Meng"]
        assert hits[0].snippet == "We review 42 published cost estimates."  # JATS stripped
        assert hits[0].published_at.date().isoformat() == "2024-04-12"

    async def test_partial_dates_do_not_become_precise_ones(self, source) -> None:
        hits = await source.search(query())
        assert hits[1].published_at.date().isoformat() == "2023-08-01"

    async def test_references_resolve_only_deposited_dois(self, source) -> None:
        seed = (await source.search(query()))[0]
        references = await source.get_references(seed, limit=10)
        assert [hit.doi for hit in references] == ["10.1016/j.joule.2021.06.004"]

    async def test_forward_citations_are_declared_unavailable(self, source) -> None:
        assert source.capabilities().supports_citing_papers is False
        assert await source.get_citing_papers(SearchHit(provider="crossref")) == []


class TestSemanticScholar:
    async def test_search_normalises_records(self) -> None:
        source = SemanticScholarSource(
            client_for({"/paper/search": json_fixture("semantic_scholar_search.json")})
        )
        hits = await source.search(query())
        assert hits[0].external_id == "abc123"
        assert hits[0].doi == "10.1016/j.enpol.2024.114001"
        assert hits[0].raw["is_review"] is True
        assert hits[0].raw["pmid"] == "12345678"

    async def test_citation_intents_are_preserved(self) -> None:
        source = SemanticScholarSource(
            client_for(
                {
                    "/citations": json_fixture("semantic_scholar_citations.json"),
                    "/paper/search": json_fixture("semantic_scholar_search.json"),
                }
            )
        )
        seed = (await source.search(query()))[0]
        citing = await source.get_citing_papers(seed, limit=5)
        assert citing[0].title.startswith("A replication")
        assert citing[0].raw["citation_intents"] == ["methodology"]
        assert citing[0].raw["influential_citation"] is True

    async def test_api_key_is_sent_only_when_configured(self) -> None:
        requests: list[httpx.Request] = []
        keyed = SemanticScholarSource(
            client_for({"/paper/search": json_fixture("semantic_scholar_search.json")}, record=requests),
            api_key="secret-key",
        )
        await keyed.search(query())
        assert requests[0].headers["x-api-key"] == "secret-key"

        requests.clear()
        keyless = SemanticScholarSource(
            client_for(
                {"/paper/search": json_fixture("semantic_scholar_search.json")},
                record=requests,
            )
        )
        await keyless.search(query())
        assert "x-api-key" not in requests[0].headers


class TestArxiv:
    @pytest.fixture
    def source(self):
        return ArxivSource(client_for({"/query": fixture("arxiv_search.xml")}))

    async def test_atom_feed_is_parsed(self, source) -> None:
        hits = await source.search(query())
        assert hits[0].external_id == "2501.11234"  # version suffix removed
        assert hits[0].authors == ["J. Whitfield", "A. Osei"]
        assert hits[0].url == "https://arxiv.org/abs/2501.11234"
        assert hits[0].raw["primary_category"] == "eess.SY"

    async def test_records_are_always_preprints(self, source) -> None:
        hits = await source.search(query())
        assert hits[0].source_type is SourceType.ACADEMIC_PREPRINT
        assert hits[0].doi == "10.1016/j.apenergy.2025.12345"

    async def test_malformed_feed_reports_a_source_failure(self) -> None:
        source = ArxivSource(client_for({"/query": "<feed>truncated"}))
        with pytest.raises(SourceUnavailable):
            await source.search(query())


class TestNews:
    async def test_gdelt_needs_no_credentials(self) -> None:
        source = GdeltNewsSource(client_for({"/doc/doc": json_fixture("gdelt_search.json")}))
        assert source.capabilities().requires_api_key is False
        hits = await source.search(query("nuscale", families=[SourceFamily.NEWS]))
        assert len(hits) == 2
        assert hits[0].source_family is SourceFamily.NEWS
        assert hits[0].publisher == "reuters.com"
        assert hits[0].published_at.date().isoformat() == "2023-11-08"
        assert hits[0].raw["seendate_is_index_time"] is True

    async def test_gdelt_results_are_secondary_until_shown_otherwise(self) -> None:
        source = GdeltNewsSource(client_for({"/doc/doc": json_fixture("gdelt_search.json")}))
        hits = await source.search(query("nuscale", families=[SourceFamily.NEWS]))
        assert all(hit.source_type is SourceType.SECONDARY_NEWS_REPORTING for hit in hits)

    async def test_news_wrapper_reshapes_a_web_provider(self) -> None:
        requests: list[httpx.Request] = []
        brave = BraveSearchSource(
            client_for({"/news/search": json_fixture("brave_news.json")}, record=requests),
            api_key="k",
        )
        source = NewsSearchSource(brave)
        hits = await source.search(query("smr"))
        assert "/news/search" in str(requests[0].url)
        assert hits[0].source_family is SourceFamily.NEWS
        assert source.capabilities().families == (SourceFamily.NEWS,)


class TestWebProviders:
    async def test_brave_results_are_normalised(self) -> None:
        source = BraveSearchSource(
            client_for({"/web/search": json_fixture("brave_search.json")}), api_key="k"
        )
        hits = await source.search(query("nrc design approval"))
        assert hits[0].url == "https://www.nrc.gov/reactors/design-approval.html"
        assert hits[0].source_type is SourceType.REGULATORY_DOCUMENT

    async def test_tavily_results_are_normalised(self) -> None:
        source = TavilySearchSource(
            client_for({"/search": json_fixture("tavily_search.json")}), api_key="k"
        )
        hits = await source.search(query("annual report"))
        assert hits[0].source_type is SourceType.CORPORATE_FILING

    async def test_unconfigured_provider_says_so_clearly(self) -> None:
        source = BraveSearchSource(client_for({}), api_key=None)
        with pytest.raises(SourceNotConfigured):
            await source.search(query())

    async def test_search_snippets_are_never_promoted_to_evidence(self) -> None:
        source = BraveSearchSource(client_for({}), api_key="k")
        with pytest.raises(SourceUnavailable):
            await source.fetch(SearchHit(provider="brave", url="https://e.com/a"))


class TestProviderFailures:
    async def test_server_errors_are_retryable_failures(self) -> None:
        source = OpenAlexSource(client_for({"/works": httpx.Response(503, json={})}))
        with pytest.raises(SourceUnavailable) as exc:
            await source.search(query())
        assert exc.value.retryable

    async def test_client_errors_are_not_retried(self) -> None:
        source = OpenAlexSource(client_for({"/works": httpx.Response(400, json={})}))
        with pytest.raises(SourceRejected):
            await source.search(query())

    async def test_transient_failures_are_retried_then_succeed(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            if len(attempts) == 1:
                return httpx.Response(429, headers={"retry-after": "0"})
            return httpx.Response(200, json=json_fixture("openalex_search.json"))

        client = SafeHttpClient(
            AcquisitionPolicy(per_host_min_interval_seconds=0.0, retry_backoff_seconds=0.0),
            transport=httpx.MockTransport(handler),
        )
        hits = await OpenAlexSource(client).search(query())
        assert len(attempts) == 2
        assert hits


class TestSocialSources:
    """Public social material: attributed statements, not verification."""

    async def test_bluesky_posts_carry_their_account(self) -> None:
        from research.sources.social import BlueskySource

        source = BlueskySource(
            client_for({"searchPosts": json_fixture("bluesky_search.json")})
        )
        hits = await source.search(query("smr cancellation", families=[SourceFamily.SOCIAL]))
        assert len(hits) == 2
        first = hits[0]
        assert first.source_type is SourceType.SOCIAL_POST
        assert first.source_family is SourceFamily.SOCIAL
        assert first.url == "https://bsky.app/profile/utilityanalyst.bsky.social/post/3kxyz"
        assert first.raw["account_handle"] == "utilityanalyst.bsky.social"
        assert first.raw["account_did"] == "did:plc:abc123"
        assert first.published_at.date().isoformat() == "2023-11-09"

    async def test_bluesky_needs_no_credential(self) -> None:
        from research.sources.social import BlueskySource

        assert BlueskySource(client_for({})).capabilities().requires_api_key is False

    async def test_a_post_becomes_self_reported_evidence(self) -> None:
        from research.graph.claims import SELF_REPORTED_TYPES
        from research.sources.social import BlueskySource

        source = BlueskySource(
            client_for({"searchPosts": json_fixture("bluesky_search.json")})
        )
        hit = (await source.search(query("smr", families=[SourceFamily.SOCIAL])))[0]
        document = await source.fetch(hit)
        assert document.source_type in SELF_REPORTED_TYPES
        assert "binding constraint" in document.text
        assert document.metadata["account_handle"] == "utilityanalyst.bsky.social"

    async def test_mastodon_reads_public_tag_timelines(self) -> None:
        from research.sources.social import MastodonSource

        requests: list[httpx.Request] = []
        source = MastodonSource(
            client_for({"/timelines/tag/": json_fixture("mastodon_tag.json")}, record=requests)
        )
        hits = await source.search(query("nuclear data centre agreements",
                                         families=[SourceFamily.SOCIAL]))
        assert "/api/v1/timelines/tag/nuclear" in str(requests[0].url)
        assert len(hits) == 1, "a boost is a copy, not a second source"
        assert hits[0].raw["account_handle"] == "gridwatcher@mastodon.social"
        # HTML post bodies are stripped by the same extractor the fetcher uses.
        assert "<p>" not in hits[0].snippet
        assert "broken ground" in hits[0].snippet

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("nuclear energy policy debate", "nuclear"),
            ("recent news about smr cancellations", "cancellations"),
            ("datacentre power agreements", "datacentre"),
        ],
    )
    async def test_the_tag_is_the_subject_not_the_longest_word(
        self, text: str, expected: str
    ) -> None:
        from research.sources.social import MastodonSource

        requests: list[httpx.Request] = []
        source = MastodonSource(
            client_for({"/timelines/tag/": json_fixture("mastodon_tag.json")}, record=requests)
        )
        await source.search(query(text, families=[SourceFamily.SOCIAL]))
        assert f"/timelines/tag/{expected}" in str(requests[0].url)

    async def test_mastodon_honours_an_explicit_hashtag(self) -> None:
        from research.sources.social import MastodonSource

        requests: list[httpx.Request] = []
        source = MastodonSource(
            client_for({"/timelines/tag/": json_fixture("mastodon_tag.json")}, record=requests)
        )
        await source.search(
            ResearchQuery(text="anything", filters={"hashtag": "#SMR"},
                          families=[SourceFamily.SOCIAL])
        )
        assert "/timelines/tag/smr" in str(requests[0].url)

    async def test_youtube_returns_video_records_and_says_so(self) -> None:
        from research.sources.social import YouTubeSource

        source = YouTubeSource(
            client_for({"/search": json_fixture("youtube_search.json")}), api_key="k"
        )
        hits = await source.search(query("smr economics", families=[SourceFamily.VIDEO]))
        assert hits[0].source_type is SourceType.VIDEO
        assert hits[0].url == "https://www.youtube.com/watch?v=vid123"
        assert hits[0].raw["transcript_available"] is False
        assert source.capabilities().supports_full_text is False

    async def test_youtube_without_a_key_says_what_is_missing(self) -> None:
        from research.sources.social import YouTubeSource

        source = YouTubeSource(client_for({}), api_key=None)
        with pytest.raises(SourceNotConfigured) as exc:
            await source.search(query("x", families=[SourceFamily.VIDEO]))
        assert "RESEARCH_YOUTUBE_API_KEY" in str(exc.value)


class TestProviderQuirks:
    """Regressions for things only a live call revealed."""

    def test_crossref_selects_only_fields_that_route_accepts(self) -> None:
        from research.sources.crossref import SELECT_FIELDS

        # Crossref rejects the entire request when one select field is not
        # available on /works. 'language' is returned in full records but is
        # not selectable there, and asking for it made every search fail.
        assert "language" not in SELECT_FIELDS
        assert "DOI" in SELECT_FIELDS and "abstract" in SELECT_FIELDS

    async def test_arxiv_ands_the_leading_terms(self) -> None:
        requests: list[httpx.Request] = []
        source = ArxivSource(client_for({"/query": fixture("arxiv_search.xml")}, record=requests))
        await source.search(query("small modular reactor levelized cost of electricity"))
        sent = unquote(str(requests[0].url)).replace("+", " ")
        # A bag of words matches on any term and returns whatever is most
        # cited for the commonest one; ANDing every term returns nothing.
        assert "all:small AND all:modular AND all:reactor AND all:levelized" in sent
        assert "all:electricity" not in sent, "a long AND chain over-restricts"
        assert '"' not in sent, "a quoted phrase matches nothing on arXiv"

    async def test_arxiv_quotes_when_a_phrase_is_asked_for(self) -> None:
        requests: list[httpx.Request] = []
        source = ArxivSource(client_for({"/query": fixture("arxiv_search.xml")}, record=requests))
        await source.search(
            ResearchQuery(text="small modular reactor", filters={"phrase": True})
        )
        assert "%22" in str(requests[0].url)

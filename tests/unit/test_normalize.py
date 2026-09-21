"""URL, DOI, text and fingerprint normalisation."""

from __future__ import annotations

import pytest

from research.normalize import (
    canonicalize_url,
    containment,
    content_hash,
    doi_to_url,
    extract_doi,
    hamming_distance,
    jaccard,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
    registrable_domain,
    shingles,
    simhash,
    simhash_bands,
    title_key,
    url_host,
    url_identity_key,
)


class TestCanonicalizeUrl:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://Example.COM/A/B/", "https://example.com/A/B"),
            ("http://example.com:80/x", "http://example.com/x"),
            ("https://example.com:443/x", "https://example.com/x"),
            ("https://example.com/x#section", "https://example.com/x"),
            ("example.com/x", "https://example.com/x"),
            ("https://example.com", "https://example.com/"),
        ],
    )
    def test_normalises(self, raw: str, expected: str) -> None:
        assert canonicalize_url(raw) == expected

    def test_strips_tracking_parameters_but_keeps_meaningful_ones(self) -> None:
        url = "https://example.com/a?utm_source=twitter&id=7&fbclid=xyz&page=2"
        assert canonicalize_url(url) == "https://example.com/a?id=7&page=2"

    def test_sorts_query_parameters_so_order_is_not_identity(self) -> None:
        assert canonicalize_url("https://e.com/a?b=2&a=1") == canonicalize_url(
            "https://e.com/a?a=1&b=2"
        )

    def test_unwraps_known_redirectors(self) -> None:
        wrapped = (
            "https://news.google.com/rss/articles/CBMi?url="
            "https%3A%2F%2Fapnews.com%2Farticle%2Fabc&hl=en"
        )
        assert canonicalize_url(wrapped) == "https://apnews.com/article/abc"

    @pytest.mark.parametrize(
        "raw", ["javascript:alert(1)", "data:text/html,x", "mailto:a@b.c", "", None, "#anchor"]
    )
    def test_refuses_non_document_urls(self, raw) -> None:
        assert canonicalize_url(raw) is None


class TestUrlIdentity:
    def test_amp_and_www_variants_share_an_identity(self) -> None:
        assert url_identity_key("https://www.bbc.co.uk/news/story-1/amp") == url_identity_key(
            "https://bbc.co.uk/news/story-1"
        )

    def test_different_documents_keep_different_identities(self) -> None:
        assert url_identity_key("https://e.com/a") != url_identity_key("https://e.com/b")

    def test_host_strips_subdomain_prefixes(self) -> None:
        assert url_host("https://m.example.com/a") == "example.com"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("https://sub.news.bbc.co.uk/x", "bbc.co.uk"),
            ("https://www.reuters.com/a", "reuters.com"),
            ("https://example.org", "example.org"),
        ],
    )
    def test_registrable_domain(self, raw: str, expected: str) -> None:
        assert registrable_domain(raw) == expected


class TestDoi:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("10.1038/S41586-020-2649-2", "10.1038/s41586-020-2649-2"),
            ("https://doi.org/10.1038/x", "10.1038/x"),
            ("http://dx.doi.org/10.1038/x", "10.1038/x"),
            ("doi:10.1038/x", "10.1038/x"),
            ("DOI: 10.1038/x.", "10.1038/x"),
            ("see 10.1234/abc).", "10.1234/abc"),
            ("10.1234/abc(1)", "10.1234/abc(1)"),
        ],
    )
    def test_normalises(self, raw: str, expected: str) -> None:
        assert normalize_doi(raw) == expected

    @pytest.mark.parametrize("raw", ["", None, "not a doi", "10.x/abc", "https://example.com/a"])
    def test_rejects_non_dois(self, raw) -> None:
        assert normalize_doi(raw) is None

    def test_case_folding_makes_providers_agree(self) -> None:
        assert normalize_doi("10.1000/ABC") == normalize_doi("https://doi.org/10.1000/abc")

    def test_extracts_from_prose(self) -> None:
        text = "As shown in 10.1016/j.enpol.2024.114001 and 10.1038/s41586-020-2649-2."
        assert extract_doi(text) == [
            "10.1016/j.enpol.2024.114001",
            "10.1038/s41586-020-2649-2",
        ]

    def test_round_trips_to_url(self) -> None:
        assert doi_to_url("DOI:10.1000/xyz") == "https://doi.org/10.1000/xyz"


class TestArxiv:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("arXiv:2401.01234v2", "2401.01234"),
            ("https://arxiv.org/abs/2401.01234", "2401.01234"),
            ("https://arxiv.org/pdf/2401.01234v3.pdf", "2401.01234"),
            ("10.48550/arXiv.2401.01234", "2401.01234"),
            ("hep-th/9901001v3", "hep-th/9901001"),
        ],
    )
    def test_normalises(self, raw: str, expected: str) -> None:
        assert normalize_arxiv_id(raw) == expected

    def test_rejects_unrelated_urls(self) -> None:
        assert normalize_arxiv_id("https://example.com/abs/2401.01234") is None


class TestTitles:
    def test_drops_outlet_suffix_and_punctuation(self) -> None:
        assert normalize_title("The Costs of SMRs - Reuters") == "costs of smrs"

    def test_key_is_stable_across_formatting(self) -> None:
        assert title_key("Small Modular Reactors: A Review!") == title_key(
            "small modular reactors a review"
        )

    def test_refuses_keys_too_weak_to_identify_a_document(self) -> None:
        assert title_key("Q3 results") is None
        assert title_key("Introduction") is None


class TestFingerprints:
    def test_hash_ignores_case_and_whitespace(self) -> None:
        assert content_hash("A  B\n\nC") == content_hash("a b\nc")

    def test_hash_separates_fields(self) -> None:
        assert content_hash("ab", "c") != content_hash("a", "bc")

    def test_identical_text_has_distance_zero(self) -> None:
        text = "small modular reactors cost more than projected in every published review " * 4
        assert hamming_distance(simhash(text), simhash(text)) == 0

    def test_simhash_needs_enough_text_to_be_meaningful(self) -> None:
        assert simhash("too short") is None

    def test_bands_index_a_fingerprint(self) -> None:
        text = "the utility group said it would review the plan for the reactor project " * 3
        bands = simhash_bands(simhash(text))
        assert len(bands) == 4
        assert simhash_bands(None) == []

    def test_jaccard_and_containment_differ_on_supersets(self) -> None:
        small = shingles("alpha beta gamma delta epsilon", 3)
        large = shingles("alpha beta gamma delta epsilon zeta eta theta iota kappa", 3)
        assert containment(small, large) == 1.0
        assert jaccard(small, large) < 1.0

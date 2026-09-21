"""Public social sources.

A social post is evidence that somebody said something. It is rarely evidence
that what they said is so, and the rest of the system treats it that way:
posts are ``social_post``, which claim assessment counts as self-reported, so
a claim resting only on posts is never marked supported.

What these adapters do add is attribution. Each post carries the account that
made it, as an identifier the entity registry can resolve, so 'the company
said X' can be separated from 'an account claiming to be the company said X'.

Reddit is deliberately absent: its API requires registered OAuth credentials
and its terms restrict what may be stored and redistributed, so it is not
something to enable by default. The interface is ready for it if an operator
has the standing to use it.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from research.errors import SourceNotConfigured, SourceUnavailable
from research.models.common import RetrievalMethod, SourceFamily, SourceType
from research.models.evidence import EvidenceDocument
from research.models.query import ResearchQuery, SearchHit
from research.normalize.document import build_document
from research.normalize.html import extract_page
from research.normalize.text import normalize_whitespace, truncate
from research.normalize.urls import canonicalize_url
from research.sources.base import SourceAdapter, SourceCapabilities
from research.sources.http import SafeHttpClient

_HASHTAG_RE = re.compile(r"[^A-Za-z0-9]+")


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class BlueskySource(SourceAdapter):
    """Bluesky's public AppView. No credential, no login."""

    name = "bluesky"
    family = SourceFamily.SOCIAL
    source_type = SourceType.SOCIAL_POST
    base_url = "https://public.api.bsky.app"

    def __init__(self, client: SafeHttpClient, *, base_url: str | None = None) -> None:
        self.client = client
        self.base_url = (base_url or self.base_url).rstrip("/")

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=self.name,
            families=(SourceFamily.SOCIAL,),
            default_source_type=SourceType.SOCIAL_POST,
            returns_inline_documents=True,
            supports_date_filter=True,
            requires_api_key=False,
            max_results_per_query=100,
            tags=frozenset({"keyless", "social"}),
            notes="Public post search. Posts are attributed statements, not verification.",
        )

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        params: dict[str, Any] = {"q": query.text, "limit": min(query.limit, 100)}
        if query.published_after:
            params["since"] = query.published_after.date().isoformat()
        if query.published_before:
            params["until"] = query.published_before.date().isoformat()
        if query.language:
            params["lang"] = query.language

        payload = await self.client.get_json(
            f"{self.base_url}/xrpc/app.bsky.feed.searchPosts",
            provider=self.name,
            params=params,
        )
        if not isinstance(payload, dict):
            raise SourceUnavailable("unexpected Bluesky payload", provider=self.name)
        return [
            hit
            for hit in (self._to_hit(post) for post in payload.get("posts", []))
            if hit is not None
        ]

    def _to_hit(self, post: Any) -> SearchHit | None:
        if not isinstance(post, dict):
            return None
        record = post.get("record") or {}
        author = post.get("author") or {}
        handle = author.get("handle")
        uri = post.get("uri") or ""
        rkey = uri.rsplit("/", 1)[-1] if uri else None
        text = normalize_whitespace(record.get("text") or "")
        if not text or not handle:
            return None
        return SearchHit(
            provider=self.name,
            external_id=uri or None,
            url=f"https://bsky.app/profile/{handle}/post/{rkey}" if rkey else None,
            title=truncate(text, 120),
            snippet=text,
            authors=[author.get("displayName") or handle],
            published_at=_parse_time(record.get("createdAt") or post.get("indexedAt")),
            source_type=SourceType.SOCIAL_POST,
            source_family=SourceFamily.SOCIAL,
            publisher="bsky.app",
            language=(record.get("langs") or [None])[0],
            raw={
                "account_handle": handle,
                "account_did": author.get("did"),
                "account_display_name": author.get("displayName"),
                # Engagement counts describe reach, never accuracy.
                "reply_count": post.get("replyCount"),
                "repost_count": post.get("repostCount"),
                "like_count": post.get("likeCount"),
                "post_text": text,
                "_endpoint": f"{self.base_url}/xrpc/app.bsky.feed.searchPosts",
                "_retrieval_method": str(RetrievalMethod.SEARCH),
            },
        )

    async def fetch(self, hit: SearchHit) -> EvidenceDocument:
        return _document_from_post(hit, provider=self.name)


class MastodonSource(SourceAdapter):
    """Public hashtag timelines on a Mastodon instance.

    Full-text search on Mastodon needs an account token, so the keyless route
    is the public tag timeline: the query's strongest keyword becomes the tag.
    That is narrower than search and the capabilities say so.
    """

    name = "mastodon"
    family = SourceFamily.SOCIAL
    source_type = SourceType.SOCIAL_POST
    base_url = "https://mastodon.social"

    def __init__(
        self,
        client: SafeHttpClient,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.client = client
        self.base_url = (base_url or self.base_url).rstrip("/")
        self._api_key = api_key

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=self.name,
            families=(SourceFamily.SOCIAL,),
            default_source_type=SourceType.SOCIAL_POST,
            returns_inline_documents=True,
            supports_date_filter=False,
            requires_api_key=False,
            max_results_per_query=40,
            tags=frozenset({"keyless", "social", "tag-only"}),
            notes=(
                "Public hashtag timelines on one instance. Full-text search would "
                "need an account token; this does not."
            ),
        )

    def _tag(self, query: ResearchQuery) -> str:
        explicit = query.filters.get("hashtag")
        if explicit:
            return _HASHTAG_RE.sub("", str(explicit)).lower()
        words = [word for word in query.text.split() if len(word) > 3]
        longest = max(words, key=len) if words else query.text
        return _HASHTAG_RE.sub("", longest).lower()

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        tag = self._tag(query)
        if not tag:
            return []
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else None
        endpoint = f"{self.base_url}/api/v1/timelines/tag/{tag}"
        payload = await self.client.get_json(
            endpoint,
            provider=self.name,
            params={"limit": min(query.limit, 40)},
            headers=headers,
        )
        if not isinstance(payload, list):
            raise SourceUnavailable("unexpected Mastodon payload", provider=self.name)
        hits = [
            hit
            for hit in (self._to_hit(status, tag, endpoint) for status in payload)
            if hit is not None
        ]
        return hits

    def _to_hit(self, status: Any, tag: str, endpoint: str) -> SearchHit | None:
        if not isinstance(status, dict) or status.get("reblog"):
            # A boost is a copy of someone else's post, not a second source.
            return None
        account = status.get("account") or {}
        handle = account.get("acct")
        # Post bodies are HTML; the same extractor the web fetcher uses strips it.
        text = normalize_whitespace(extract_page(status.get("content") or "").text)
        if not text or not handle:
            return None
        instance = self.base_url.split("://", 1)[-1]
        return SearchHit(
            provider=self.name,
            external_id=status.get("uri") or status.get("id"),
            url=canonicalize_url(status.get("url")),
            title=truncate(text, 120),
            snippet=text,
            authors=[account.get("display_name") or handle],
            published_at=_parse_time(status.get("created_at")),
            source_type=SourceType.SOCIAL_POST,
            source_family=SourceFamily.SOCIAL,
            publisher=instance,
            language=status.get("language"),
            raw={
                "account_handle": handle if "@" in handle else f"{handle}@{instance}",
                "account_url": account.get("url"),
                "account_display_name": account.get("display_name"),
                "account_bot": account.get("bot"),
                "hashtag": tag,
                "reblog_count": status.get("reblogs_count"),
                "favourite_count": status.get("favourites_count"),
                "post_text": text,
                "_endpoint": endpoint,
                "_retrieval_method": str(RetrievalMethod.SEARCH),
            },
        )

    async def fetch(self, hit: SearchHit) -> EvidenceDocument:
        return _document_from_post(hit, provider=self.name)


class YouTubeSource(SourceAdapter):
    """YouTube Data API search.

    Returns video records - title, description, channel, date - not
    transcripts. Caption download needs OAuth as the video's owner, so
    transcript evidence is a separate capability rather than something this
    adapter can quietly pretend to provide.
    """

    name = "youtube"
    family = SourceFamily.VIDEO
    source_type = SourceType.VIDEO
    base_url = "https://www.googleapis.com/youtube/v3"

    def __init__(
        self,
        client: SafeHttpClient,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.client = client
        self._api_key = api_key
        self.base_url = (base_url or self.base_url).rstrip("/")

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=self.name,
            families=(SourceFamily.VIDEO, SourceFamily.SOCIAL),
            default_source_type=SourceType.VIDEO,
            returns_inline_documents=True,
            supports_full_text=False,
            supports_date_filter=True,
            requires_api_key=True,
            max_results_per_query=50,
            tags=frozenset({"video"}),
            notes="Video metadata only; transcripts need a separate capability.",
        )

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        if not self._api_key:
            raise SourceNotConfigured(
                "YouTube needs an API key; set RESEARCH_YOUTUBE_API_KEY to enable it",
                provider=self.name,
            )
        params: dict[str, Any] = {
            "part": "snippet",
            "q": query.text,
            "type": "video",
            "maxResults": min(query.limit, 50),
            "key": self._api_key,
        }
        if query.published_after:
            params["publishedAfter"] = query.published_after.strftime("%Y-%m-%dT%H:%M:%SZ")
        if query.published_before:
            params["publishedBefore"] = query.published_before.strftime("%Y-%m-%dT%H:%M:%SZ")

        payload = await self.client.get_json(
            f"{self.base_url}/search", provider=self.name, params=params
        )
        if not isinstance(payload, dict):
            raise SourceUnavailable("unexpected YouTube payload", provider=self.name)
        return [
            hit
            for hit in (self._to_hit(item) for item in payload.get("items", []))
            if hit is not None
        ]

    def _to_hit(self, item: Any) -> SearchHit | None:
        if not isinstance(item, dict):
            return None
        video_id = (item.get("id") or {}).get("videoId")
        snippet = item.get("snippet") or {}
        if not video_id or not snippet.get("title"):
            return None
        return SearchHit(
            provider=self.name,
            external_id=video_id,
            url=f"https://www.youtube.com/watch?v={video_id}",
            title=normalize_whitespace(snippet["title"]),
            snippet=normalize_whitespace(snippet.get("description") or ""),
            authors=[snippet.get("channelTitle")] if snippet.get("channelTitle") else [],
            published_at=_parse_time(snippet.get("publishedAt")),
            source_type=SourceType.VIDEO,
            source_family=SourceFamily.VIDEO,
            publisher=snippet.get("channelTitle"),
            raw={
                "video_id": video_id,
                "channel_id": snippet.get("channelId"),
                "channel_title": snippet.get("channelTitle"),
                # Stated plainly so nothing downstream mistakes a description
                # for what was said in the video.
                "transcript_available": False,
                "_endpoint": f"{self.base_url}/search",
                "_retrieval_method": str(RetrievalMethod.SEARCH),
            },
        )

    async def fetch(self, hit: SearchHit) -> EvidenceDocument:
        return _document_from_post(hit, provider=self.name, family=SourceFamily.VIDEO)


def _document_from_post(
    hit: SearchHit,
    *,
    provider: str,
    family: SourceFamily = SourceFamily.SOCIAL,
) -> EvidenceDocument:
    """Promote a post or video record to evidence.

    The account is kept as structured metadata rather than folded into the
    text, so a worker can attribute a statement without parsing prose.
    """
    from research.models.common import Provenance

    metadata = {key: value for key, value in hit.raw.items() if not key.startswith("_")}
    return build_document(
        provider=provider,
        source_type=hit.source_type,
        source_family=family,
        provenance=Provenance(
            provider=provider,
            retrieval_method=RetrievalMethod.SEARCH,
            provider_endpoint=str(hit.raw.get("_endpoint") or ""),
            requested_url=hit.url,
        ),
        title=hit.title,
        text=hit.raw.get("post_text") or hit.snippet,
        url=hit.url,
        external_id=hit.external_id,
        authors=[author for author in hit.authors if author],
        published_at=hit.published_at,
        publisher=hit.publisher,
        language=hit.language,
        metadata=metadata,
    )

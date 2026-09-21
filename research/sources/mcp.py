"""Model Context Protocol servers as research sources.

An MCP server that exposes a search tool is a provider like any other, so it
belongs behind :class:`~research.sources.base.ResearchSource` rather than in
the agent's vocabulary. A worker still asks for "academic literature"; that
one of the things answering is an MCP server is a configuration detail, and
the closed action vocabulary does not grow by a single verb.

**HTTP servers only, deliberately.** The usual way to run an MCP server is
stdio: the client launches the server as a subprocess. This system does not
launch subprocesses - the research loop has no shell, and a test asserts
that nothing in the package imports ``subprocess`` - and reading a program
path out of a config file and executing it is exactly the thing that rule
exists to prevent. An operator who wants a stdio server can run it and point
this at its address; what crosses into this process is then a network
response, which the same SSRF guard, content-type allowlist, size cap and
timeout apply to as to any other provider.

What is implemented is the JSON-RPC subset a source needs: ``initialize``,
``tools/list``, and ``tools/call``. Anything a tool returns is external
content and is treated as such - it becomes a search hit or a document, it
never becomes an instruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from research.errors import SourceNotConfigured, SourceUnavailable
from research.models.common import Provenance, RetrievalMethod, SourceFamily, SourceType
from research.models.evidence import EvidenceDocument
from research.models.query import ResearchQuery, SearchHit
from research.normalize.document import build_document
from research.normalize.text import clean_text, normalize_whitespace
from research.sources.base import SourceCapabilities
from research.sources.http import SafeHttpClient

PROTOCOL_VERSION = "2025-06-18"

#: Tool names, in the order preferred, that a server might offer for search.
#: Nothing is guessed beyond a name: a server naming its tool something else
#: is named explicitly in configuration.
SEARCH_TOOL_NAMES = ("search", "web_search", "search_documents", "query", "find")
FETCH_TOOL_NAMES = ("fetch", "get_document", "read", "open")


@dataclass(slots=True)
class McpTool:
    name: str
    description: str = ""
    schema: dict[str, Any] = field(default_factory=dict)

    def query_argument(self) -> str:
        """The argument that takes the search text, from the tool's own schema."""
        properties = (self.schema or {}).get("properties") or {}
        for candidate in ("query", "q", "search", "question", "text", "keywords"):
            if candidate in properties:
                return candidate
        required = (self.schema or {}).get("required") or []
        if required:
            return str(required[0])
        return "query"


class McpClient:
    """The slice of MCP a source needs, over Streamable HTTP."""

    def __init__(
        self,
        client: SafeHttpClient,
        url: str,
        *,
        name: str = "mcp",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.client = client
        self.url = url
        self.name = name
        self.headers = dict(headers or {})
        self._next_id = 0
        self._tools: list[McpTool] | None = None
        self._session: str | None = None

    async def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._next_id += 1
        headers = {
            "content-type": "application/json",
            # Streamable HTTP servers may answer either way.
            "accept": "application/json, text/event-stream",
            **self.headers,
        }
        if self._session:
            headers["mcp-session-id"] = self._session

        response = await self.client.request(
            "POST",
            self.url,
            provider=self.name,
            headers=headers,
            json_body={
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": method,
                **({"params": params} if params is not None else {}),
            },
        )
        session = response.headers.get("mcp-session-id")
        if session:
            self._session = session

        payload = _decode(response.text, provider=self.name)
        if isinstance(payload, dict) and payload.get("error"):
            error = payload["error"]
            raise SourceUnavailable(
                f"{self.name} refused {method}: "
                f"{error.get('message') or error}",
                provider=self.name,
            )
        return (payload or {}).get("result") if isinstance(payload, dict) else None

    async def initialize(self) -> None:
        await self.call(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "research", "version": "0.1.0"},
            },
        )

    async def tools(self) -> list[McpTool]:
        if self._tools is not None:
            return self._tools
        await self.initialize()
        result = await self.call("tools/list") or {}
        self._tools = [
            McpTool(
                name=str(entry.get("name") or ""),
                description=normalize_whitespace(str(entry.get("description") or ""))[:400],
                schema=entry.get("inputSchema") or {},
            )
            for entry in (result.get("tools") or [])
            if entry.get("name")
        ]
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Call a tool and return its content blocks.

        Blocks are external content. They are parsed into hits or documents
        and are never interpreted as anything else, whatever they contain.
        """
        result = await self.call("tools/call", {"name": name, "arguments": arguments}) or {}
        if result.get("isError"):
            raise SourceUnavailable(
                f"{self.name} tool {name!r} reported an error", provider=self.name
            )
        return [block for block in (result.get("content") or []) if isinstance(block, dict)]


class McpSource:
    """One MCP server, presented as a research source."""

    def __init__(
        self,
        client: SafeHttpClient,
        *,
        url: str | None,
        name: str = "mcp",
        family: SourceFamily = SourceFamily.WEB,
        source_type: SourceType = SourceType.WEB_PAGE,
        search_tool: str | None = None,
        fetch_tool: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.family = family
        self.source_type = source_type
        self.url = url
        self._search_tool = search_tool
        self._fetch_tool = fetch_tool
        self.mcp = McpClient(client, url, name=name, headers=headers) if url else None

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            name=self.name,
            families=(self.family,),
            default_source_type=self.source_type,
            supports_search=True,
            supports_fetch=bool(self._fetch_tool),
            returns_inline_documents=True,
            requires_api_key=False,
            notes="MCP server over HTTP; tools are discovered at first use",
            tags=frozenset({"mcp"}),
        )

    async def _tool(self, configured: str | None, candidates: Sequence[str]) -> McpTool | None:
        if self.mcp is None:
            raise SourceNotConfigured(
                f"{self.name} has no server address configured", provider=self.name
            )
        tools = await self.mcp.tools()
        by_name = {tool.name: tool for tool in tools}
        if configured:
            tool = by_name.get(configured)
            if tool is None:
                raise SourceNotConfigured(
                    f"{self.name} has no tool named {configured!r}; it offers "
                    + ", ".join(sorted(by_name) or ["nothing"]),
                    provider=self.name,
                )
            return tool
        for candidate in candidates:
            if candidate in by_name:
                return by_name[candidate]
        return None

    async def search(self, query: ResearchQuery) -> list[SearchHit]:
        tool = await self._tool(self._search_tool, SEARCH_TOOL_NAMES)
        if tool is None:
            raise SourceNotConfigured(
                f"{self.name} offers no tool that looks like a search; name one "
                "in configuration",
                provider=self.name,
            )
        blocks = await self.mcp.call_tool(
            tool.name,
            {
                tool.query_argument(): query.text,
                # Hints a provider may or may not honour, passed through
                # as the query gave them rather than translated.
                **{key: value for key, value in (query.filters or {}).items()},
            },
        )
        return _hits_from(blocks, provider=self.name, query=query, source_type=self.source_type)

    async def fetch(self, hit: SearchHit) -> EvidenceDocument:
        tool = await self._tool(self._fetch_tool, FETCH_TOOL_NAMES)
        if tool is None:
            raise SourceUnavailable(
                f"{self.name} offers no fetch tool; its search results carry "
                "whatever text they carry",
                provider=self.name,
            )
        blocks = await self.mcp.call_tool(tool.name, {"url": hit.url or hit.external_id or ""})
        text = "\n\n".join(_text_of(block) for block in blocks if _text_of(block))
        return build_document(
            provider=self.name,
            source_type=hit.source_type or self.source_type,
            source_family=self.family,
            provenance=Provenance(
                provider=self.name,
                retrieval_method=RetrievalMethod.DIRECT_FETCH,
                provider_endpoint=self.url,
                requested_url=hit.url,
            ),
            title=hit.title,
            text=clean_text(text),
            url=hit.url,
            external_id=hit.external_id,
        )


def _decode(body: str, *, provider: str) -> Any:
    """Read a JSON-RPC reply, whether it arrived plain or as an SSE stream."""
    text = (body or "").strip()
    if not text:
        return None
    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except ValueError as exc:
            raise SourceUnavailable(
                f"{provider} returned malformed JSON: {exc}", provider=provider
            ) from exc
    # text/event-stream: the payload is the last complete `data:` line.
    payloads = [
        line[len("data:") :].strip()
        for line in text.splitlines()
        if line.startswith("data:")
    ]
    for raw in reversed(payloads):
        try:
            return json.loads(raw)
        except ValueError:
            continue
    raise SourceUnavailable(f"{provider} returned no usable payload", provider=provider)


def _text_of(block: dict[str, Any]) -> str:
    if block.get("type") == "text":
        return str(block.get("text") or "")
    return ""


def _hits_from(
    blocks: Sequence[dict[str, Any]],
    *,
    provider: str,
    query: ResearchQuery,
    source_type: SourceType,
) -> list[SearchHit]:
    """Turn tool output into search hits.

    A server may answer with structured JSON or with prose. Structured
    results are read as results; prose is kept as one hit carrying the text,
    because discarding it would lose the answer and parsing it would invent
    structure that is not there.
    """
    hits: list[SearchHit] = []
    for block in blocks:
        text = _text_of(block)
        if not text:
            continue
        parsed = _maybe_json(text)
        entries = _entries(parsed)
        if entries:
            for entry in entries:
                hit = _hit_from_entry(entry, provider=provider, source_type=source_type)
                if hit is not None:
                    hits.append(hit)
            continue
        hits.append(
            SearchHit(
                provider=provider,
                title=query.text[:200],
                snippet=clean_text(text)[:2000],
                source_type=source_type,
                raw={"unstructured": True},
            )
        )
    return hits


def _maybe_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        return json.loads(stripped)
    except ValueError:
        return None


def _entries(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return [entry for entry in parsed if isinstance(entry, dict)]
    if isinstance(parsed, dict):
        for key in ("results", "items", "documents", "hits", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
        if any(key in parsed for key in ("url", "title", "link")):
            return [parsed]
    return []


def _hit_from_entry(
    entry: dict[str, Any], *, provider: str, source_type: SourceType
) -> SearchHit | None:
    url = entry.get("url") or entry.get("link") or entry.get("href")
    title = entry.get("title") or entry.get("name") or entry.get("headline")
    snippet = (
        entry.get("snippet")
        or entry.get("summary")
        or entry.get("description")
        or entry.get("content")
        or entry.get("text")
    )
    if not (url or title or snippet):
        return None
    return SearchHit(
        provider=provider,
        url=str(url) if url else None,
        title=normalize_whitespace(str(title))[:300] if title else None,
        snippet=clean_text(str(snippet))[:2000] if snippet else None,
        external_id=str(entry.get("id")) if entry.get("id") else None,
        source_type=source_type,
        raw={key: value for key, value in entry.items() if not key.startswith("_")},
    )

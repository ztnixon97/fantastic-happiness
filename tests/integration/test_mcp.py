"""MCP servers as research sources, against a server that answers in-process.

No subprocess is launched here for the same reason none is launched in the
package: the point of putting MCP behind the source interface is that what
crosses into this process is a network response like any other.
"""

from __future__ import annotations

import json

import httpx
import pytest

from research.config import AcquisitionPolicy, ResearchConfig
from research.errors import ConfigError, SourceNotConfigured, SourceUnavailable
from research.models.common import SourceFamily
from research.models.query import ResearchQuery, SearchHit
from research.sources.base import ResearchSource
from research.sources.http import SafeHttpClient
from research.sources.mcp import McpSource


def fake_server(
    *,
    tools: list[dict] | None = None,
    content: list[dict] | None = None,
    is_error: bool = False,
    sse: bool = False,
    error: dict | None = None,
):
    """An MCP server that answers JSON-RPC over HTTP, and records what it was asked."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        method = body.get("method")
        if error:
            result = {"jsonrpc": "2.0", "id": body["id"], "error": error}
        elif method == "initialize":
            result = {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"protocolVersion": "2025-06-18", "serverInfo": {"name": "fake"}},
            }
        elif method == "tools/list":
            result = {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {"tools": tools if tools is not None else [_search_tool()]},
            }
        elif method == "tools/call":
            result = {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "content": content if content is not None else [],
                    "isError": is_error,
                },
            }
        else:  # pragma: no cover - the client asks for nothing else
            result = {"jsonrpc": "2.0", "id": body["id"], "result": {}}

        if sse:
            return httpx.Response(
                200,
                text=f"event: message\ndata: {json.dumps(result)}\n\n",
                headers={"content-type": "text/event-stream", "mcp-session-id": "s1"},
            )
        return httpx.Response(200, json=result, headers={"mcp-session-id": "s1"})

    return handler, calls


def _search_tool(name: str = "search", argument: str = "query") -> dict:
    return {
        "name": name,
        "description": "Search the index",
        "inputSchema": {"type": "object", "properties": {argument: {"type": "string"}}},
    }


def source_for(handler, **kwargs) -> McpSource:
    client = SafeHttpClient(
        AcquisitionPolicy(per_host_min_interval_seconds=0.0),
        transport=httpx.MockTransport(handler),
    )
    kwargs.setdefault("url", "https://mcp.test/mcp")
    kwargs.setdefault("name", "fake_mcp")
    return McpSource(client, **kwargs)


RESULTS = [
    {
        "type": "text",
        "text": json.dumps(
            {
                "results": [
                    {
                        "title": "Reactor cost review",
                        "url": "https://example.test/review",
                        "snippet": "Median levelized cost of $120/MWh.",
                        "id": "r-1",
                    },
                    {
                        "title": "Construction delays",
                        "url": "https://example.test/delays",
                        "summary": "Eleven months behind schedule.",
                    },
                ]
            }
        ),
    }
]


class TestItIsJustASource:
    def test_it_satisfies_the_source_protocol(self) -> None:
        handler, _ = fake_server()
        assert isinstance(source_for(handler), ResearchSource)

    def test_the_action_vocabulary_does_not_grow(self) -> None:
        """An MCP server is a provider, not a new verb for the worker."""
        from research.agents.actions import ACTIONS

        assert not any("mcp" in name for name in ACTIONS)

    async def test_it_searches_and_returns_hits(self) -> None:
        handler, calls = fake_server(content=RESULTS)
        hits = await source_for(handler).search(ResearchQuery(text="reactor cost"))

        assert [hit.title for hit in hits] == ["Reactor cost review", "Construction delays"]
        assert hits[0].url == "https://example.test/review"
        assert hits[0].external_id == "r-1"
        assert [call["method"] for call in calls] == ["initialize", "tools/list", "tools/call"]

    async def test_the_query_goes_in_the_argument_the_schema_names(self) -> None:
        handler, calls = fake_server(
            tools=[_search_tool(argument="q")], content=RESULTS
        )
        await source_for(handler).search(ResearchQuery(text="reactor cost"))
        assert calls[-1]["params"]["arguments"] == {"q": "reactor cost"}

    async def test_tools_are_discovered_once(self) -> None:
        handler, calls = fake_server(content=RESULTS)
        source = source_for(handler)
        await source.search(ResearchQuery(text="one"))
        await source.search(ResearchQuery(text="two"))
        assert [call["method"] for call in calls].count("tools/list") == 1

    async def test_an_event_stream_reply_is_read(self) -> None:
        handler, _ = fake_server(content=RESULTS, sse=True)
        hits = await source_for(handler).search(ResearchQuery(text="reactor cost"))
        assert len(hits) == 2


class TestAwkwardServers:
    async def test_prose_is_kept_rather_than_parsed_into_structure(self) -> None:
        """Inventing fields a server did not send would be worse than none."""
        handler, _ = fake_server(
            content=[{"type": "text", "text": "The median estimate is $120 per MWh."}]
        )
        hits = await source_for(handler).search(ResearchQuery(text="reactor cost"))
        assert len(hits) == 1
        assert "median estimate" in hits[0].snippet
        assert hits[0].raw == {"unstructured": True}

    async def test_a_bare_list_of_results_is_read(self) -> None:
        handler, _ = fake_server(
            content=[{"type": "text", "text": json.dumps([{"title": "A", "url": "https://a.test"}])}]
        )
        hits = await source_for(handler).search(ResearchQuery(text="q"))
        assert [hit.title for hit in hits] == ["A"]

    async def test_non_text_blocks_are_ignored(self) -> None:
        handler, _ = fake_server(
            content=[{"type": "image", "data": "..."}, {"type": "text", "text": "kept"}]
        )
        hits = await source_for(handler).search(ResearchQuery(text="q"))
        assert len(hits) == 1

    async def test_a_server_with_no_search_tool_says_so(self) -> None:
        handler, _ = fake_server(tools=[{"name": "delete_everything", "inputSchema": {}}])
        with pytest.raises(SourceNotConfigured, match="no tool that looks like a search"):
            await source_for(handler).search(ResearchQuery(text="q"))

    async def test_naming_a_tool_that_does_not_exist_lists_what_does(self) -> None:
        handler, _ = fake_server(tools=[_search_tool("find_docs")])
        with pytest.raises(SourceNotConfigured, match="find_docs"):
            await source_for(handler, search_tool="nope").search(ResearchQuery(text="q"))

    async def test_a_tool_error_is_a_source_failure(self) -> None:
        handler, _ = fake_server(content=[], is_error=True)
        with pytest.raises(SourceUnavailable):
            await source_for(handler).search(ResearchQuery(text="q"))

    async def test_a_jsonrpc_error_is_reported_with_its_message(self) -> None:
        handler, _ = fake_server(error={"code": -32601, "message": "method not found"})
        with pytest.raises(SourceUnavailable, match="method not found"):
            await source_for(handler).search(ResearchQuery(text="q"))

    async def test_fetching_without_a_fetch_tool_says_so(self) -> None:
        handler, _ = fake_server()
        with pytest.raises(SourceUnavailable, match="no fetch tool"):
            await source_for(handler).fetch(SearchHit(provider="fake_mcp", url="https://a.test"))

    async def test_a_fetch_tool_produces_a_document(self) -> None:
        handler, _ = fake_server(
            tools=[_search_tool(), {"name": "fetch", "inputSchema": {}}],
            content=[{"type": "text", "text": "The full text of the record."}],
        )
        document = await source_for(handler, fetch_tool="fetch").fetch(
            SearchHit(provider="fake_mcp", url="https://a.test/record", title="Record")
        )
        assert document.id == "", "identity belongs to the store"
        assert "full text of the record" in document.text


class TestConfiguration:
    def test_a_server_is_an_address_not_a_command(self) -> None:
        """This system will not start an MCP server, only talk to one."""
        from research.config import McpServer

        with pytest.raises(ConfigError, match="will not start one"):
            McpServer.from_dict({"name": "local", "command": "npx some-server"})

    def test_configured_servers_become_sources(self) -> None:
        from research.config import McpServer
        from research.sources.registry import build_registry

        config = ResearchConfig(
            mcp_servers=[McpServer(url="https://mcp.test/mcp", name="house_index")]
        )
        registry = build_registry(config, SafeHttpClient(config.acquisition))
        assert "house_index" in {row["name"] for row in registry.describe()}

    def test_a_disabled_server_is_not_registered(self) -> None:
        from research.config import McpServer
        from research.sources.registry import build_registry

        config = ResearchConfig(
            mcp_servers=[
                McpServer(url="https://mcp.test/mcp", name="off", enabled=False)
            ]
        )
        registry = build_registry(config, SafeHttpClient(config.acquisition))
        assert "off" not in {row["name"] for row in registry.describe()}

    def test_a_server_serves_the_family_it_was_configured_for(self) -> None:
        handler, _ = fake_server()
        source = source_for(handler, family=SourceFamily.ACADEMIC)
        assert source.capabilities().serves(SourceFamily.ACADEMIC)
        assert not source.capabilities().serves(SourceFamily.NEWS)

"""The read-only investigation UI.

The server is exercised through its route table, and a live socket is used
for the properties that only exist at the HTTP level: methods, headers and
static assets.
"""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from research.acquisition.pipeline import EvidenceAcquirer
from research.models.claim import EvidenceStance
from research.models.common import Provenance, SourceFamily, SourceType
from research.models.investigation import InvestigationStatus, StopReason
from research.normalize.document import build_document
from research.operations.claims import ClaimOperations
from research.ui.server import UiServer, serve

ASSETS = Path(__file__).resolve().parents[2] / "research" / "ui" / "assets"

WIRE = (
    "WASHINGTON (Reuters) - The utility group said on Wednesday it had agreed to "
    "terminate the flagship small modular reactor project after subscription levels "
    "fell short of what was needed to proceed."
)


@pytest.fixture
def populated(store):
    investigation = store.investigations.create("Are SMRs competitive for data centres?")
    acquirer = EvidenceAcquirer(store)

    def add(title, text, url, source_type, family):
        return acquirer.persist(
            build_document(
                provider="test",
                source_type=source_type,
                source_family=family,
                provenance=Provenance(provider="test"),
                title=title,
                text=text,
                url=url,
            ),
            investigation_id=investigation.id,
        ).document

    original = add(
        "Project terminated", WIRE, "https://www.reuters.test/a",
        SourceType.ORIGINAL_NEWS_REPORTING, SourceFamily.NEWS,
    )
    copy = add(
        "Project terminated", WIRE, "https://www.gazette.test/a",
        SourceType.SECONDARY_NEWS_REPORTING, SourceFamily.NEWS,
    )
    paper = add(
        "Historical cost escalation in nuclear construction",
        "Realized costs exceeded initial estimates by a median of 117 percent.",
        "https://doi.org/10.1016/j.joule.2021.06.004",
        SourceType.ACADEMIC_PEER_REVIEWED, SourceFamily.ACADEMIC,
    )
    operations = ClaimOperations(store, investigation_id=investigation.id)
    claim = operations.create_claim("The flagship project was terminated")
    operations.link_evidence(
        claim.id, original.id, EvidenceStance.SUPPORTS,
        excerpt="agreed to terminate the flagship small modular reactor project",
    )
    operations.link_evidence(claim.id, copy.id, EvidenceStance.SUPPORTS)
    store.investigations.set_status(
        investigation.id,
        InvestigationStatus.COMPLETED,
        stop_reason=StopReason.NO_OPEN_TASKS,
        stop_detail="nothing left to run",
    )
    return store, investigation, {"original": original, "copy": copy, "paper": paper,
                                  "claim": claim}


@pytest.fixture
def ui(populated) -> UiServer:
    store, _, _ = populated
    return UiServer(store)


def get(ui: UiServer, path: str) -> tuple[int, dict]:
    return ui.dispatch(path)


class TestRoutes:
    def test_investigations_are_listed(self, ui, populated) -> None:
        _, investigation, _ = populated
        status, payload = get(ui, "/api/investigations")
        assert status == 200
        assert payload["investigations"][0]["id"] == investigation.id
        assert payload["investigations"][0]["documents"] == 3

    def test_overview_carries_counts_budget_and_stopping_rules(self, ui, populated) -> None:
        _, investigation, _ = populated
        status, payload = get(ui, f"/api/investigations/{investigation.id}")
        assert status == 200
        assert payload["counts"]["documents"] == 3
        assert payload["counts"]["independent_sources"] == 2
        assert payload["stop_reason"] == "no_open_tasks"
        assert "searches" in payload["budget"]
        assert set(payload["stopping_rules"]) >= {"budget", "runtime", "open_tasks"}

    def test_claims_carry_their_links_and_excerpts(self, ui, populated) -> None:
        _, investigation, fixtures = populated
        _, payload = get(ui, f"/api/investigations/{investigation.id}/claims")
        claim = payload["claims"][0]
        assert claim["claim_id"] == fixtures["claim"].id
        assert claim["support"]["independent_sources"] == 1
        assert any(link["excerpt"] for link in claim["links"])

    def test_the_graph_folds_copies_into_one_node(self, ui, populated) -> None:
        _, investigation, fixtures = populated
        _, payload = get(ui, f"/api/investigations/{investigation.id}/graph")
        evidence = [node for node in payload["nodes"] if node["kind"] == "evidence"]
        assert len(evidence) == 2, "three documents, two independent sources"
        wire = next(node for node in evidence if node["id"] == fixtures["original"].id)
        assert wire["copies"] == [fixtures["copy"].id]
        # Both documents support the claim, but they are one source, so the
        # graph draws one edge from the node that represents them.
        edges = payload["edges"]
        assert all(edge["source"] != fixtures["copy"].id for edge in edges)
        supports = [edge for edge in edges if edge["kind"] == "supports"]
        assert len(supports) == 1
        assert supports[0]["source"] == fixtures["original"].id

    def test_an_edge_from_a_copy_alone_is_attributed_to_the_original(
        self, ui, populated
    ) -> None:
        store, investigation, fixtures = populated
        operations = ClaimOperations(store, investigation_id=investigation.id)
        claim = operations.create_claim("The termination was widely reported")
        operations.link_evidence(claim.id, fixtures["copy"].id, EvidenceStance.SUPPORTS)
        _, payload = get(ui, f"/api/investigations/{investigation.id}/graph")
        edge = next(
            edge for edge in payload["edges"] if edge["target"] == claim.id
        )
        assert edge["source"] == fixtures["original"].id
        assert edge["via_copy"] is True

    def test_documents_are_returned_as_text(self, ui, populated) -> None:
        _, _, fixtures = populated
        status, payload = get(ui, f"/api/documents/{fixtures['original'].id}")
        assert status == 200
        assert "terminate the flagship" in payload["text"]
        assert payload["content_hash"].startswith("sha256:")
        assert payload["provenance"]

    def test_a_copy_says_what_it_is_a_copy_of(self, ui, populated) -> None:
        _, _, fixtures = populated
        _, payload = get(ui, f"/api/documents/{fixtures['copy'].id}")
        assert payload["duplicate_of"] == fixtures["original"].id
        assert payload["duplicate_relation"] == "syndicated_copy"
        assert payload["duplicate_reason"]

    @pytest.mark.parametrize(
        "suffix", ["/tasks", "/claims", "/questions", "/graph", "/entities",
                   "/timeline", "/sources", "/activity"],
    )
    def test_every_view_answers(self, ui, populated, suffix: str) -> None:
        _, investigation, _ = populated
        status, payload = get(ui, f"/api/investigations/{investigation.id}{suffix}")
        assert status == 200
        assert isinstance(payload, dict) and payload

    def test_malformed_identifiers_are_refused_before_the_store(self, ui) -> None:
        for path in (
            "/api/investigations/../../etc/passwd",
            "/api/investigations/nonsense",
            "/api/documents/evidence:1;DROP TABLE documents",
        ):
            status, payload = get(ui, path)
            assert status in (400, 404)
            assert "error" in payload

    def test_unknown_investigations_are_a_clean_404(self, ui) -> None:
        status, payload = get(ui, "/api/investigations/investigation:999")
        assert status == 404
        assert "no such investigation" in payload["error"]

    def test_unknown_endpoints_are_refused(self, ui) -> None:
        status, _ = get(ui, "/api/secrets")
        assert status == 404


class TestHttpSurface:
    @pytest.fixture
    def server(self, populated):
        store, investigation, fixtures = populated
        httpd = serve(store, host="127.0.0.1", port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield httpd.server_address[1], investigation, fixtures
        httpd.shutdown()
        httpd.server_close()

    def request(self, port: int, path: str, method: str = "GET"):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(method, path)
        response = connection.getresponse()
        body = response.read()
        connection.close()
        return response.status, dict(response.getheaders()), body

    def test_the_page_and_its_assets_are_served(self, server) -> None:
        port, _, _ = server
        for path, content_type in (
            ("/", "text/html"), ("/app.js", "text/javascript"), ("/app.css", "text/css")
        ):
            status, headers, body = self.request(port, path)
            assert status == 200
            assert headers["Content-Type"].startswith(content_type)
            assert body

    def test_every_response_carries_the_security_headers(self, server) -> None:
        port, _, _ = server
        _, headers, _ = self.request(port, "/")
        assert "default-src 'none'" in headers["Content-Security-Policy"]
        assert "script-src 'self'" in headers["Content-Security-Policy"]
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Referrer-Policy"] == "no-referrer"

    @pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
    def test_writes_are_refused(self, server, method: str) -> None:
        port, investigation, _ = server
        status, _, body = self.request(port, f"/api/investigations/{investigation.id}", method)
        assert status == 405
        assert b"read-only" in body

    def test_api_responses_are_json(self, server) -> None:
        port, investigation, _ = server
        status, headers, body = self.request(port, f"/api/investigations/{investigation.id}")
        assert status == 200
        assert headers["Content-Type"].startswith("application/json")
        assert json.loads(body)["id"] == investigation.id

    def test_the_favicon_is_answered_rather_than_404ing(self, server) -> None:
        port, _, _ = server
        status, _, _ = self.request(port, "/favicon.ico")
        assert status == 204

    def test_unknown_paths_do_not_reach_the_filesystem(self, server) -> None:
        port, _, _ = server
        for path in ("/server.py", "/../server.py", "/assets/app.js", "/etc/passwd"):
            status, _, _ = self.request(port, path)
            assert status == 404


class TestClientSafety:
    def test_the_client_never_assigns_markup(self) -> None:
        """Retrieved content is rendered as text, and stays that way."""
        source = (ASSETS / "app.js").read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("//")
        )
        for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
            assert forbidden not in code, f"{forbidden} would turn evidence into markup"

    def test_the_page_loads_no_remote_resources(self) -> None:
        html = (ASSETS / "app.html").read_text(encoding="utf-8")
        assert "http://" not in html and "https://" not in html

    def test_the_client_uses_no_inline_styles(self) -> None:
        """The page's own CSP forbids them, so using one is a silent failure."""
        source = (ASSETS / "app.js").read_text(encoding="utf-8")
        assert "style:" not in source
        assert 'setAttribute("style"' not in source

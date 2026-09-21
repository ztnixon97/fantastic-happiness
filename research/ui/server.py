"""A small read-only server for the investigation UI.

Read-only in the strongest sense available: only GET is answered, the route
table is fixed, and every handler is a projection of stored state. Nothing
the UI can reach performs research, spends budget or writes to the store.

It binds to the loopback interface by default. The content it serves includes
text retrieved from the open web, so the page is sent under a content policy
that forbids remote script and the client renders all of it as text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from research.config import ResearchConfig
from research.errors import NotFound
from research.storage.store import ResearchStore
from research.ui.api import InvestigationView, investigations

ASSETS = Path(__file__).resolve().parent / "assets"

#: Identifiers only ever look like this. Anything else is refused before it
#: reaches the store.
_ID_RE = re.compile(r"^[a-z_]+:\d+$")

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


@dataclass(slots=True)
class Route:
    pattern: re.Pattern[str]
    handler: Callable[..., dict[str, Any]]


class UiServer:
    """Wires the read models to a fixed route table."""

    def __init__(self, store: ResearchStore, *, config: ResearchConfig | None = None) -> None:
        self.store = store
        self.config = config or ResearchConfig()
        self.routes = [
            Route(re.compile(r"^/api/investigations$"), self._investigations),
            Route(re.compile(r"^/api/investigations/(?P<id>[^/]+)$"), self._overview),
            Route(re.compile(r"^/api/investigations/(?P<id>[^/]+)/tasks$"), self._tasks),
            Route(re.compile(r"^/api/investigations/(?P<id>[^/]+)/claims$"), self._claims),
            Route(re.compile(r"^/api/investigations/(?P<id>[^/]+)/questions$"), self._questions),
            Route(re.compile(r"^/api/investigations/(?P<id>[^/]+)/graph$"), self._graph),
            Route(re.compile(r"^/api/investigations/(?P<id>[^/]+)/entities$"), self._entities),
            Route(re.compile(r"^/api/investigations/(?P<id>[^/]+)/timeline$"), self._timeline),
            Route(re.compile(r"^/api/investigations/(?P<id>[^/]+)/sources$"), self._sources),
            Route(re.compile(r"^/api/investigations/(?P<id>[^/]+)/activity$"), self._activity),
            Route(re.compile(r"^/api/documents/(?P<id>[^/]+)$"), self._document),
        ]

    # -- routing --------------------------------------------------------
    def dispatch(self, path: str) -> tuple[int, dict[str, Any]]:
        for route in self.routes:
            match = route.pattern.match(path)
            if not match:
                continue
            arguments = {key: unquote(value) for key, value in match.groupdict().items()}
            for value in arguments.values():
                if not _ID_RE.match(value):
                    return 400, {"error": f"malformed identifier: {value!r}"}
            try:
                return 200, route.handler(**arguments)
            except NotFound as exc:
                return 404, {"error": str(exc)}
        return 404, {"error": f"no such endpoint: {path}"}

    def _view(self, investigation_id: str) -> InvestigationView:
        return InvestigationView(self.store, investigation_id, config=self.config)

    def _investigations(self) -> dict[str, Any]:
        return investigations(self.store)

    def _overview(self, id: str) -> dict[str, Any]:
        return self._view(id).overview()

    def _tasks(self, id: str) -> dict[str, Any]:
        return self._view(id).tasks()

    def _claims(self, id: str) -> dict[str, Any]:
        return self._view(id).claims()

    def _questions(self, id: str) -> dict[str, Any]:
        return self._view(id).open_questions()

    def _graph(self, id: str) -> dict[str, Any]:
        return self._view(id).graph()

    def _entities(self, id: str) -> dict[str, Any]:
        return self._view(id).entities()

    def _timeline(self, id: str) -> dict[str, Any]:
        return self._view(id).timeline()

    def _sources(self, id: str) -> dict[str, Any]:
        return self._view(id).sources()

    def _activity(self, id: str) -> dict[str, Any]:
        return self._view(id).activity()

    def _document(self, id: str) -> dict[str, Any]:
        # The document's own investigation supplies the view; a document is
        # only ever read through the investigation that holds it.
        document = self.store.documents.get(id)
        return self._view(document.investigation_id or "").document(id)


def _asset(name: str) -> tuple[str, bytes] | None:
    """Serve one of the three static files, by exact name only."""
    content_types = {"app.html": "text/html", "app.js": "text/javascript", "app.css": "text/css"}
    if name not in content_types:
        return None
    path = ASSETS / name
    if not path.is_file():  # pragma: no cover - packaging error
        return None
    return f"{content_types[name]}; charset=utf-8", path.read_bytes()


def make_handler(server: UiServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "research-ui"

        def do_GET(self) -> None:  # noqa: N802 - http.server's interface
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send_asset("app.html")
                return
            if path.startswith("/api/"):
                status, payload = server.dispatch(path)
                self._send_json(status, payload)
                return
            asset = path.lstrip("/")
            if asset in ("app.js", "app.css"):
                self._send_asset(asset)
                return
            if asset == "favicon.ico":
                # Every browser asks; answering "nothing here" beats a 404 in
                # the console on every page load.
                self._respond(204, "image/x-icon", b"")
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            # The UI is an inspection surface. Research runs from the CLI.
            self._send_json(405, {"error": "this interface is read-only"})

        do_PUT = do_POST
        do_DELETE = do_POST
        do_PATCH = do_POST

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, default=str, ensure_ascii=False).encode("utf-8")
            self._respond(status, "application/json; charset=utf-8", body)

        def _send_asset(self, name: str) -> None:
            asset = _asset(name)
            if asset is None:
                self._send_json(404, {"error": "not found"})
                return
            content_type, body = asset
            self._respond(200, content_type, body)

        def _respond(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for header, value in SECURITY_HEADERS.items():
                self.send_header(header, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            # Quiet by default; the CLI prints what matters.
            return

    return Handler


def serve(
    store: ResearchStore,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    config: ResearchConfig | None = None,
) -> ThreadingHTTPServer:
    """Create the server. The caller decides when to serve and when to stop."""
    handler = make_handler(UiServer(store, config=config))
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd

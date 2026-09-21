"""A deliberately small, deliberately suspicious HTTP client.

Every rule here is enforced in code rather than requested in a prompt:

* only ``http``/``https``,
* no loopback, private, link-local or otherwise internal destinations,
* redirects followed one hop at a time and re-validated each time,
* response bodies capped before they are read into memory,
* content types allowlisted rather than sniffed,
* retries bounded, with ``Retry-After`` honoured,
* per-host pacing so an investigation is not mistaken for an attack.

The client has no access to the host environment: credentials are passed in
explicitly by the adapter that owns them.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

from research.config import AcquisitionPolicy
from research.errors import SourceRejected, SourceUnavailable, UnsafeRequest

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(slots=True)
class HttpResponse:
    url: str
    final_url: str
    status_code: int
    headers: Mapping[str, str]
    content: bytes
    content_type: str | None
    elapsed_ms: int
    truncated: bool = False

    @property
    def text(self) -> str:
        charset = "utf-8"
        if self.content_type and "charset=" in self.content_type:
            charset = self.content_type.split("charset=", 1)[1].split(";")[0].strip()
        try:
            return self.content.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):  # pragma: no cover - exotic charsets
            return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        import json

        try:
            return json.loads(self.text)
        except ValueError as exc:
            raise SourceUnavailable(
                f"malformed JSON from {self.final_url}: {exc}", provider="http"
            ) from exc


def _is_internal_address(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


class HostPacer:
    """Per-host minimum interval between requests."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def wait(self, host: str) -> None:
        if self.min_interval <= 0:
            return
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            last = self._last.get(host)
            now = time.monotonic()
            if last is not None:
                delay = self.min_interval - (now - last)
                if delay > 0:
                    await asyncio.sleep(delay)
            self._last[host] = time.monotonic()


class SafeHttpClient:
    """The only component in the system that makes outbound requests."""

    def __init__(
        self,
        policy: AcquisitionPolicy | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self.policy = policy or AcquisitionPolicy()
        self._sleep = sleep
        self._pacer = HostPacer(self.policy.per_host_min_interval_seconds)
        self._semaphore = asyncio.Semaphore(self.policy.max_concurrent_requests)
        # Redirects are followed manually so each hop can be re-validated.
        self._client = httpx.AsyncClient(
            transport=transport,
            follow_redirects=False,
            timeout=httpx.Timeout(
                self.policy.request_timeout_seconds,
                connect=self.policy.connect_timeout_seconds,
            ),
            headers={"User-Agent": self.policy.user_agent},
        )
        #: Set when a mock transport is supplied: DNS checks are meaningless
        #: then, and the transport is the only reachable destination anyway.
        self._mocked = transport is not None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "SafeHttpClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- safety ---------------------------------------------------------
    async def _validate(self, url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            raise UnsafeRequest(f"refusing non-http(s) URL: {url}")
        host = parts.hostname
        if not host:
            raise UnsafeRequest(f"refusing URL without a host: {url}")
        if self.policy.allow_private_hosts or self._mocked:
            return host
        if host in ("localhost", "localhost.localdomain") or host.endswith(".local"):
            raise UnsafeRequest(f"refusing internal host: {host}")
        if _is_internal_address(host):
            raise UnsafeRequest(f"refusing internal address: {host}")

        addresses = await self._resolve(host)
        if addresses is None:
            # Resolution is unavailable (for example behind an egress proxy
            # that resolves on our behalf). Literal-address and name checks
            # above still applied; the proxy enforces the rest.
            if os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"):
                return host
            raise UnsafeRequest(f"cannot resolve host, refusing: {host}")
        for address in addresses:
            if _is_internal_address(address):
                raise UnsafeRequest(
                    f"host {host} resolves to an internal address ({address})"
                )
        return host

    async def _resolve(self, host: str) -> list[str] | None:
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(
                host, None, proto=socket.IPPROTO_TCP
            )
        except (socket.gaierror, OSError):
            return None
        return [info[4][0] for info in infos]

    def _check_content_type(self, content_type: str | None, url: str) -> None:
        if not content_type:
            return
        base = content_type.split(";", 1)[0].strip().lower()
        if base not in self.policy.allowed_content_types:
            raise SourceRejected(
                f"unsupported content type {base!r} at {url}", provider="http"
            )

    # -- requests -------------------------------------------------------
    async def request(
        self,
        method: str,
        url: str,
        *,
        provider: str = "http",
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json_body: Any = None,
        accept: str | None = None,
    ) -> HttpResponse:
        """Perform one logical request, including redirects and retries."""
        started = time.monotonic()
        current_url = url
        request_headers = dict(headers or {})
        if accept:
            request_headers.setdefault("Accept", accept)

        for hop in range(self.policy.max_redirects + 1):
            host = await self._validate(current_url)
            response = await self._send_with_retries(
                method, current_url, host, provider, params, request_headers, json_body
            )
            if response.is_redirect and response.headers.get("location"):
                location = str(
                    httpx.URL(current_url).join(response.headers["location"])
                )
                await response.aclose()
                if hop >= self.policy.max_redirects:
                    raise SourceUnavailable(
                        f"too many redirects from {url}", provider=provider
                    )
                current_url = location
                params = None  # query already encoded into the redirect target
                continue

            content_type = response.headers.get("content-type")
            if response.status_code >= 400:
                await response.aclose()
                message = f"{provider} returned HTTP {response.status_code} for {current_url}"
                if response.status_code in RETRYABLE_STATUS:
                    raise SourceUnavailable(
                        message, provider=provider, status_code=response.status_code
                    )
                raise SourceRejected(
                    message, provider=provider, status_code=response.status_code
                )

            self._check_content_type(content_type, current_url)
            content, truncated = await self._read_capped(response)
            await response.aclose()
            return HttpResponse(
                url=url,
                final_url=current_url,
                status_code=response.status_code,
                headers=dict(response.headers),
                content=content,
                content_type=content_type,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                truncated=truncated,
            )

        raise SourceUnavailable(f"redirect loop for {url}", provider=provider)

    async def _send_with_retries(
        self,
        method: str,
        url: str,
        host: str,
        provider: str,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str],
        json_body: Any,
    ) -> httpx.Response:
        attempt = 0
        while True:
            await self._pacer.wait(host)
            try:
                async with self._semaphore:
                    request = self._client.build_request(
                        method, url, params=params, headers=headers, json=json_body
                    )
                    response = await self._client.send(request, stream=True)
            except httpx.HTTPError as exc:
                if attempt >= self.policy.max_retries:
                    raise SourceUnavailable(
                        f"{provider} request failed: {exc}", provider=provider
                    ) from exc
                await self._backoff(attempt)
                attempt += 1
                continue

            if response.status_code in RETRYABLE_STATUS and attempt < self.policy.max_retries:
                retry_after = self._retry_after(response)
                status = response.status_code
                await response.aclose()
                if (
                    retry_after is not None
                    and retry_after > self.policy.max_retry_after_seconds
                ):
                    raise SourceUnavailable(
                        f"{provider} asked to be retried in {retry_after:.0f}s, which is "
                        "longer than this run will wait; treating it as unavailable",
                        provider=provider,
                        status_code=status,
                    )
                await self._backoff(attempt, retry_after)
                attempt += 1
                continue
            return response

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        raw = response.headers.get("retry-after")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    async def _backoff(self, attempt: int, retry_after: float | None = None) -> None:
        delay = retry_after if retry_after is not None else (
            self.policy.retry_backoff_seconds * (2**attempt)
        )
        await self._sleep(min(delay, 30.0))

    async def _read_capped(self, response: httpx.Response) -> tuple[bytes, bool]:
        """Read the body, stopping at the configured ceiling.

        Streaming with a cap means a hostile or merely enormous response
        cannot exhaust memory before anyone looks at it.
        """
        limit = self.policy.max_response_bytes
        chunks: list[bytes] = []
        total = 0
        truncated = False
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
            total += len(chunk)
            if total >= limit:
                truncated = True
                break
        return b"".join(chunks)[:limit], truncated

    async def get_json(
        self,
        url: str,
        *,
        provider: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        response = await self.request(
            "GET",
            url,
            provider=provider,
            params=params,
            headers=headers,
            accept="application/json",
        )
        return response.json()

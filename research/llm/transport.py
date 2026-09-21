"""HTTP for model APIs.

Deliberately separate from :class:`research.sources.http.SafeHttpClient`.
That client exists to fetch *untrusted* material discovered during research,
so it refuses private addresses and sniffs nothing. A model endpoint is the
opposite case: it is configured by the operator, it is where a local or
self-hosted model legitimately lives on a private address, and it is trusted
to the extent that the operator trusts their own configuration.

Mixing the two policies would either block local models or weaken the
research fetcher. They stay apart.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from research.errors import ModelError

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})


@dataclass(slots=True)
class TransportPolicy:
    timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 15.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0


class ModelTransport:
    """A small JSON-over-HTTPS client with bounded retries."""

    def __init__(
        self,
        policy: TransportPolicy | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self.policy = policy or TransportPolicy()
        self._sleep = sleep
        self._client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(
                self.policy.timeout_seconds, connect=self.policy.connect_timeout_seconds
            ),
        )

    async def post_json(
        self,
        url: str,
        *,
        provider: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                response = await self._client.post(
                    url, json=dict(payload), headers=dict(headers or {})
                )
            except httpx.HTTPError as exc:
                if attempt >= self.policy.max_retries:
                    raise ModelError(
                        f"{provider} request failed: {exc}", provider=provider, retryable=True
                    ) from exc
                await self._backoff(attempt)
                attempt += 1
                continue

            if response.status_code in RETRYABLE_STATUS and attempt < self.policy.max_retries:
                await self._backoff(attempt, _retry_after(response))
                attempt += 1
                continue
            if response.status_code >= 400:
                raise ModelError(
                    f"{provider} returned HTTP {response.status_code}: "
                    f"{response.text[:300]}",
                    provider=provider,
                    retryable=response.status_code in RETRYABLE_STATUS,
                )
            try:
                return response.json()
            except ValueError as exc:
                raise ModelError(
                    f"{provider} returned a non-JSON body", provider=provider
                ) from exc

    async def _backoff(self, attempt: int, retry_after: float | None = None) -> None:
        delay = (
            retry_after
            if retry_after is not None
            else self.policy.retry_backoff_seconds * (2**attempt)
        )
        await self._sleep(min(delay, 60.0))

    async def aclose(self) -> None:
        await self._client.aclose()


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None

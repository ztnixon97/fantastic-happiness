"""URL canonicalisation.

Two jobs, deliberately separated:

``canonicalize_url``
    Produce a clean URL that is still *fetchable*. Tracking parameters are
    removed, the fragment is dropped, the host is lowercased and IDNA-encoded.

``url_identity_key``
    Produce a key for *comparison only*. This is more aggressive: it drops the
    scheme, ``www.``/``m.`` prefixes, AMP suffixes and trailing slashes. Two
    URLs with the same identity key point at the same document; the key is
    never used to make a request.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

#: Parameters that never change which document is returned.
TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_name", "utm_reader", "utm_brand", "utm_social",
        "utm_social-type", "utm_swu", "utm_cid",
        "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "twclid", "igshid",
        "mc_cid", "mc_eid", "yclid", "_hsenc", "_hsmi", "hsctatracking",
        "ref", "ref_src", "ref_url", "referrer", "source", "src",
        "cmpid", "cmp", "ncid", "ito", "at_medium", "at_campaign",
        "sh", "spm", "share", "shared", "smid", "smtyp", "partner",
        "__twitter_impression", "guccounter", "guce_referrer",
        "guce_referrer_sig", "amp", "outputType", "s_kwcid",
    }
)

#: Hosts whose URLs wrap a real destination in a query parameter.
REDIRECT_HOSTS = {
    "news.google.com": ("url",),
    "www.google.com": ("url", "q"),
    "google.com": ("url", "q"),
    "l.facebook.com": ("u",),
    "out.reddit.com": ("url",),
    "www.bing.com": ("u",),
}

SAFE_SCHEMES = frozenset({"http", "https"})

_HOST_PREFIXES = ("www.", "m.", "mobile.", "amp.")
_AMP_SUFFIXES = ("/amp", "/amp/", "/amp.html", ".amp", "/amp-story")


def _idna(host: str) -> str:
    try:
        return host.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeDecodeError):
        return host


def canonicalize_url(url: str | None) -> str | None:
    """Return a cleaned, still-fetchable URL, or ``None`` if unusable.

    Relative URLs are the caller's problem: resolve them against their base
    before calling, since only the caller knows what that base was.
    """
    if not url:
        return None
    url = url.strip()
    if not url or url.startswith(("javascript:", "data:", "mailto:", "#")):
        return None

    if "://" not in url:
        url = "https://" + url.lstrip("/")

    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in SAFE_SCHEMES:
        return None

    host = _idna(parts.hostname.lower()) if parts.hostname else ""
    if not host:
        return None

    # Unwrap known redirector URLs so the real destination is what we store.
    params = parse_qsl(parts.query, keep_blank_values=False)
    for param_name in REDIRECT_HOSTS.get(host, ()):  # pragma: no branch
        for key, value in params:
            if key == param_name and "://" in unquote(value):
                return canonicalize_url(unquote(value))

    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    else:
        netloc = host

    kept = [
        (key, value)
        for key, value in params
        if key.lower() not in TRACKING_PARAMS
    ]
    kept.sort()
    query = urlencode(kept)

    path = quote(unquote(parts.path), safe="/%:@!$&'()*+,;=~-._")
    if path.endswith("/") and path != "/":
        path = path.rstrip("/")
    if not path:
        path = "/"

    return urlunsplit((scheme, netloc, path, query, ""))


def url_host(url: str | None) -> str | None:
    """Lowercased hostname of a URL, without ``www.``."""
    if not url:
        return None
    parts = urlsplit(url if "://" in url else "https://" + url)
    if not parts.hostname:
        return None
    host = parts.hostname.lower()
    for prefix in _HOST_PREFIXES:
        if host.startswith(prefix) and host.count(".") > 1:
            host = host[len(prefix):]
            break
    return host or None


#: Suffixes that need three labels to identify an organisation.
_MULTI_LABEL_SUFFIXES = (
    ".co.uk", ".ac.uk", ".gov.uk", ".org.uk", ".co.jp", ".com.au", ".co.nz",
    ".com.br", ".co.in", ".com.cn", ".gov.au", ".ac.jp", ".org.au",
)


def registrable_domain(url_or_host: str | None) -> str | None:
    """Approximate registrable domain, used to reason about outlet identity.

    This is a deliberately small heuristic rather than a bundled public-suffix
    list: it decides whether two copies of an article sit on the *same* site,
    and an occasional miss degrades to 'treated as different outlets', which
    is the conservative direction for independence counting.
    """
    host = url_host(url_or_host) if url_or_host and "/" in (url_or_host or "") else url_or_host
    host = url_host(host) if host else None
    if not host:
        return None
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    for suffix in _MULTI_LABEL_SUFFIXES:
        if host.endswith(suffix):
            return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def url_identity_key(url: str | None) -> str | None:
    """Aggressively normalised key for comparing two URLs. Never fetched."""
    canonical = canonicalize_url(url)
    if not canonical:
        return None
    parts = urlsplit(canonical)
    host = url_host(canonical) or ""
    path = parts.path
    for suffix in _AMP_SUFFIXES:
        if path.endswith(suffix) and len(path) > len(suffix):
            path = path[: -len(suffix)]
            break
    path = path.rstrip("/")
    if path.endswith("index.html"):
        path = path[: -len("index.html")].rstrip("/")
    params = [
        (key, value)
        for key, value in parse_qsl(parts.query)
        if key.lower() not in {"amp", "outputtype", "page"}
    ]
    params.sort()
    query = urlencode(params)
    key = f"{host}{path}"
    if query:
        key = f"{key}?{query}"
    return key.lower()

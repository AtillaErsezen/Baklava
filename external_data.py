"""Find, fetch, and join public datasets for augmentation.

URLs reaching this module come from an LLM reading web search results, so
treat every URL as hostile: safe_fetch is the only network path to data and
it enforces https, public-address-only DNS, pinned connections, capped
bodies, and non-executable parsers. Any benefit from augmentation must be
measured elsewhere; nothing here claims it.
"""

from __future__ import annotations

import gzip
import http.client
import io
import ipaddress
import os
import re
import socket
import ssl
import time
from urllib.parse import urljoin, urlsplit

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tavily import TavilyClient

SEARCH_DOMAINS = [
    "openml.org", "huggingface.co", "archive.ics.uci.edu", "data.gov", "catalog.data.gov",
    "github.com", "raw.githubusercontent.com", "zenodo.org", "kaggle.com",
]
AUTH_ONLY_DOMAINS = ("kaggle.com",)
DATA_EXTS = (".csv.gz", ".csv", ".parquet", ".tsv")
BLOCKED_NAMES = ("localhost", "metadata", "metadata.google.internal")
BLOCKED_SUFFIXES = (".localhost", ".internal", ".local", ".localdomain", ".home.arpa")
REDIRECT_CODES = (301, 302, 303, 307, 308)
MAX_REDIRECTS = 3
MAX_LINKS = 10
SNIPPET_CHARS = 200
CHUNK = 1 << 20
DECOMPRESS_FACTOR = 4  # decompressed size cap = DECOMPRESS_FACTOR * max_bytes
MIN_MATCH_RATE = 0.30
MIN_COVERAGE = 0.80
LEAK_THRESHOLD = 0.95
LEAK_BINS = 10
_LINK_RE = re.compile(r"https://[^\s\"'<>()\[\]{}|\\^`,]+")


class FetchError(ValueError):
    """Raised when a URL or its response violates a fetch safety rule."""


# ---------- search ----------

def _client() -> TavilyClient | None:
    key = os.environ.get("TAVILY_API_KEY")
    return TavilyClient(api_key=key) if key else None


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().rstrip(".")


def _on_domain(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def search(query: str, purpose: str, max_results: int = 8) -> list[dict] | dict:
    """Tavily search restricted to dataset hosts; returns compact hits or {"error": ...}."""
    client = _client()
    if client is None:
        return {"error": "TAVILY_API_KEY is not set; external dataset search is unavailable."}
    try:
        res = client.search(query, max_results=max_results, include_domains=SEARCH_DOMAINS)
    except Exception as e:  # network or API failure is reported, not raised
        return {"error": f"Tavily search failed: {type(e).__name__}: {e}"}
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": (r.get("content") or "")[:SNIPPET_CHARS],
            "downloadable": not _on_domain(_host(r.get("url", "")), AUTH_ONLY_DOMAINS),
            "purpose": purpose,
        }
        for r in res.get("results", [])
    ]


# ---------- link discovery ----------

def _has_data_ext(url: str) -> bool:
    return urlsplit(url).path.lower().endswith(DATA_EXTS)


def _github_raw(url: str) -> str:
    """Rewrite github.com/o/r/blob/ref/path to raw.githubusercontent.com/o/r/ref/path."""
    p = urlsplit(url)
    parts = p.path.split("/")
    if _host(url) in ("github.com", "www.github.com") and len(parts) > 4 and parts[3] == "blob":
        path = "/".join(parts[:3] + parts[4:])
        return f"https://raw.githubusercontent.com{path}" + (f"?{p.query}" if p.query else "")
    return url


def find_file_links(url: str) -> list[str]:
    """Return direct https data-file links for a URL (itself, or links found on the page)."""
    if _has_data_ext(url):
        return [_github_raw(url)]
    client = _client()
    if client is None:
        return []
    res = client.extract(url)
    text = " ".join(str(r.get("raw_content") or "") for r in res.get("results", []))
    out: list[str] = []
    for link in _LINK_RE.findall(text):
        link = _github_raw(link.rstrip(".;:!"))
        if _has_data_ext(link) and link not in out:
            out.append(link)
    return out[:MAX_LINKS]


# ---------- safe fetch ----------

def _bad_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
        or ip.is_reserved or ip.is_unspecified or not ip.is_global
    )


def _validate(url: str) -> tuple[str, str, str]:
    """Check scheme, host, and every resolved address; return (host, pinned_ip, path)."""
    p = urlsplit(url)
    if p.scheme != "https":
        raise FetchError(f"only https URLs are allowed, got {p.scheme or 'none'!r}")
    if p.username is not None or p.password is not None:
        raise FetchError("URLs with embedded credentials are not allowed")
    try:
        port = p.port
    except ValueError as e:
        raise FetchError(f"invalid port: {e}") from e
    if port not in (None, 443):
        raise FetchError(f"only port 443 is allowed, got {port}")
    host = _host(url)
    if not host:
        raise FetchError("URL has no host")
    if host in BLOCKED_NAMES or host.endswith(BLOCKED_SUFFIXES):
        raise FetchError(f"host {host!r} is an internal name")
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None and _bad_ip(literal):
        raise FetchError(f"host {host!r} is a non-public address")
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as e:
        raise FetchError(f"cannot resolve host {host!r}: {e}") from e
    ips = [ipaddress.ip_address(i[4][0].split("%")[0]) for i in infos]
    if not ips:
        raise FetchError(f"host {host!r} resolved to no addresses")
    bad = [str(i) for i in ips if _bad_ip(i)]
    if bad:
        raise FetchError(f"host {host!r} resolves to non-public address(es) {bad}")
    path = (p.path or "/") + (f"?{p.query}" if p.query else "")
    return host, str(ips[0]), path


class _PinnedHTTPS(http.client.HTTPSConnection):
    """HTTPS connection to a pre-validated IP, with SNI and cert checks against the hostname."""

    def __init__(self, host: str, ip: str, timeout: float):
        super().__init__(host, 443, timeout=timeout, context=ssl.create_default_context())
        self._ip = ip

    def connect(self) -> None:
        sock = socket.create_connection((self._ip, 443), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _http_get(host: str, ip: str, path: str, timeout: float):
    """GET over a pinned connection; returns an http.client response (never follows redirects)."""
    conn = _PinnedHTTPS(host, ip, timeout)
    conn.request("GET", path, headers={"User-Agent": "baklava-agent/1", "Accept-Encoding": "identity"})
    return conn.getresponse()


def _read_capped(resp, max_bytes: int, deadline: float) -> bytes:
    length = resp.getheader("Content-Length")
    if length and length.isdigit() and int(length) > max_bytes:
        raise FetchError(f"Content-Length {length} exceeds max_bytes={max_bytes}")
    buf = bytearray()
    while chunk := resp.read(CHUNK):
        buf += chunk
        if len(buf) > max_bytes:
            raise FetchError(f"body exceeds max_bytes={max_bytes}")
        if time.monotonic() > deadline:
            raise FetchError("download exceeded the timeout budget")
    return bytes(buf)


def _download(url: str, max_bytes: int, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    for _ in range(MAX_REDIRECTS + 1):
        host, ip, path = _validate(url)
        try:
            resp = _http_get(host, ip, path, timeout)
        except (OSError, http.client.HTTPException) as e:
            raise FetchError(f"request to {host!r} failed: {type(e).__name__}: {e}") from e
        try:
            if resp.status in REDIRECT_CODES:
                loc = resp.getheader("Location")
                if not loc:
                    raise FetchError(f"redirect {resp.status} without Location")
                url = urljoin(url, loc)
                continue
            if resp.status != 200:
                raise FetchError(f"HTTP {resp.status} from {host!r}")
            return _read_capped(resp, max_bytes, deadline)
        finally:
            resp.close()
    raise FetchError(f"more than {MAX_REDIRECTS} redirects")


def _gunzip_capped(body: bytes, cap: int) -> bytes:
    try:
        out = gzip.GzipFile(fileobj=io.BytesIO(body)).read(cap + 1)
    except (OSError, EOFError) as e:
        raise FetchError(f"invalid gzip body: {e}") from e
    if len(out) > cap:
        raise FetchError(f"decompressed body exceeds {cap} bytes")
    return out


def _parse(body: bytes, ext: str, max_rows: int, cap: int) -> pd.DataFrame:
    if ext == ".parquet":
        pf = pq.ParquetFile(io.BytesIO(body))
        size = sum(pf.metadata.row_group(i).total_byte_size for i in range(pf.metadata.num_row_groups))
        if size > cap:
            raise FetchError(f"parquet uncompressed size {size} exceeds {cap} bytes")
        batches = pf.iter_batches(batch_size=min(max_rows, 65_536))
        rows, parts = 0, []
        for b in batches:
            parts.append(b)
            rows += b.num_rows
            if rows >= max_rows:
                break
        table = pa.Table.from_batches(parts, schema=pf.schema_arrow) if parts else pf.schema_arrow.empty_table()
        return table.slice(0, max_rows).to_pandas()
    if ext == ".csv.gz":
        body = _gunzip_capped(body, cap)
    sep = "\t" if ext == ".tsv" else ","
    return pd.read_csv(io.BytesIO(body), sep=sep, nrows=max_rows, low_memory=False)


def safe_fetch(url: str, max_bytes: int = 50_000_000, timeout: float = 30, max_rows: int = 200_000) -> pd.DataFrame:
    """Download a public csv/tsv/csv.gz/parquet over hardened https and parse it (timeout is total)."""
    url = _github_raw(url)
    ext = next((e for e in DATA_EXTS if urlsplit(url).path.lower().endswith(e)), None)
    if ext is None:
        raise FetchError(f"unsupported file type; allowed: {', '.join(DATA_EXTS)} (never pickle/joblib/npy)")
    body = _download(url, max_bytes, timeout)
    try:
        return _parse(body, ext, max_rows, DECOMPRESS_FACTOR * max_bytes)
    except FetchError:
        raise
    except Exception as e:  # any parser failure on untrusted bytes becomes a FetchError
        raise FetchError(f"could not parse {ext} body: {type(e).__name__}: {e}") from e


# ---------- enrichment ----------

def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    if min(table.shape) < 2:
        return 0.0
    expected = table.sum(1, keepdims=True) @ table.sum(0, keepdims=True) / table.sum()
    chi2 = ((table - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))


def _binned(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > LEAK_BINS:
        return pd.qcut(s, LEAK_BINS, duplicates="drop")
    return s.astype(str)


def _association(x: pd.Series, y: pd.Series) -> float:
    """|Spearman| for numeric pairs, Cramer's V (numeric binned) otherwise."""
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return 0.0
    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        r = x.corr(y, method="spearman")
        return 0.0 if pd.isna(r) else abs(float(r))
    return _cramers_v(_binned(x), _binned(y))


def enrich(df: pd.DataFrame, ext: pd.DataFrame, left_key: str, right_key: str,
           columns: list[str] | None = None, target: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Left-join ext columns (prefixed ext_) onto df; row count and index unchanged."""
    cols = [c for c in (columns or ext.columns) if c != right_key and c != target]
    missing = [c for c in cols if c not in ext.columns]
    if missing:
        raise ValueError(f"columns not in ext: {missing}")
    right = ext.drop_duplicates(right_key, keep="first").set_index(right_key)[cols]
    keys = df[left_key]
    if keys.dtype != right.index.dtype:
        keys, right.index = keys.astype(str), right.index.astype(str)
    match_rate = float(keys.isin(right.index).mean()) if len(df) else 0.0
    if match_rate < MIN_MATCH_RATE:
        raise ValueError(f"match_rate {match_rate:.2f} below {MIN_MATCH_RATE}; refusing to join")
    right = right.add_prefix("ext_")
    joined = right.reindex(keys.to_numpy())
    joined.index = df.index
    out = pd.concat([df, joined], axis=1)
    flags = []
    for raw, new in zip(cols, right.columns):
        if target is None:
            break
        if target.lower() in str(raw).lower() or (target in df and _association(out[new], out[target]) > LEAK_THRESHOLD):
            flags.append(new)
    return out, {"match_rate": match_rate, "new_columns": list(right.columns), "leakage_flags": flags}


def _cast_like(s: pd.Series, dtype) -> pd.Series:
    try:
        return s.astype(dtype)
    except (ValueError, TypeError):
        if pd.api.types.is_numeric_dtype(dtype):
            return pd.to_numeric(s, errors="coerce")
        return s


def align_more_rows(df: pd.DataFrame, ext: pd.DataFrame, target: str,
                    column_map: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Shape ext like df (rename, cast, fill missing with NaN) and drop rows already in df."""
    ext = ext.rename(columns=column_map or {})
    if target not in ext.columns:
        raise ValueError(f"target {target!r} not present in ext after column_map")
    feats = [c for c in df.columns if c != target]
    missing = [c for c in feats if c not in ext.columns]
    coverage = 1.0 - len(missing) / len(feats) if feats else 1.0
    if coverage < MIN_COVERAGE:
        raise ValueError(f"coverage {coverage:.2f} below {MIN_COVERAGE}; missing {missing}")
    rows = pd.DataFrame(
        {c: (_cast_like(ext[c], df[c].dtype) if c in ext.columns else pd.Series(np.nan, index=ext.index))
         for c in df.columns}
    )
    seen = rows.merge(df.drop_duplicates(), how="left", indicator=True)["_merge"].eq("both").to_numpy()
    rows = rows[~seen].reset_index(drop=True)
    return rows, {"n_rows": len(rows), "coverage": coverage, "missing_columns": missing}

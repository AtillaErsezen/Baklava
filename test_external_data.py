import gzip
import io
import os
import socket
from unittest import mock

import numpy as np
import pandas as pd

import external_data as ed

PUBLIC_IP = "93.184.216.34"


class FakeResp:
    """Minimal stand-in for an http.client response."""

    def __init__(self, status: int = 200, body: bytes = b"", headers: dict | None = None):
        self.status = status
        self._buf = io.BytesIO(body)
        self._headers = {k.lower(): v for k, v in (headers or {}).items()}

    def getheader(self, name: str, default=None):
        return self._headers.get(name.lower(), default)

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def close(self) -> None:
        pass


def _dns(table: dict[str, list[str]]):
    """Fake getaddrinfo backed by a host -> [ip] table; unknown hosts fail."""

    def fake(host, port, *args, **kwargs):
        if host not in table:
            raise socket.gaierror("unknown host")
        out = []
        for ip in table[host]:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            out.append((fam, socket.SOCK_STREAM, 6, "", (ip, port)))
        return out

    return fake


def _net(dns: dict[str, list[str]], routes: dict[str, FakeResp]):
    """Patch DNS and the HTTP layer; routes keyed by 'host/path'."""
    calls: list[tuple[str, str, str]] = []

    def fake_get(host, ip, path, timeout):
        calls.append((host, ip, path))
        return routes[host + path]

    patches = [
        mock.patch.object(ed.socket, "getaddrinfo", _dns(dns)),
        mock.patch.object(ed, "_http_get", fake_get),
    ]
    return patches, calls


def _expect_fetch_error(url: str, dns=None, routes=None, **kw) -> str:
    patches, calls = _net(dns or {}, routes or {})
    for p in patches:
        p.start()
    try:
        ed.safe_fetch(url, **kw)
    except ed.FetchError as e:
        return str(e)
    finally:
        for p in patches:
            p.stop()
    raise AssertionError(f"expected FetchError for {url}")


def test_rejects_http():
    assert "https" in _expect_fetch_error("http://www.openml.org/x.csv")


def test_rejects_loopback_literal():
    _expect_fetch_error("https://127.0.0.1/x.csv")


def test_rejects_private_literal():
    _expect_fetch_error("https://10.0.0.5/x.csv")


def test_rejects_ipv6_loopback():
    _expect_fetch_error("https://[::1]/x.csv")


def test_rejects_localhost():
    _expect_fetch_error("https://localhost/x.csv", dns={"localhost": [PUBLIC_IP]})


def test_rejects_internal_names():
    _expect_fetch_error("https://metadata.google.internal/x.csv", dns={"metadata.google.internal": [PUBLIC_IP]})
    _expect_fetch_error("https://db.corp.internal/x.csv", dns={"db.corp.internal": [PUBLIC_IP]})


def test_rejects_host_resolving_to_metadata_ip():
    _expect_fetch_error("https://evil.huggingface.co/x.csv", dns={"evil.huggingface.co": ["169.254.169.254"]})


def test_rejects_if_any_address_private():
    _expect_fetch_error("https://mixed.huggingface.co/x.csv", dns={"mixed.huggingface.co": [PUBLIC_IP, "192.168.1.1"]})


def test_rejects_redirect_to_private_host():
    dns = {"data.openml.org": [PUBLIC_IP], "inner.huggingface.co": ["10.1.2.3"]}
    routes = {"data.openml.org/x.csv": FakeResp(302, headers={"Location": "https://inner.huggingface.co/x.csv"})}
    _expect_fetch_error("https://data.openml.org/x.csv", dns=dns, routes=routes)


def test_rejects_too_many_redirects():
    dns = {"data.openml.org": [PUBLIC_IP]}
    routes = {"data.openml.org/x.csv": FakeResp(302, headers={"Location": "/x.csv"})}
    assert "redirect" in _expect_fetch_error("https://data.openml.org/x.csv", dns=dns, routes=routes)


def test_rejects_oversize_body():
    dns = {"data.openml.org": [PUBLIC_IP]}
    body = b"a,b\n" + b"1,2\n" * 1000
    routes = {"data.openml.org/x.csv": FakeResp(200, body)}
    assert "max_bytes" in _expect_fetch_error("https://data.openml.org/x.csv", dns=dns, routes=routes, max_bytes=100)


def test_rejects_oversize_content_length():
    dns = {"data.openml.org": [PUBLIC_IP]}
    routes = {"data.openml.org/x.csv": FakeResp(200, b"a\n1\n", {"Content-Length": "999999"})}
    _expect_fetch_error("https://data.openml.org/x.csv", dns=dns, routes=routes, max_bytes=100)


def test_rejects_pickle_extension():
    for url in ["https://data.openml.org/x.pkl", "https://data.openml.org/m.joblib", "https://data.openml.org/a.npy"]:
        _expect_fetch_error(url, dns={"data.openml.org": [PUBLIC_IP]})


def test_rejects_non_443_port_and_userinfo():
    _expect_fetch_error("https://data.openml.org:8080/x.csv", dns={"data.openml.org": [PUBLIC_IP]})
    _expect_fetch_error("https://u:p@data.openml.org/x.csv", dns={"data.openml.org": [PUBLIC_IP]})


def test_rejects_gzip_bomb():
    dns = {"data.openml.org": [PUBLIC_IP]}
    body = gzip.compress(b"a\n" + b"1\n" * 500_000)
    routes = {"data.openml.org/x.csv.gz": FakeResp(200, body)}
    _expect_fetch_error("https://data.openml.org/x.csv.gz", dns=dns, routes=routes, max_bytes=len(body) + 10)


def _fetch(url: str, dns: dict, routes: dict, **kw):
    patches, calls = _net(dns, routes)
    for p in patches:
        p.start()
    try:
        return ed.safe_fetch(url, **kw), calls
    finally:
        for p in patches:
            p.stop()


def test_accepts_public_csv():
    dns = {"data.openml.org": [PUBLIC_IP]}
    routes = {"data.openml.org/x.csv": FakeResp(200, b"a,b\n1,2\n3,4\n")}
    df, calls = _fetch("https://data.openml.org/x.csv?v=1", dns, routes)
    assert isinstance(df, pd.DataFrame) and df.shape == (2, 2) and list(df.columns) == ["a", "b"]
    assert calls == [("data.openml.org", PUBLIC_IP, "/x.csv")]  # query stripped: no data rides out


def test_follows_safe_redirect_and_caps_rows():
    dns = {"data.openml.org": [PUBLIC_IP], "cdn.zenodo.org": [PUBLIC_IP]}
    body = gzip.compress(b"a\tb\n" + b"1\t2\n" * 50)
    routes = {
        "data.openml.org/x.tsv": FakeResp(301, headers={"Location": "https://cdn.zenodo.org/y.tsv"}),
        "cdn.zenodo.org/y.tsv": FakeResp(200, b"a\tb\n" + b"1\t2\n" * 50),
        "data.openml.org/z.csv.gz": FakeResp(200, body),
    }
    df, calls = _fetch("https://data.openml.org/x.tsv", dns, routes, max_rows=10)
    assert df.shape == (10, 2) and [c[0] for c in calls] == ["data.openml.org", "cdn.zenodo.org"]
    df, _ = _fetch("https://data.openml.org/z.csv.gz", dns, routes)
    assert df.shape == (50, 1)  # tab separated but read as csv: one column


def test_accepts_parquet():
    buf = io.BytesIO()
    pd.DataFrame({"a": range(20), "b": list("xy") * 10}).to_parquet(buf)
    dns = {"data.openml.org": [PUBLIC_IP]}
    routes = {"data.openml.org/x.parquet": FakeResp(200, buf.getvalue())}
    df, _ = _fetch("https://data.openml.org/x.parquet", dns, routes, max_rows=5)
    assert df.shape == (5, 2) and df["a"].tolist() == [0, 1, 2, 3, 4]


def test_github_blob_to_raw():
    url = "https://github.com/owner/repo/blob/main/data/train.csv"
    assert ed.find_file_links(url) == ["https://raw.githubusercontent.com/owner/repo/main/data/train.csv"]
    assert ed.find_file_links("https://zenodo.org/x/data.parquet") == ["https://zenodo.org/x/data.parquet"]


def test_find_file_links_extracts():
    page = (
        "See https://github.com/o/r/blob/main/a.csv and https://www.openml.org/b.tsv?raw=1, "
        "also http://insecure.org/c.csv and https://www.openml.org/model.pkl and https://www.openml.org/d.csv.gz."
    )

    class FakeClient:
        def __init__(self, api_key):
            pass

        def extract(self, url):
            return {"results": [{"url": url, "raw_content": page}]}

    with mock.patch.dict(os.environ, {"TAVILY_API_KEY": "k"}), mock.patch.object(ed, "TavilyClient", FakeClient):
        links = ed.find_file_links("https://www.openml.org/dataset-page")
    assert links == [
        "https://raw.githubusercontent.com/o/r/main/a.csv",
        "https://www.openml.org/b.tsv?raw=1",
        "https://www.openml.org/d.csv.gz",
    ]


def test_search_without_key_returns_error():
    env = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}
    with mock.patch.dict(os.environ, env, clear=True):
        out = ed.search("house prices", "more rows")
    assert isinstance(out, dict) and "TAVILY_API_KEY" in out["error"]


def test_search_compacts_results():
    seen = {}

    class FakeClient:
        def __init__(self, api_key):
            pass

        def search(self, query, **kw):
            seen.update(kw)
            return {"results": [
                {"title": "UCI Adult", "url": "https://archive.ics.uci.edu/dataset/2/adult", "content": "x" * 500},
                {"title": "Kaggle", "url": "https://www.kaggle.com/datasets/a/b", "content": "k"},
            ]}

    with mock.patch.dict(os.environ, {"TAVILY_API_KEY": "k"}), mock.patch.object(ed, "TavilyClient", FakeClient):
        out = ed.search("adult income", "complementary features", max_results=3)
    assert seen["max_results"] == 3 and "openml.org" in seen["include_domains"]
    assert len(out[0]["snippet"]) <= 200 and out[0]["downloadable"] and out[0]["purpose"] == "complementary features"
    assert out[1]["downloadable"] is False


def _base_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"id": np.arange(n), "x": rng.normal(size=n), "y": rng.normal(size=n)},
        index=pd.RangeIndex(1000, 1000 + n),
    )


def test_enrich_keeps_rows_and_match_rate():
    df = _base_df()
    ext = pd.DataFrame({"key": list(range(80)) + [0, 1], "pop": np.arange(82.0), "y": 0.0})
    out, rep = ed.enrich(df, ext, "id", "key", target="y")
    assert len(out) == len(df) and out.index.equals(df.index)
    assert abs(rep["match_rate"] - 0.8) < 1e-9 and rep["new_columns"] == ["ext_pop"]
    assert "ext_y" not in out.columns and out.loc[1000, "ext_pop"] == 0.0


def test_enrich_flags_leak():
    df = _base_df()
    ext = pd.DataFrame({"key": df["id"], "leak": df["y"].to_numpy() * 2 + 1, "noise": np.random.default_rng(1).normal(size=100),
                        "y_lag": np.random.default_rng(2).normal(size=100)})
    _, rep = ed.enrich(df, ext, "id", "key", target="y")
    assert "ext_leak" in rep["leakage_flags"] and "ext_y_lag" in rep["leakage_flags"]
    assert "ext_noise" not in rep["leakage_flags"]


def test_enrich_refuses_low_match():
    df = _base_df()
    ext = pd.DataFrame({"key": range(10), "pop": 1.0})
    try:
        ed.enrich(df, ext, "id", "key")
    except ValueError as e:
        assert "match_rate" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_align_more_rows():
    df = pd.DataFrame({"age": [30, 40], "income": [1.0, 2.0], "city": ["a", "b"], "t": [0, 1]})
    ext = pd.DataFrame({"Age": ["30", "50", "60"], "Income": [1.0, 3.0, 4.0], "city": ["a", "c", "d"], "label": [0, 1, 0]})
    rows, rep = ed.align_more_rows(df, ext, "t", {"Age": "age", "Income": "income", "label": "t"})
    assert rep["n_rows"] == 2 and rep["coverage"] == 1.0 and rep["missing_columns"] == []
    assert list(rows.columns) == list(df.columns) and rows["age"].dtype == df["age"].dtype
    wide = df.assign(p=1, q=2)
    rows2, rep2 = ed.align_more_rows(wide, ext.assign(p=5, q=6), "t", {"Age": "age", "label": "t"})
    assert rep2["missing_columns"] == ["income"] and rep2["coverage"] == 0.8 and rows2["income"].isna().all()
    try:
        ed.align_more_rows(df, ext[["Age", "label"]], "t", {"Age": "age", "label": "t"})
    except ValueError as e:
        assert "coverage" in str(e)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")

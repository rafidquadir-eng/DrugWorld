"""Concurrent HTTPS fetches into memory, within EBI's per-client connection limit.

EBI's FTP host serves idr0119 over HTTPS (HTTP/1.1) and refuses connections above roughly
24-32 concurrent ones per client (reports/D0_THROUGHPUT.md); the BIA S3 bucket takes 128. One
urllib3 pool per process with `streams` keep-alive connections; refused or failed requests are
retried with exponential back-off (12 attempts, waits capped at 60 s; HTTP 404: 3;
raised from 8 attempts / 30 s after the D0 pilot, where a few refused requests ran out), and a URL that keeps failing is
returned with its error. Nothing is written to disk.
"""
from __future__ import annotations

import random
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

import urllib3

MAX_ATTEMPTS = 12
MAX_404_ATTEMPTS = 3


class Fetcher:
    def __init__(self, streams: int = 24):
        self.streams = streams
        self.http = urllib3.PoolManager(num_pools=4, maxsize=streams, block=True,
                                        retries=False, timeout=urllib3.Timeout(20, 120))
        self.pool = ThreadPoolExecutor(streams)

    def get(self, url: str) -> tuple[bytes | None, str | None]:
        err = None
        for att in range(MAX_ATTEMPTS):
            try:
                r = self.http.request("GET", url, preload_content=True)
                if r.status == 200:
                    return r.data, None
                err = f"HTTP {r.status}"
                if r.status == 404 and att + 1 >= MAX_404_ATTEMPTS:
                    break
            except Exception as exc:          # connection refused, reset, timeout
                err = f"{type(exc).__name__}: {exc}"[:200]
            time.sleep(min(60.0, 0.5 * 2 ** att) * (0.5 + random.random()))
        return None, f"{err} after {att + 1} attempts"

    def map(self, urls: Iterable[str]) -> Iterator[tuple[str, bytes | None, str | None]]:
        """Fetch all `urls` concurrently; yields (url, body, error) in input order."""
        urls = list(urls)
        for u, (b, e) in zip(urls, self.pool.map(self.get, urls)):
            yield u, b, e

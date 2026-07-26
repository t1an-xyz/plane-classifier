"""A small throttled + retrying HTTP session shared by every source module.

Wikimedia will return HTTP 429 if you hammer the API, so we keep a per-host
minimum gap between requests and back off on transient errors.
"""

from __future__ import annotations

import time
from urllib.parse import urlsplit

import requests

from . import config

# Never sleep longer than this on a retry, even if Retry-After says to.
MAX_BACKOFF = 8.0


class ThrottledSession:
    """requests.Session wrapper with per-host pacing and retry/backoff."""

    def __init__(self, user_agent: str | None = None) -> None:
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": user_agent or config.USER_AGENT}
        )
        # last request time per host
        self._last: dict[str, float] = {}
        self._gaps: dict[str, float] = {}
        # optional per-host User-Agent override (upload.wikimedia.org now 403s
        # obvious bot UAs on static media, so downloads need a browser-like UA)
        self._uas: dict[str, str] = {}

    def set_gap(self, host: str, seconds: float) -> None:
        self._gaps[host] = seconds

    def set_ua(self, host: str, user_agent: str) -> None:
        self._uas[host] = user_agent

    def _throttle(self, url: str) -> None:
        host = urlsplit(url).netloc
        gap = self._gaps.get(host, config.API_THROTTLE)
        now = time.time()
        wait = gap - (now - self._last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.time()

    def get(
        self,
        url: str,
        *,
        params: dict | None = None,
        stream: bool = False,
        tries: int = 5,
        timeout: int | tuple[int, int] = 45,
    ) -> requests.Response:
        last_exc: Exception | None = None
        host = urlsplit(url).netloc
        headers = {"User-Agent": self._uas[host]} if host in self._uas else None
        for attempt in range(tries):
            self._throttle(url)
            try:
                resp = self._session.get(
                    url, params=params, stream=stream, timeout=timeout,
                    headers=headers,
                )
            except requests.RequestException as exc:  # network hiccup
                last_exc = exc
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 200:
                return resp
            # Non-200: the (possibly streamed) body MUST be released, otherwise
            # the connection stays checked out of urllib3's pool and later
            # requests block forever waiting for a free connection.
            retryable = resp.status_code in (429, 500, 502, 503, 504)
            retry_after = resp.headers.get("Retry-After")
            resp.close()
            if not retryable:
                resp.raise_for_status()
            # Out of retries -> give up quietly (caller decides what to do).
            if attempt >= tries - 1:
                break
            # Back off, but never honour an absurdly long Retry-After: we have
            # thousands of candidate images, so it's cheaper to skip and move on.
            delay = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else 2 * (attempt + 1)
            )
            time.sleep(min(delay, MAX_BACKOFF))
        if last_exc:
            raise last_exc
        raise RuntimeError(f"GET failed after {tries} tries: {url}")

    def get_json(self, url: str, params: dict, tries: int = 5) -> dict:
        # ``get`` already retries transient HTTP errors (429/5xx) internally,
        # so here we only re-loop for non-JSON bodies and MediaWiki maxlag.
        for attempt in range(tries):
            resp = self.get(url, params=params)
            if "json" in resp.headers.get("content-type", ""):
                data = resp.json()
                # MediaWiki maxlag / throttling comes back as an error object
                if "error" in data and data["error"].get("code") == "maxlag":
                    time.sleep(2 * (attempt + 1))
                    continue
                return data
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Non-JSON response after {tries} tries: {url}")

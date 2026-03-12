#!/usr/bin/env python3
"""Shared network resilience helpers.

Focus:
- Detect proxy-related failures/noisy 5xx responses.
- Retry once through a no-proxy lane (`trust_env=False`) to cut proxy blast radius.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Iterable, Optional

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import ProxyError, RequestException, SSLError, Timeout

DEFAULT_PROXY_HINT_MARKERS = (
    "proxy error",
    "proxy",
    "upstream",
    "connect error",
    "tunnel",
)

DEFAULT_PROXY_EXCEPTION_TYPES = (ProxyError, ReqConnectionError, SSLError, Timeout)
DEFAULT_PROXY_GATEWAY_STATUS = (502, 503, 504)


def _blank_stats() -> dict[str, Any]:
    return {
        "requests_total": 0,
        "bypass_attempted": 0,
        "bypass_success": 0,
        "bypass_failed": 0,
        "no_proxy_retries": 0,
        "no_proxy_retry_exhausted": 0,
        "hint_response": 0,
        "hint_exception": 0,
        "reason_counts": {
            "response_hint": 0,
            "exception_hint": 0,
        },
        "last_reason": None,
    }


_STATS_LOCK = threading.Lock()
_STATS: dict[str, Any] = _blank_stats()


def reset_proxy_bypass_stats() -> None:
    global _STATS
    with _STATS_LOCK:
        _STATS = _blank_stats()


def get_proxy_bypass_stats() -> dict[str, Any]:
    with _STATS_LOCK:
        return {
            "requests_total": int(_STATS.get("requests_total") or 0),
            "bypass_attempted": int(_STATS.get("bypass_attempted") or 0),
            "bypass_success": int(_STATS.get("bypass_success") or 0),
            "bypass_failed": int(_STATS.get("bypass_failed") or 0),
            "no_proxy_retries": int(_STATS.get("no_proxy_retries") or 0),
            "no_proxy_retry_exhausted": int(_STATS.get("no_proxy_retry_exhausted") or 0),
            "hint_response": int(_STATS.get("hint_response") or 0),
            "hint_exception": int(_STATS.get("hint_exception") or 0),
            "reason_counts": {
                "response_hint": int((((_STATS.get("reason_counts") or {}).get("response_hint")) or 0)),
                "exception_hint": int((((_STATS.get("reason_counts") or {}).get("exception_hint")) or 0)),
            },
            "last_reason": _STATS.get("last_reason"),
        }


def _record_request() -> None:
    with _STATS_LOCK:
        _STATS["requests_total"] = int(_STATS.get("requests_total") or 0) + 1


def _record_hint(source: str) -> None:
    with _STATS_LOCK:
        key = "hint_response" if source == "response_hint" else "hint_exception"
        _STATS[key] = int(_STATS.get(key) or 0) + 1
        _STATS["bypass_attempted"] = int(_STATS.get("bypass_attempted") or 0) + 1
        reason_counts = _STATS.get("reason_counts")
        if not isinstance(reason_counts, dict):
            reason_counts = {"response_hint": 0, "exception_hint": 0}
            _STATS["reason_counts"] = reason_counts
        reason_counts[source] = int(reason_counts.get(source) or 0) + 1
        _STATS["last_reason"] = source


def _record_bypass_result(success: bool, *, retries: int = 0, exhausted: bool = False) -> None:
    with _STATS_LOCK:
        key = "bypass_success" if success else "bypass_failed"
        _STATS[key] = int(_STATS.get(key) or 0) + 1
        _STATS["no_proxy_retries"] = int(_STATS.get("no_proxy_retries") or 0) + max(0, int(retries))
        if exhausted:
            _STATS["no_proxy_retry_exhausted"] = int(_STATS.get("no_proxy_retry_exhausted") or 0) + 1


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = int(raw)
    except Exception:
        value = int(default)
    return max(minimum, min(maximum, value))


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    return max(minimum, min(maximum, value))


def _no_proxy_retry_policy() -> tuple[int, float]:
    max_attempts = _int_env("NET_RESILIENCE_NO_PROXY_RETRY_MAX", 2, minimum=1, maximum=8)
    backoff_sec = _float_env("NET_RESILIENCE_NO_PROXY_RETRY_BACKOFF_SEC", 0.10, minimum=0.0, maximum=5.0)
    return max_attempts, backoff_sec


def _call_no_proxy_with_retry(
    call: Callable[[], requests.Response],
) -> tuple[Optional[requests.Response], Optional[RequestException], int, bool]:
    max_attempts, backoff_sec = _no_proxy_retry_policy()
    last_exc: Optional[RequestException] = None
    for attempt_idx in range(max_attempts):
        try:
            return call(), None, attempt_idx, False
        except RequestException as exc:
            last_exc = exc
            if attempt_idx + 1 < max_attempts and backoff_sec > 0:
                time.sleep(backoff_sec * (2**attempt_idx))
    retries = max(0, max_attempts - 1)
    return None, last_exc, retries, True


def is_proxy_hint_text(text: Any, markers: Optional[Iterable[str]] = None) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    active_markers = tuple(markers or DEFAULT_PROXY_HINT_MARKERS)
    return any(marker in raw for marker in active_markers)


def is_proxy_hint_response(
    resp: requests.Response,
    markers: Optional[Iterable[str]] = None,
    gateway_statuses: Optional[Iterable[int]] = None,
) -> bool:
    status = int(getattr(resp, "status_code", 0) or 0)
    gateway_set = {int(code) for code in (gateway_statuses or DEFAULT_PROXY_GATEWAY_STATUS)}
    if status in gateway_set:
        return True
    if status < 500:
        return False
    return is_proxy_hint_text(getattr(resp, "text", ""), markers=markers)


def request_no_proxy(method: str, url: str, **kwargs: Any) -> requests.Response:
    sess = requests.Session()
    try:
        sess.trust_env = False
        return sess.request(method=method, url=url, **kwargs)
    finally:
        sess.close()


def request_with_proxy_bypass(
    method: str,
    url: str,
    *,
    request_func: Optional[Callable[..., requests.Response]] = None,
    no_proxy_request_func: Optional[Callable[..., requests.Response]] = None,
    markers: Optional[Iterable[str]] = None,
    proxy_exception_types: Optional[tuple[type[BaseException], ...]] = None,
    **kwargs: Any,
) -> requests.Response:
    request_impl = request_func or requests.request
    no_proxy_impl = no_proxy_request_func or request_no_proxy
    exc_types = proxy_exception_types or DEFAULT_PROXY_EXCEPTION_TYPES
    _record_request()
    try:
        resp = request_impl(method=method, url=url, **kwargs)
    except exc_types as exc:  # type: ignore[misc]
        if is_proxy_hint_text(exc, markers=markers):
            _record_hint("exception_hint")
            out, _exc, retries, exhausted = _call_no_proxy_with_retry(
                lambda: no_proxy_impl(method=method, url=url, **kwargs)
            )
            if out is not None:
                _record_bypass_result(True, retries=retries)
                return out
            _record_bypass_result(False, retries=retries, exhausted=exhausted)
        raise

    if is_proxy_hint_response(resp, markers=markers):
        _record_hint("response_hint")
        out, _exc, retries, exhausted = _call_no_proxy_with_retry(
            lambda: no_proxy_impl(method=method, url=url, **kwargs)
        )
        if out is not None:
            _record_bypass_result(True, retries=retries)
            return out
        _record_bypass_result(False, retries=retries, exhausted=exhausted)
        return resp
    return resp


def get_no_proxy(url: str, **kwargs: Any) -> requests.Response:
    return request_no_proxy("GET", url, **kwargs)


def get_with_proxy_bypass(
    url: str,
    *,
    request_get: Optional[Callable[..., requests.Response]] = None,
    no_proxy_get: Optional[Callable[..., requests.Response]] = None,
    markers: Optional[Iterable[str]] = None,
    proxy_exception_types: Optional[tuple[type[BaseException], ...]] = None,
    **kwargs: Any,
) -> requests.Response:
    request_impl = request_get or requests.get
    no_proxy_impl = no_proxy_get or get_no_proxy
    exc_types = proxy_exception_types or DEFAULT_PROXY_EXCEPTION_TYPES
    _record_request()
    try:
        resp = request_impl(url, **kwargs)
    except exc_types as exc:  # type: ignore[misc]
        if is_proxy_hint_text(exc, markers=markers):
            _record_hint("exception_hint")
            out, _exc, retries, exhausted = _call_no_proxy_with_retry(
                lambda: no_proxy_impl(url, **kwargs)
            )
            if out is not None:
                _record_bypass_result(True, retries=retries)
                return out
            _record_bypass_result(False, retries=retries, exhausted=exhausted)
        raise

    if is_proxy_hint_response(resp, markers=markers):
        _record_hint("response_hint")
        out, _exc, retries, exhausted = _call_no_proxy_with_retry(lambda: no_proxy_impl(url, **kwargs))
        if out is not None:
            _record_bypass_result(True, retries=retries)
            return out
        _record_bypass_result(False, retries=retries, exhausted=exhausted)
        return resp
    return resp

"""Bounded retries for source catalogue and package GET requests."""

from __future__ import annotations

import errno
import hashlib
import http.client
import json
import logging
from pathlib import Path
import random
import shutil
import socket
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

MAXIMUM_ATTEMPTS = 5
INITIAL_DELAY_SECONDS = 1.0
MAXIMUM_DELAY_SECONDS = 8.0
JITTER_RATIO = 0.25
LOGGER = logging.getLogger(__name__)
_RECEIPT_ATTRIBUTE = "_simready_download_retry_receipt"
_NETWORK_ERRNO = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNRESET,
        errno.ETIMEDOUT,
        errno.EHOSTUNREACH,
        errno.ENETUNREACH,
        errno.EPIPE,
        errno.ENETDOWN,
        errno.ECONNREFUSED,
    }
)
_TRANSIENT_SSL_REASON = frozenset(
    {
        "UNEXPECTED_EOF_WHILE_READING",
        "DECRYPTION_FAILED_OR_BAD_RECORD_MAC",
        "SSLV3_ALERT_BAD_RECORD_MAC",
    }
)


class IncompleteTransferError(ValueError):
    """A response ended before supplying its declared complete body."""


def is_transient(error):
    """Identify interrupted transfers without retrying authority or content errors."""
    seen = set()
    while isinstance(error, urllib.error.URLError) and not isinstance(
        error, urllib.error.HTTPError
    ):
        if id(error) in seen or not isinstance(error.reason, BaseException):
            return False
        seen.add(id(error))
        error = error.reason
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or 500 <= error.code < 600
    if isinstance(error, ssl.SSLCertVerificationError):
        return False
    if isinstance(error, (ssl.SSLEOFError, ssl.SSLZeroReturnError)):
        return True
    if isinstance(error, ssl.SSLError):
        return getattr(error, "reason", None) in _TRANSIENT_SSL_REASON
    if isinstance(error, socket.gaierror):
        return error.errno == socket.EAI_AGAIN
    if isinstance(
        error,
        (
            IncompleteTransferError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            TimeoutError,
            ConnectionError,
        ),
    ):
        return True
    return isinstance(error, OSError) and error.errno in _NETWORK_ERRNO


def retry_delay(failed_attempt):
    """Return a capped exponential delay with jitter before the next request."""
    if type(failed_attempt) is not int or not 1 <= failed_attempt < MAXIMUM_ATTEMPTS:
        raise ValueError("A retry delay requires a remaining attempt")
    nominal = min(
        MAXIMUM_DELAY_SECONDS, INITIAL_DELAY_SECONDS * 2 ** (failed_attempt - 1)
    )
    return min(
        MAXIMUM_DELAY_SECONDS,
        nominal * random.uniform(1.0 - JITTER_RATIO, 1.0 + JITTER_RATIO),
    )


def retry_receipt(error):
    """Return exhaustion evidence bound to this exact selected exception."""
    receipt = getattr(error, _RECEIPT_ATTRIBUTE, None)
    if (
        not isinstance(receipt, dict)
        or not is_transient(error)
        or receipt.get("schema") != "simready-bench-download-exhaustion-v1"
        or receipt.get("method") != "GET"
        or receipt.get("attempts") != MAXIMUM_ATTEMPTS
        or receipt.get("maximum_attempts") != MAXIMUM_ATTEMPTS
        or receipt.get("exception_class")
        != type(error).__module__ + "." + type(error).__name__
        or (
            isinstance(error, urllib.error.HTTPError)
            and receipt.get("http_status") != error.code
        )
    ):
        return None
    return dict(receipt)


def _record_exhaustion(error, url):
    receipt = {
        "schema": "simready-bench-download-exhaustion-v1",
        "method": "GET",
        "attempts": MAXIMUM_ATTEMPTS,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "transient": True,
        "host": urllib.parse.urlsplit(url).hostname,
        "url_sha256": hashlib.sha256(url.encode("utf-8")).hexdigest(),
        "exception_class": type(error).__module__ + "." + type(error).__name__,
    }
    if isinstance(error, urllib.error.HTTPError):
        receipt["http_status"] = error.code
    try:
        setattr(error, _RECEIPT_ATTRIBUTE, receipt)
    except (AttributeError, TypeError):
        pass


def _retry_get(operation, url):
    for attempt in range(1, MAXIMUM_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as error:
            transient = is_transient(error)
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            if not transient or attempt == MAXIMUM_ATTEMPTS:
                if transient:
                    _record_exhaustion(error, url)
                raise
            delay = retry_delay(attempt)
            LOGGER.warning(
                json.dumps(
                    {
                        "event": "asset_download_retry",
                        "host": urllib.parse.urlsplit(url).hostname,
                        "failed_attempt": attempt,
                        "maximum_attempts": MAXIMUM_ATTEMPTS,
                        "error_type": type(error).__name__,
                        "delay_seconds": round(delay, 6),
                    },
                    ensure_ascii=True,
                )
            )
            time.sleep(delay)
    raise AssertionError("Unreachable source download retry state")


def read_json_page(url, *, timeout=60):
    """Read a complete catalogue response before accepting its JSON and next link."""

    def read():
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
            declared = response.headers.get("Content-Length")
            if declared is not None:
                if len(payload) < int(declared):
                    raise IncompleteTransferError("Incomplete source catalogue response")
                if len(payload) > int(declared):
                    raise ValueError("Source catalogue response size mismatch")
            return json.loads(payload), response.headers.get("Link", "")

    return _retry_get(read, url)


def download_file(url, temporary, expected_size, *, timeout=120):
    """Restart an interrupted package file from zero and remove failed transfers."""
    temporary = Path(temporary)

    def transfer():
        try:
            with (
                urllib.request.urlopen(url, timeout=timeout) as response,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            size = temporary.stat().st_size
            if size < expected_size:
                raise IncompleteTransferError("Incomplete source package file")
            if size > expected_size:
                raise ValueError("download size mismatch")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    return _retry_get(transfer, url)

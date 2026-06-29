"""
dotnet_executor.py
==================
Handles direct communication and execution commands to the .NET AiCommand API.
Includes a thread-safe Circuit Breaker and Retry policy.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import requests

from metrics import metrics_collector
from settings import get_dotnet_config
from telemetry import request_id_var, session_id_var, trace_id_var
from utils import extract_records as _extract_records

logger = logging.getLogger(__name__)

# =============================================================================
# .NET CONFIGURATION
# =============================================================================

DOTNET_API_BASE_URL = get_dotnet_config().BASE_URL
DOTNET_EXECUTE_URL = f"{DOTNET_API_BASE_URL}/api/AiCommand/execute"
DOTNET_API_KEY = os.getenv("DOTNET_API_KEY", "")
from dotnet_security import resolve_dotnet_verify_ssl

DOTNET_VERIFY_SSL = resolve_dotnet_verify_ssl(logger)

_dotnet_session = requests.Session()

DOTNET_CONNECT_TIMEOUT = float(os.getenv("DOTNET_CONNECT_TIMEOUT", "5.0"))
DOTNET_READ_TIMEOUT = float(os.getenv("DOTNET_READ_TIMEOUT", "30.0"))

if not DOTNET_VERIFY_SSL:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# CIRCUIT BREAKER IMPLEMENTATION
# =============================================================================


class CircuitBreaker:
    """
    Thread-safe Circuit Breaker.
    Trips after 5 consecutive transient failures.
    Transitions from OPEN to HALF-OPEN after a 10-second cooldown.
    """

    def __init__(
        self, failure_threshold: int = 5, recovery_timeout: float = 10.0
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_state_change = time.time()
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        """Check if request is allowed. Transitions state if cooldown elapsed."""
        with self._lock:
            now = time.time()
            if self.state == "OPEN":
                if now - self.last_state_change >= self.recovery_timeout:
                    old_state = self.state
                    self.state = "HALF-OPEN"
                    self.last_state_change = now
                    logger.info(
                        json.dumps(
                            {
                                "message": "Circuit breaker state transition",
                                "previous_state": old_state,
                                "new_state": self.state,
                            }
                        )
                    )
                    return True
                return False
            return True

    def record_success(self) -> None:
        """Reset failures and close circuit breaker on success."""
        with self._lock:
            if self.state != "CLOSED":
                logger.info(
                    json.dumps(
                        {
                            "message": "Circuit breaker recovered",
                            "previous_state": self.state,
                            "new_state": "CLOSED",
                        }
                    )
                )
            self.state = "CLOSED"
            self.failure_count = 0
            self.last_state_change = time.time()

    def record_failure(self) -> None:
        """Increment failure counter and trip circuit breaker if threshold reached."""
        with self._lock:
            self.failure_count += 1
            now = time.time()
            logger.warning(
                json.dumps(
                    {
                        "message": "Circuit breaker recorded failure",
                        "failure_count": self.failure_count,
                        "failure_threshold": self.failure_threshold,
                    }
                )
            )
            if (
                self.state == "HALF-OPEN"
                or self.failure_count >= self.failure_threshold
            ):
                if self.state != "OPEN":
                    logger.error(
                        json.dumps(
                            {
                                "message": "Circuit breaker tripped",
                                "previous_state": self.state,
                                "new_state": "OPEN",
                            }
                        )
                    )
                self.state = "OPEN"
                self.last_state_change = now


# Global singleton instance of CircuitBreaker
_cb = CircuitBreaker()


# =============================================================================
# CALL EXECUTION WITH RETRY & TIMEOUT
# =============================================================================


def _call_dotnet(
    payload: Dict,
    trace_id: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    query_type: Optional[str] = None,
) -> Tuple[Any, Optional[str]]:
    """
    Execute a single .NET API call.

    Returns (data, None) on success, or (None, error_message) on failure.
    Includes:
      - Circuit Breaker check
      - Exponential backoff retries (1s, 2s, 4s) for 429, 502, 503, 504 and connection errors/timeouts.
      - Strict timeout=(5, 30) for every request.
    """
    effective_trace_id = trace_id or trace_id_var.get() or "N/A"
    effective_request_id = request_id or request_id_var.get() or "N/A"
    effective_session_id = session_id or session_id_var.get() or "N/A"

    if not _cb.allow_request():
        logger.warning(
            json.dumps(
                {
                    "message": "Request rejected: Circuit breaker is OPEN",
                    "trace_id": effective_trace_id,
                }
            )
        )
        return None, "Circuit breaker is open. Backend is temporarily unreachable."

    headers = {
        "Content-Type": "application/json",
        "X-Request-Id": effective_request_id,
        "X-Trace-Id": effective_trace_id,
        "X-Session-Id": effective_session_id,
    }
    if DOTNET_API_KEY:
        headers["X-Api-Key"] = DOTNET_API_KEY

    logger.info(
        {
            "message": "Initiating .NET API call",
            "trace_id": effective_trace_id,
            "session_id": effective_session_id,
            "query_type": query_type or "unknown",
            "payload_keys": (
                list(payload.keys())
                if isinstance(payload, dict)
                else type(payload).__name__
            ),
            "api_url": DOTNET_EXECUTE_URL,
        }
    )

    delays = [1.0, 2.0, 4.0]
    max_attempts = len(delays) + 1
    attempt = 0
    last_error = None
    saw_timeout = False
    start = time.time()

    while attempt < max_attempts:
        if attempt > 0:
            delay = delays[attempt - 1]
            logger.info(
                {
                    "message": "Retrying .NET call",
                    "trace_id": effective_trace_id,
                    "delay_sec": delay,
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                }
            )
            time.sleep(delay)

        attempt += 1
        try:
            resp = _dotnet_session.post(
                DOTNET_EXECUTE_URL,
                json=payload,
                headers=headers,
                timeout=(DOTNET_CONNECT_TIMEOUT, DOTNET_READ_TIMEOUT),
                verify=DOTNET_VERIFY_SSL,
            )

            # Success
            if resp.status_code < 400:
                text_content = (resp.text or "").strip()
                if not text_content or text_content == "null":
                    response_body = {
                        "success": True,
                        "data": [],
                        "message": "No data returned.",
                    }
                else:
                    response_body = resp.json()
                records = _extract_records(response_body)
                logger.info(
                    {
                        "trace_id": effective_trace_id,
                        "session_id": effective_session_id,
                        "query_type": query_type or "unknown",
                        "status_code": resp.status_code,
                        "duration_ms": round((time.time() - start) * 1000, 2),
                        "record_count": len(records),
                    }
                )
                _cb.record_success()
                return response_body, None

            # Handle HTTP errors
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text[:400]

            last_error = f"Backend returned HTTP {resp.status_code}: {err_body}"
            logger.info(
                {
                    "trace_id": effective_trace_id,
                    "session_id": effective_session_id,
                    "query_type": query_type or "unknown",
                    "status_code": resp.status_code,
                    "duration_ms": round((time.time() - start) * 1000, 2),
                    "record_count": (
                        len(_extract_records(err_body))
                        if isinstance(err_body, (dict, list))
                        else 0
                    ),
                }
            )

            # Only retry on 429, 502, 503, 504
            if resp.status_code in (429, 502, 503, 504):
                logger.warning(
                    {
                        "message": "Transient HTTP error. Will retry.",
                        "trace_id": effective_trace_id,
                        "status_code": resp.status_code,
                    }
                )
                continue
            elif resp.status_code >= 500:
                # Non-retriable server errors (e.g., 500) count as circuit breaker failures
                logger.error(
                    {
                        "message": "Non-retriable server error",
                        "trace_id": effective_trace_id,
                        "status_code": resp.status_code,
                    }
                )
                _cb.record_failure()
                metrics_collector.inc_dotnet_failure()
                return None, last_error
            else:
                # Validation or client errors (e.g., 400, 422) do not retry and do not count as system failures
                logger.info(
                    {
                        "message": "Non-retriable client error",
                        "trace_id": effective_trace_id,
                        "status_code": resp.status_code,
                    }
                )
                return None, last_error

        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = (
                f"Cannot connect to .NET backend at {DOTNET_EXECUTE_URL}. ({exc})"
            )
            if isinstance(exc, requests.Timeout):
                saw_timeout = True
            logger.warning(
                {
                    "message": "Transient connection or timeout error. Will retry.",
                    "trace_id": effective_trace_id,
                    "error": str(exc),
                }
            )
            continue
        except requests.RequestException as exc:
            # Non-transient or generic requests exception
            last_error = f"Backend request failed: {exc}"
            logger.error(
                json.dumps(
                    {
                        "message": "Non-retriable RequestException",
                        "trace_id": effective_trace_id,
                        "error": str(exc),
                    }
                )
            )
            _cb.record_failure()
            metrics_collector.inc_dotnet_failure()
            return None, last_error
        except ValueError as exc:
            # JSON decoding error
            last_error = f"Backend returned invalid JSON: {exc}"
            logger.error(
                json.dumps(
                    {
                        "message": "Non-retriable JSON parse failure",
                        "trace_id": effective_trace_id,
                        "error": str(exc),
                    }
                )
            )
            _cb.record_failure()
            metrics_collector.inc_dotnet_failure()
            return None, last_error

    # If all retries failed due to transient/connection issues
    _cb.record_failure()
    if saw_timeout:
        metrics_collector.inc_timeout_failure()
    else:
        metrics_collector.inc_dotnet_failure()
    return None, last_error

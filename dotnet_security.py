"""
dotnet_security.py
===================
Shared TLS verification policy for .NET backend calls.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _is_truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_dotnet_verify_ssl(log: Optional[logging.Logger] = None) -> bool:
    """
    Resolve whether TLS verification should remain enabled.

    Secure default:
      - no env override => True
      - DOTNET_SKIP_SSL_VERIFY=1 => False and warn
      - DOTNET_VERIFY_SSL can still be used as an explicit compatibility override
    """
    skip_raw = os.getenv("DOTNET_SKIP_SSL_VERIFY")
    verify_raw = os.getenv("DOTNET_VERIFY_SSL")

    if skip_raw is not None:
        verify_ssl = not _is_truthy(skip_raw)
        if not verify_ssl:
            (log or logger).warning(
                "DOTNET TLS verification disabled via DOTNET_SKIP_SSL_VERIFY; "
                "this should only be used in local development."
            )
        return verify_ssl

    if verify_raw is not None:
        return _is_truthy(verify_raw)

    return True

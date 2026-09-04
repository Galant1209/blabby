"""Optional, privacy-bounded PostHog capture for backend truth events.

This module deliberately has no startup requirement.  Payment must remain
usable when the optional capture key is absent or when PostHog is unavailable.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from typing import Any

import requests


logger = logging.getLogger(__name__)

POSTHOG_CAPTURE_KEY_ENV = "POSTHOG_CAPTURE_API_KEY"
POSTHOG_HOST_ENV = "POSTHOG_HOST"
DEFAULT_POSTHOG_HOST = "https://us.i.posthog.com"

# Keep this allowlist intentionally smaller than PostHog's arbitrary property
# surface.  In particular, no user profile, request, callback, or credential
# field is accepted here.
SAFE_PROPERTY_KEYS = frozenset({
    "amount_twd",
    "currency",
    "entitlement_active",
    "failure_stage",
    "http_status",
    "order_id",
    "payment_provider",
    "plan",
    "source",
    "stage",
    "subscription_status",
})

SAFE_EVENT_NAMES = frozenset({
    "checkout_order_created",
    "checkout_success_viewed",
    "subscription_activated",
})

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")
_SAFE_ORDER_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _safe_properties(properties: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only bounded, scalar properties from the explicit allowlist."""
    if not isinstance(properties, Mapping):
        return {}

    safe: dict[str, Any] = {}
    for key, value in properties.items():
        if key not in SAFE_PROPERTY_KEYS:
            continue
        if isinstance(value, bool):
            safe[key] = value
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            if key == "http_status" and 100 <= value <= 599:
                safe[key] = value
            elif key == "amount_twd" and 1 <= value <= 1_000_000:
                safe[key] = value
            continue
        if not isinstance(value, str) or len(value) > 100:
            continue
        if key == "order_id":
            if _SAFE_ORDER_ID.fullmatch(value):
                safe[key] = value
        elif _SAFE_IDENTIFIER.fullmatch(value):
            safe[key] = value
    return safe


def capture_event(
    event: str,
    *,
    distinct_id: str,
    properties: Mapping[str, Any] | None = None,
) -> bool:
    """Best-effort PostHog capture; never raises into a product request.

    ``POSTHOG_CAPTURE_API_KEY`` is intentionally separate from the personal
    query/admin credential.  No credential is required for local startup or
    for payment operation.
    """
    if event not in SAFE_EVENT_NAMES:
        logger.warning("[telemetry] rejected event name=%s", event)
        return False
    if not isinstance(distinct_id, str) or not distinct_id.strip():
        logger.warning("[telemetry] skipped event without distinct_id=%s", event)
        return False

    api_key = (os.getenv(POSTHOG_CAPTURE_KEY_ENV) or "").strip()
    if not api_key:
        return False

    host = (os.getenv(POSTHOG_HOST_ENV) or DEFAULT_POSTHOG_HOST).strip().rstrip("/")
    payload = {
        "api_key": api_key,
        "event": event,
        "distinct_id": distinct_id.strip(),
        "properties": _safe_properties(properties),
    }

    try:
        response = requests.post(
            f"{host}/capture/",
            json=payload,
            timeout=1.5,
        )
        if response.status_code >= 400:
            logger.warning(
                "[telemetry] PostHog rejected event=%s status=%s",
                event,
                response.status_code,
            )
            return False
        return True
    except Exception:
        logger.warning("[telemetry] PostHog capture failed event=%s", event,
                       exc_info=True)
        return False

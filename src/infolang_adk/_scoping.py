"""Maps ADK's ``(app_name, user_id)`` memory scope onto an InfoLang namespace.

InfoLang namespaces are a single flat string per bank. ADK's
:class:`~google.adk.memory.base_memory_service.BaseMemoryService` scopes memory
two-dimensionally, by application and user. This module collapses that pair
into one deterministic namespace string.

The default scheme is intentionally simple and lossy-safe: it sanitizes both
segments to a conservative charset and joins them with a prefix, so two
different ``(app_name, user_id)`` pairs collide only if their *sanitized*
forms are identical (e.g. ``"team/a"`` and ``"team a"`` both sanitize to
``"team-a"``). If your ``app_name``/``user_id`` values can only differ in
characters this scheme treats as equivalent, supply a custom
``namespace_for`` callable to :class:`~infolang_adk.InfoLangMemoryService`.
"""

from __future__ import annotations

import re

DEFAULT_NAMESPACE_PREFIX = "adk"

# InfoLang namespaces are opaque strings; the runtime API does not publish a
# formal charset restriction. This sanitizer keeps namespaces readable and
# URL/query-string safe (recall/list_recent send namespace as a URL query
# parameter) by keeping alphanumerics, dash, underscore, and dot, and
# replacing everything else (including "/" and ":", which InfoLang uses
# internally in some contexts) with a dash.
_UNSAFE_CHARS = re.compile(r"[^a-zA-Z0-9_.-]+")


def _sanitize_segment(segment: str) -> str:
    cleaned = _UNSAFE_CHARS.sub("-", segment.strip()).strip("-")
    return cleaned or "unknown"


def default_namespace_for(
    app_name: str, user_id: str, *, prefix: str = DEFAULT_NAMESPACE_PREFIX
) -> str:
    """Builds the default InfoLang namespace for an ADK ``(app_name, user_id)`` pair.

    Format: ``"{prefix}-{sanitized app_name}-{sanitized user_id}"``. Every
    memory service instance uses one namespace per distinct application/user
    pair, so recall/remember calls for one user never see another user's (or
    another app's) memories, and clearing one user's memory
    (``namespace``-scoped) never touches another's.
    """

    return f"{prefix}-{_sanitize_segment(app_name)}-{_sanitize_segment(user_id)}"

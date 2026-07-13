"""InfoLang memory service for Google ADK agents.

Quickstart::

    from google.adk.runners import Runner
    from infolang_adk import InfoLangMemoryService

    memory_service = InfoLangMemoryService(api_key="il_live_...")
    runner = Runner(
        agent=my_agent,
        app_name="my_app",
        session_service=my_session_service,
        memory_service=memory_service,
    )
"""

from __future__ import annotations

from ._scoping import DEFAULT_NAMESPACE_PREFIX, default_namespace_for
from ._version import __version__
from .memory_service import InfoLangMemoryService

__all__ = [
    "__version__",
    "InfoLangMemoryService",
    "default_namespace_for",
    "DEFAULT_NAMESPACE_PREFIX",
]

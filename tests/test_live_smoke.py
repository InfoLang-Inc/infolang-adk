"""Optional live smoke test against the real InfoLang API.

Skipped unless ``INFOLANG_API_KEY`` is set. Only ever writes to namespaces
prefixed ``ittest-adk-`` and deletes everything it wrote before returning,
whether the test passes or fails.

Not run by default `pytest` (see the module-level skip below); intended for
manual verification, e.g.::

    INFOLANG_API_KEY=il_live_... pytest tests/test_live_smoke.py -q
"""

from __future__ import annotations

import os
import uuid

import pytest
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types

from infolang_adk import InfoLangMemoryService

pytestmark = pytest.mark.skipif(
    not os.environ.get("INFOLANG_API_KEY"),
    reason="live smoke test requires INFOLANG_API_KEY",
)


@pytest.fixture
def live_scope() -> tuple[str, str]:
    # Unique per run so concurrent CI runs (if any) never collide, and so a
    # crashed prior run's leftovers can't cause a false pass/fail here.
    run_id = uuid.uuid4().hex[:8]
    return f"ittest-adk-app-{run_id}", f"user-{run_id}"


async def test_live_add_and_search_round_trip(live_scope: tuple[str, str]) -> None:
    app_name, user_id = live_scope
    svc = InfoLangMemoryService(namespace_prefix="ittest-adk")
    namespace = svc._default_namespace_for(app_name, user_id)  # noqa: SLF001
    assert namespace.startswith("ittest-adk-")

    try:
        session = Session(
            id="live-session",
            app_name=app_name,
            user_id=user_id,
            events=[
                Event(
                    author="user",
                    content=types.Content(
                        parts=[types.Part(text="InfoLang smoke test canary fact.")],
                        role="user",
                    ),
                )
            ],
        )
        await svc.add_session_to_memory(session)

        response = await svc.search_memory(
            app_name=app_name, user_id=user_id, query="smoke test canary fact"
        )
        assert any("canary" in (m.content.parts[0].text or "") for m in response.memories)
    finally:
        # Best-effort cleanup: forget everything remembered in this namespace.
        await svc._client.reset_namespace(namespace)  # noqa: SLF001
        await svc.aclose()

from __future__ import annotations

import json

import httpx
import respx
from google.adk.events.event import Event
from google.genai import types

from infolang_adk import InfoLangMemoryService
from tests.conftest import BASE_URL


def _text_event(author: str, text: str) -> Event:
    return Event(author=author, content=types.Content(parts=[types.Part(text=text)], role="user"))


@respx.mock
async def test_add_events_to_memory_sends_delta_with_custom_metadata(
    service: InfoLangMemoryService,
) -> None:
    route = respx.post(f"{BASE_URL}/v1/execute").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"op": "remember_batch", "ok": True, "payload": {"results": [{"id": "m1"}]}}
                ]
            },
        )
    )

    await service.add_events_to_memory(
        app_name="myapp",
        user_id="u1",
        events=[_text_event("user", "remember this")],
        session_id="sess-9",
        custom_metadata={"channel": "slack", "priority": 3, "nested": {"a": 1}},
    )

    assert route.called
    body = json.loads(route.calls.last.request.content)
    items = body["operations"][0]["args"]["items"]
    assert len(items) == 1
    tags = items[0]["tags"]
    assert "adk-author:user" in tags
    assert "adk-session:sess-9" in tags
    assert "channel:slack" in tags
    assert "priority:3" in tags
    # Non-scalar custom_metadata values are dropped, not serialized.
    assert not any(t.startswith("nested:") for t in tags)


@respx.mock
async def test_add_events_to_memory_no_session_id_omits_session_tag(
    service: InfoLangMemoryService,
) -> None:
    route = respx.post(f"{BASE_URL}/v1/execute").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"op": "remember_batch", "ok": True, "payload": {"results": [{"id": "m1"}]}}
                ]
            },
        )
    )

    await service.add_events_to_memory(
        app_name="myapp", user_id="u1", events=[_text_event("user", "hi")]
    )

    body = json.loads(route.calls.last.request.content)
    tags = body["operations"][0]["args"]["items"][0]["tags"]
    assert not any(t.startswith("adk-session:") for t in tags)


@respx.mock
async def test_add_events_to_memory_empty_events_is_noop(
    service: InfoLangMemoryService,
) -> None:
    route = respx.post(f"{BASE_URL}/v1/execute")
    await service.add_events_to_memory(app_name="myapp", user_id="u1", events=[])
    assert not route.called

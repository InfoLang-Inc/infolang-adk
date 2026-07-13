from __future__ import annotations

import json

import httpx
import respx
from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types

from infolang_adk import InfoLangMemoryService
from tests.conftest import BASE_URL


def _text_event(author: str, text: str) -> Event:
    return Event(author=author, content=types.Content(parts=[types.Part(text=text)], role="user"))


def _empty_event(author: str = "tool") -> Event:
    return Event(
        author=author,
        content=types.Content(
            parts=[types.Part(function_call=types.FunctionCall(name="lookup"))], role="model"
        ),
    )


def _no_content_event(author: str = "system") -> Event:
    return Event(author=author)


@respx.mock
async def test_add_session_to_memory_batches_text_events(
    service: InfoLangMemoryService,
) -> None:
    route = respx.post(f"{BASE_URL}/v1/execute").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "op": "remember_batch",
                        "ok": True,
                        "payload": {"results": [{"id": "m1"}, {"id": "m2"}]},
                    }
                ]
            },
        )
    )

    session = Session(
        id="sess-1",
        app_name="myapp",
        user_id="u1",
        events=[
            _text_event("user", "what is InfoLang?"),
            _empty_event(),
            _no_content_event(),
            _text_event("assistant", "InfoLang is a memory API."),
        ],
    )

    await service.add_session_to_memory(session)

    assert route.called
    body = json.loads(route.calls.last.request.content)
    op = body["operations"][0]
    assert op["op"] == "remember_batch"
    assert op["args"]["namespace"] == "adk-myapp-u1"
    items = op["args"]["items"]
    # The function-call-only event and the content-less event both have no
    # text and are dropped.
    assert len(items) == 2
    assert items[0]["text"] == "what is InfoLang?"
    assert "adk-author:user" in items[0]["tags"]
    assert "adk-session:sess-1" in items[0]["tags"]
    assert items[1]["source"] == "assistant"


@respx.mock
async def test_add_session_to_memory_with_no_text_events_is_noop(
    service: InfoLangMemoryService,
) -> None:
    route = respx.post(f"{BASE_URL}/v1/execute")
    session = Session(id="sess-2", app_name="myapp", user_id="u1", events=[_empty_event()])

    await service.add_session_to_memory(session)

    assert not route.called

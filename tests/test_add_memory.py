from __future__ import annotations

import json

import httpx
import pytest
import respx
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types

from infolang_adk import InfoLangMemoryService
from tests.conftest import BASE_URL


def _entry(text: str, author: str | None = None, **metadata: object) -> MemoryEntry:
    return MemoryEntry(
        content=types.Content(parts=[types.Part(text=text)], role="user"),
        author=author,
        custom_metadata=dict(metadata),
    )


@respx.mock
async def test_add_memory_ingests_explicit_entries(service: InfoLangMemoryService) -> None:
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

    await service.add_memory(
        app_name="myapp",
        user_id="u1",
        memories=[
            _entry("fact one", author="user", topic="billing"),
            _entry("fact two"),
        ],
        custom_metadata={"source": "import"},
    )

    assert route.called
    body = json.loads(route.calls.last.request.content)
    items = body["operations"][0]["args"]["items"]
    assert body["operations"][0]["args"]["namespace"] == "adk-myapp-u1"
    assert items[0]["text"] == "fact one"
    assert "adk-author:user" in items[0]["tags"]
    assert "topic:billing" in items[0]["tags"]
    assert "source:import" in items[0]["tags"]
    # Second entry has no author -> no author tag, but still gets the shared
    # custom_metadata tag.
    assert not any(t.startswith("adk-author:") for t in items[1]["tags"])
    assert "source:import" in items[1]["tags"]


async def test_add_memory_empty_list_raises(service: InfoLangMemoryService) -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        await service.add_memory(app_name="myapp", user_id="u1", memories=[])


async def test_add_memory_entry_without_text_raises(service: InfoLangMemoryService) -> None:
    blank = MemoryEntry(content=types.Content(parts=[types.Part(text="   ")], role="user"))
    with pytest.raises(ValueError, match="non-whitespace text"):
        await service.add_memory(app_name="myapp", user_id="u1", memories=[blank])


async def test_add_memory_entry_with_no_parts_raises(service: InfoLangMemoryService) -> None:
    no_parts = MemoryEntry(content=types.Content(role="user"))
    with pytest.raises(ValueError, match="non-whitespace text"):
        await service.add_memory(app_name="myapp", user_id="u1", memories=[no_parts])

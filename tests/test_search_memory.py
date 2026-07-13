from __future__ import annotations

import httpx
import pytest
import respx
from infolang import RateLimitError

from infolang_adk import InfoLangMemoryService
from tests.conftest import BASE_URL


@respx.mock
async def test_search_memory_maps_chunks_to_memory_entries(
    service: InfoLangMemoryService,
) -> None:
    route = respx.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(
            200,
            json={
                "chunks": [
                    {"i": "c1", "s": 0.91, "t": "auth uses bearer tokens", "g": "auth,security"},
                    {"i": "c2", "s": 0.55, "t": "unrelated fact"},
                ]
            },
        )
    )

    response = await service.search_memory(
        app_name="myapp", user_id="u1", query="how does auth work?"
    )

    assert route.called
    sent_body = route.calls.last.request.content
    assert b'"namespace":"adk-myapp-u1"' in sent_body.replace(b" ", b"")
    assert len(response.memories) == 2
    first = response.memories[0]
    assert first.id == "c1"
    assert first.content.parts[0].text == "auth uses bearer tokens"
    assert first.custom_metadata["score"] == 0.91
    assert first.custom_metadata["tags"] == "auth,security"
    assert "tags" not in response.memories[1].custom_metadata


@respx.mock
async def test_search_memory_chunk_without_score_or_tags_has_empty_metadata(
    service: InfoLangMemoryService,
) -> None:
    respx.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"chunks": [{"i": "c1", "t": "no score or tags"}]})
    )

    response = await service.search_memory(app_name="myapp", user_id="u1", query="q")

    assert response.memories[0].custom_metadata == {}


@respx.mock
async def test_search_memory_respects_top_k() -> None:
    route = respx.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(200, json={"chunks": []})
    )
    svc = InfoLangMemoryService(api_key="il_live_test", base_url=BASE_URL, search_top_k=3)
    try:
        await svc.search_memory(app_name="app", user_id="u", query="q")
    finally:
        await svc.aclose()

    sent_body = route.calls.last.request.content.replace(b" ", b"")
    assert b'"top_k":3' in sent_body


@respx.mock
async def test_search_memory_filters_by_min_score() -> None:
    respx.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(
            200,
            json={
                "chunks": [
                    {"i": "high", "s": 0.9, "t": "confident"},
                    {"i": "low", "s": 0.4, "t": "weak"},
                ]
            },
        )
    )
    svc = InfoLangMemoryService(api_key="il_live_test", base_url=BASE_URL, min_score=0.85)
    try:
        response = await svc.search_memory(app_name="app", user_id="u", query="q")
    finally:
        await svc.aclose()

    assert [m.id for m in response.memories] == ["high"]


@respx.mock
async def test_search_memory_missing_namespace_returns_empty(
    service: InfoLangMemoryService,
) -> None:
    respx.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(404, json={"error": "namespace not found"})
    )

    response = await service.search_memory(app_name="app", user_id="new-user", query="q")

    assert response.memories == []


@respx.mock
async def test_search_memory_propagates_other_api_errors() -> None:
    respx.post(f"{BASE_URL}/v1/recall").mock(
        return_value=httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
    )
    # max_retries=0 keeps this deterministic and fast: 429 is in the SDK's
    # retry set, so leaving retries on would sleep for retry-after between
    # attempts before finally raising.
    svc = InfoLangMemoryService(api_key="il_live_test", base_url=BASE_URL, max_retries=0)
    try:
        with pytest.raises(RateLimitError):
            await svc.search_memory(app_name="app", user_id="u1", query="q")
    finally:
        await svc.aclose()

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx
from infolang import AsyncInfoLang

from infolang_adk import InfoLangMemoryService
from tests.conftest import BASE_URL, WORKSPACE, WS_URL


def test_client_and_kwargs_are_mutually_exclusive() -> None:
    client = AsyncInfoLang(api_key="il_live_test", base_url=BASE_URL, workspace=WORKSPACE)
    try:
        with pytest.raises(ValueError, match="not both"):
            InfoLangMemoryService(client=client, api_key="il_live_test")
    finally:
        # Construction failed before the service took ownership; close directly.
        asyncio.run(client.aclose())


async def test_aclose_closes_owned_client() -> None:
    svc = InfoLangMemoryService(api_key="il_live_test", base_url=BASE_URL, workspace=WORKSPACE)
    await svc.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await svc._client.recall("q")  # noqa: SLF001 -- black-box "did close propagate" check


async def test_aclose_does_not_close_externally_owned_client() -> None:
    client = AsyncInfoLang(api_key="il_live_test", base_url=BASE_URL, workspace=WORKSPACE)
    svc = InfoLangMemoryService(client=client)
    await svc.aclose()
    # aclose() on a service constructed from an externally-owned client must
    # be a no-op: a request through that client should still go out (not
    # raise "client has been closed") after svc.aclose().
    with respx.mock:
        respx.post(f"{WS_URL}/recall").mock(return_value=httpx.Response(200, json={"hits": []}))
        result = await client.recall("q")
        assert result.chunks == []
    await client.aclose()


@respx.mock
async def test_custom_namespace_for_overrides_default_scoping() -> None:
    route = respx.post(f"{WS_URL}/recall").mock(return_value=httpx.Response(200, json={"hits": []}))
    svc = InfoLangMemoryService(
        api_key="il_live_test",
        base_url=BASE_URL,
        workspace=WORKSPACE,
        namespace_for=lambda app_name, user_id: f"custom-{app_name}-{user_id}",
    )
    try:
        await svc.search_memory(app_name="myapp", user_id="u1", query="q")
    finally:
        await svc.aclose()

    sent = route.calls.last.request.content.replace(b" ", b"")
    assert b'"namespace":"custom-myapp-u1"' in sent


async def test_async_context_manager_closes_owned_client() -> None:
    async with InfoLangMemoryService(
        api_key="il_live_test", base_url=BASE_URL, workspace=WORKSPACE
    ) as svc:
        client = svc._client  # noqa: SLF001
    with pytest.raises(RuntimeError, match="closed"):
        await client.recall("q")

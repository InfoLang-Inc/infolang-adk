from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from infolang_adk import InfoLangMemoryService

BASE_URL = "https://api.test.infolang.ai"

# Every gateway call is workspace-scoped (``/v2/workspaces/{ws}/...``). Tests
# pass ``workspace=`` explicitly so no fixture has to mock ``GET /v2/whoami``.
WORKSPACE = "ws_test"
WS_URL = f"{BASE_URL}/v2/workspaces/{WORKSPACE}"


def execute_ok(*memory_ids: str) -> dict[str, Any]:
    """A native OpResult envelope for an ``execute`` of N ``remember`` sub-ops.

    ``remember_batch`` is client-side sugar in the SDK: it POSTs one
    ``{"op": "remember"}`` sub-op per item to ``.../execute`` and reads the
    per-sub-op results back out of ``payload.results``.
    """

    return {
        "ok": True,
        "payload": {
            "results": [
                {"op": "remember", "ok": True, "payload": {"id": memory_id}}
                for memory_id in memory_ids
            ]
        },
    }


def remember_ops(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The ``args`` of each ``remember`` sub-op in an ``execute`` request body."""

    operations = body["operations"]
    assert all(op["op"] == "remember" for op in operations), operations
    return [op["args"] for op in operations]


@pytest.fixture
async def service() -> AsyncIterator[InfoLangMemoryService]:
    svc = InfoLangMemoryService(api_key="il_live_test", base_url=BASE_URL, workspace=WORKSPACE)
    yield svc
    await svc.aclose()

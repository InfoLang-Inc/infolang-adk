from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from infolang_adk import InfoLangMemoryService

BASE_URL = "https://api.test.infolang.ai"


@pytest.fixture
async def service() -> AsyncIterator[InfoLangMemoryService]:
    svc = InfoLangMemoryService(api_key="il_live_test", base_url=BASE_URL)
    yield svc
    await svc.aclose()

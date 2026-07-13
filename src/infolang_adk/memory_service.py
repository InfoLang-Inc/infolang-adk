"""``InfoLangMemoryService``: a real :class:`BaseMemoryService` implementation.

Backs Google ADK's pluggable agent memory with InfoLang recall/remember.
Every method the current ``google-adk`` ``BaseMemoryService`` ABC defines is
implemented against InfoLang -- including ``add_events_to_memory`` and
``add_memory``, which the base class otherwise leaves raising
``NotImplementedError``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from google.adk.memory.base_memory_service import (
    BaseMemoryService,
    SearchMemoryResponse,
)
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types as genai_types
from infolang import AsyncInfoLang
from infolang.errors import InfoLangAPIError, NotFoundError
from infolang.types import Chunk
from typing_extensions import override

from ._scoping import DEFAULT_NAMESPACE_PREFIX, default_namespace_for

# Event/Session are only used for type hints. Importing them eagerly at
# module scope works fine on the currently-installed google-adk, but this
# keeps the runtime import surface minimal and annotations stay valid via
# `from __future__ import annotations`.
if TYPE_CHECKING:
    from google.adk.events.event import Event
    from google.adk.sessions.session import Session

_log = logging.getLogger("infolang_adk")

NamespaceForFn = Callable[[str, str], str]


class InfoLangMemoryService(BaseMemoryService):
    """Google ADK ``BaseMemoryService`` backed by the InfoLang memory API.

    Maps ADK's ``(app_name, user_id)`` memory scope onto a single InfoLang
    namespace per pair (see :mod:`infolang_adk._scoping`), and implements
    every method the installed ``BaseMemoryService`` ABC exposes:

    - ``search_memory`` -- InfoLang ``recall`` scoped to the caller's namespace.
    - ``add_session_to_memory`` -- ingests a full session's events via
      ``remember_batch``.
    - ``add_events_to_memory`` -- ingests an incremental delta of events via
      ``remember_batch`` (does not require re-ingesting the whole session).
    - ``add_memory`` -- ingests explicit ``MemoryEntry`` items directly via
      ``remember_batch``, bypassing session/event framing entirely.

    Construct with either an existing :class:`infolang.AsyncInfoLang` client
    (``client=``) or the kwargs to build one (``api_key=``, ``base_url=``,
    etc. -- forwarded verbatim to ``AsyncInfoLang(...)``), not both.
    """

    def __init__(
        self,
        *,
        client: AsyncInfoLang | None = None,
        namespace_prefix: str = DEFAULT_NAMESPACE_PREFIX,
        namespace_for: NamespaceForFn | None = None,
        search_top_k: int = 10,
        min_score: float | None = None,
        **client_kwargs: Any,
    ) -> None:
        """Initializes the service.

        Args:
          client: A pre-built ``AsyncInfoLang`` client to reuse. Mutually
            exclusive with ``**client_kwargs``.
          namespace_prefix: Prefix used by the default namespace mapping.
            Ignored if ``namespace_for`` is supplied.
          namespace_for: Optional override mapping ``(app_name, user_id) ->
            namespace``. Defaults to
            :func:`infolang_adk._scoping.default_namespace_for` bound to
            ``namespace_prefix``.
          search_top_k: Default number of chunks requested per
            ``search_memory`` call.
          min_score: If set, chunks scoring below this InfoLang similarity
            score are dropped from ``search_memory`` results client-side.
          **client_kwargs: Forwarded to ``AsyncInfoLang(...)`` when ``client``
            is not given (e.g. ``api_key``, ``base_url``, ``workspace``).

        Raises:
          ValueError: If both ``client`` and ``client_kwargs`` are given.
        """

        if client is not None and client_kwargs:
            raise ValueError(
                "Pass either client=<AsyncInfoLang instance> or client "
                "construction kwargs (api_key=, base_url=, ...), not both."
            )
        self._owns_client = client is None
        self._client: AsyncInfoLang = (
            client if client is not None else AsyncInfoLang(**client_kwargs)
        )
        self._namespace_prefix = namespace_prefix
        self._namespace_for: NamespaceForFn = namespace_for or self._default_namespace_for
        self._search_top_k = search_top_k
        self._min_score = min_score

    def _default_namespace_for(self, app_name: str, user_id: str) -> str:
        return default_namespace_for(app_name, user_id, prefix=self._namespace_prefix)

    # -- required ABC methods -------------------------------------------------

    @override
    async def search_memory(
        self, *, app_name: str, user_id: str, query: str
    ) -> SearchMemoryResponse:
        """Recalls memories for ``query`` scoped to ``(app_name, user_id)``.

        Semantics: this is a single InfoLang ``recall`` call (IL-cosine
        ranked), not free-text/keyword search. A namespace that has never
        been written to (no prior ``add_session_to_memory``/``add_memory``
        call for this ``app_name``/``user_id``) returns an empty response
        rather than raising. Any other InfoLang API error propagates to the
        caller.
        """

        namespace = self._namespace_for(app_name, user_id)
        try:
            result = await self._client.recall(query, namespace=namespace, top_k=self._search_top_k)
        except NotFoundError:
            _log.debug("infolang_adk: namespace %s not found; returning no memories.", namespace)
            return SearchMemoryResponse()
        except InfoLangAPIError:
            _log.exception("infolang_adk: recall failed for namespace=%s", namespace)
            raise

        memories = [
            _chunk_to_memory_entry(chunk)
            for chunk in result.chunks
            if self._min_score is None or chunk.score is None or chunk.score >= self._min_score
        ]
        return SearchMemoryResponse(memories=memories)

    @override
    async def add_session_to_memory(self, session: Session) -> None:
        """Ingests every event in ``session`` into InfoLang.

        Re-ingesting the same session (e.g. after new turns) is safe: each
        call issues a fresh ``remember_batch``, so InfoLang accumulates one
        memory per (re-)ingested event rather than deduplicating or updating
        in place. Callers who only want the newest turns should use
        ``add_events_to_memory`` instead of repeatedly re-ingesting the whole
        session.
        """

        namespace = self._namespace_for(session.app_name, session.user_id)
        items = _events_to_remember_items(session.events, session_id=session.id)
        if not items:
            return
        await self._client.remember_batch(items, namespace=namespace)

    # -- optional ABC methods, implemented for real ----------------------------

    @override
    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Ingests an explicit delta of events, without the full session.

        ``custom_metadata`` entries with scalar values (``str``/``int``/
        ``float``/``bool``) are recorded as InfoLang tags (``"key:value"``);
        non-scalar values are dropped (InfoLang tags are strings, there is no
        structured-metadata field on ``remember``).
        """

        namespace = self._namespace_for(app_name, user_id)
        items = _events_to_remember_items(
            events, session_id=session_id, custom_metadata=custom_metadata
        )
        if not items:
            return
        await self._client.remember_batch(items, namespace=namespace)

    @override
    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[MemoryEntry],
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Ingests explicit ``MemoryEntry`` items directly, bypassing events.

        Every entry must carry non-whitespace text content; entries with no
        text (audio/image-only parts, etc.) raise ``ValueError`` since
        InfoLang only stores text.
        """

        namespace = self._namespace_for(app_name, user_id)
        items = _memory_entries_to_remember_items(memories, custom_metadata=custom_metadata)
        await self._client.remember_batch(items, namespace=namespace)

    # -- lifecycle --------------------------------------------------------------

    async def aclose(self) -> None:
        """Closes the underlying InfoLang client, if this service owns it."""

        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> InfoLangMemoryService:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


# -- conversion helpers --------------------------------------------------------


def _chunk_to_memory_entry(chunk: Chunk) -> MemoryEntry:
    metadata: dict[str, Any] = {}
    if chunk.score is not None:
        metadata["score"] = chunk.score
    if chunk.tags:
        metadata["tags"] = chunk.tags
    return MemoryEntry(
        id=chunk.id,
        content=genai_types.Content(parts=[genai_types.Part(text=chunk.text)], role="user"),
        custom_metadata=metadata,
    )


def _event_text(event: Event) -> str | None:
    if not event.content or not event.content.parts:
        return None
    parts = [part.text.strip() for part in event.content.parts if part.text]
    parts = [part for part in parts if part]
    if not parts:
        return None
    return "\n".join(parts)


def _scalar_tags(metadata: Mapping[str, object] | None) -> list[str]:
    if not metadata:
        return []
    return [
        f"{key}:{value}"
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    ]


def _events_to_remember_items(
    events: Sequence[Event],
    *,
    session_id: str | None,
    custom_metadata: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    extra_tags = _scalar_tags(custom_metadata)
    items: list[dict[str, Any]] = []
    for event in events:
        text = _event_text(event)
        if text is None:
            continue
        tags = [f"adk-author:{event.author or 'unknown'}"]
        if session_id:
            tags.append(f"adk-session:{session_id}")
        tags.extend(extra_tags)
        items.append({"text": text, "source": event.author or None, "tags": tags})
    return items


def _memory_entry_text(memory: MemoryEntry) -> str | None:
    if not memory.content or not memory.content.parts:
        return None
    parts = [part.text.strip() for part in memory.content.parts if part.text]
    parts = [part for part in parts if part]
    if not parts:
        return None
    return "\n".join(parts)


def _memory_entries_to_remember_items(
    memories: Sequence[MemoryEntry],
    *,
    custom_metadata: Mapping[str, object] | None = None,
) -> list[dict[str, Any]]:
    if not memories:
        raise ValueError("memories must contain at least one entry.")

    extra_tags = _scalar_tags(custom_metadata)
    items: list[dict[str, Any]] = []
    for index, memory in enumerate(memories):
        text = _memory_entry_text(memory)
        if text is None:
            raise ValueError(f"memories[{index}] must include non-whitespace text content.")
        tags = [f"adk-author:{memory.author}"] if memory.author else []
        tags.extend(_scalar_tags(memory.custom_metadata))
        tags.extend(extra_tags)
        items.append({"text": text, "source": memory.author, "tags": tags})
    return items

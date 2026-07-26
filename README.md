# infolang-adk

InfoLang memory for [Google ADK](https://google.github.io/adk-docs/) agents:
`InfoLangMemoryService`, a `google.adk.memory.BaseMemoryService` implementation
backed by the [InfoLang](https://infolang.ai) memory API.

## Why this exists

ADK's `BaseMemoryService` ABC declares two required async methods
(`search_memory`, `add_session_to_memory`) and two optional ones
(`add_events_to_memory`, `add_memory`) that default to raising
`NotImplementedError`. Most third-party memory integrations for agent
frameworks wrap their SDK as a function-tool the LLM calls, rather than
implementing the framework's actual memory-service interface — see
[mem0ai/mem0#3999](https://github.com/mem0ai/mem0/issues/3999), where the
maintainers confirm their ADK integration does exactly this. That approach
skips `Runner`-level memory wiring (auto session ingestion, `load_memory`
tool auto-registration, etc.) entirely.

`infolang-adk` implements the real interface: all four methods, including
the two optional ones, backed by real InfoLang calls, not stubs.

## Install

```bash
pip install infolang-adk
```

Requires Python 3.11+, `google-adk>=1.0.0`, and an InfoLang API key.

## Quickstart

```python
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from infolang_adk import InfoLangMemoryService

memory_service = InfoLangMemoryService(api_key="il_live_...")

agent = Agent(
    name="assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful assistant with long-term memory.",
)

runner = Runner(
    agent=agent,
    app_name="my_app",
    session_service=InMemorySessionService(),
    memory_service=memory_service,
)
```

`Runner` calls `add_session_to_memory` when a session ends and exposes
`search_memory` to agents that use ADK's built-in `load_memory` tool (or call
it directly, e.g. from a `before_agent_callback`). See `examples/` for a
runnable end-to-end agent.

Credentials resolve the same way as the `infolang` SDK: pass `api_key=`
explicitly, or set the `INFOLANG_API_KEY` environment variable and construct
`InfoLangMemoryService()` with no arguments. Any keyword `InfoLangMemoryService`
doesn't recognize (`base_url=`, `dev_key=`, `workspace=`, ...) is forwarded
verbatim to `infolang.AsyncInfoLang(...)`; see that class for the full list.

To reuse an existing client instead:

```python
from infolang import AsyncInfoLang
from infolang_adk import InfoLangMemoryService

client = AsyncInfoLang(api_key="il_live_...")
memory_service = InfoLangMemoryService(client=client)
# ... use memory_service ...
await memory_service.aclose()  # only closes clients infolang-adk constructed itself
```

Passing an existing `client=` means `infolang-adk` never closes it — close it
yourself (or use `async with AsyncInfoLang(...) as client:`).

## Namespace scoping

InfoLang namespaces are a single flat string per memory bank. ADK scopes
memory two-dimensionally, by `(app_name, user_id)`. `infolang-adk` collapses
that pair into one namespace per default scheme:

```
{namespace_prefix}-{sanitized app_name}-{sanitized user_id}
```

`namespace_prefix` defaults to `"adk"`. Both segments are sanitized to
`[a-zA-Z0-9_.-]` (anything else, including `/` and `:`, becomes `-`) because
`app_name`/`user_id` end up in InfoLang recall/list-memory query strings.
Two different `(app_name, user_id)` pairs whose *sanitized* forms are
identical share a namespace — e.g. `("team/a", "u1")` and `("team a", "u1")`
both sanitize to `adk-team-a-u1`. If that collision risk matters for your
`app_name`/`user_id` values, supply your own mapping:

```python
memory_service = InfoLangMemoryService(
    api_key="il_live_...",
    namespace_for=lambda app_name, user_id: f"prod-{app_name}-{user_id}",
)
```

Every InfoLang API call `infolang-adk` makes is scoped to the namespace for
the `app_name`/`user_id` of the request that triggered it — one user's agent
session never reads or writes another user's memories through this service
(scoping only fails if InfoLang's own API-key/workspace boundary is
misconfigured upstream; namespace scoping happens after that boundary, not
instead of it).

## What each method actually does

- **`search_memory(app_name, user_id, query)`** — one InfoLang `recall` call
  scoped to the namespace, IL-cosine ranked, top `search_top_k` (default 10)
  chunks. Not keyword/full-text search. A namespace with nothing written to
  it yet returns an empty result rather than raising. `min_score=` filters
  out low-confidence chunks client-side (InfoLang's SDK treats scores below
  `0.85` as a weak match; this class does not enforce that threshold for you
  — set `min_score=0.85` if you want it enforced).
- **`add_session_to_memory(session)`** — ingests every event in the session
  that has non-empty text content, one `remember_batch` call (the SDK sends
  it as a single `execute` round trip with one `remember` sub-op per event).
  Re-ingesting
  the same session (e.g. because it gets called again after more turns)
  stores the events again rather than deduplicating or updating in place —
  InfoLang has no update-in-place primitive for `remember`. If you call this
  repeatedly on a growing session, you'll get duplicate memories for
  already-ingested turns; use `add_events_to_memory` with only the new
  events instead.
- **`add_events_to_memory(app_name, user_id, events, session_id=None,
  custom_metadata=None)`** — ingests an explicit delta of events (does not
  require the full session object). `custom_metadata` scalar values become
  InfoLang tags (`"key:value"`); non-scalar values are dropped, since
  InfoLang's `remember` has no structured-metadata field, only string tags.
- **`add_memory(app_name, user_id, memories, custom_metadata=None)`** —
  ingests explicit `MemoryEntry` items directly (no event/session framing).
  Entries with no text content raise `ValueError`; InfoLang only stores text.

Every InfoLang API error other than "namespace not found" (which
`search_memory` treats as empty results) propagates to the caller as the
`infolang` SDK's typed exceptions (`AuthenticationError`, `ValidationError`,
`RateLimitError`, `ServerError`, ...) — this package does not swallow or
retry them.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy src/infolang_adk
pytest
```

Unit tests mock the HTTP layer (no live InfoLang calls). An optional live
smoke test (`tests/test_live_smoke.py`) runs only when `INFOLANG_API_KEY` is
set, and only ever writes to namespaces prefixed `ittest-adk-`, which it
deletes afterward.

## License

Apache-2.0

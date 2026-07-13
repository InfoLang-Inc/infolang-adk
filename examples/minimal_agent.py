"""Minimal ADK agent wired to InfoLang memory.

Runs two turns in one session, ends the session (which ingests it into
InfoLang via ``add_session_to_memory``), starts a *new* session, and asks the
agent to recall something from the first session via the built-in
``load_memory`` tool (backed by ``InfoLangMemoryService.search_memory``).

Requires:
  GOOGLE_API_KEY   -- for the Gemini model the agent calls.
  INFOLANG_API_KEY -- for InfoLangMemoryService (falls back to this env var
                       automatically; see the infolang SDK's credential
                       resolution).

Run:
  GOOGLE_API_KEY=... INFOLANG_API_KEY=... python examples/minimal_agent.py
"""

from __future__ import annotations

import asyncio
import uuid

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory
from google.genai import types

from infolang_adk import InfoLangMemoryService

APP_NAME = "infolang_adk_example"
USER_ID = "example-user"


async def main() -> None:
    # namespace_prefix isolates example runs from other InfoLangMemoryService
    # usage in the same InfoLang account; drop it to use the library default
    # ("adk").
    memory_service = InfoLangMemoryService(namespace_prefix="example-adk")
    session_service = InMemorySessionService()

    agent = Agent(
        name="assistant",
        model="gemini-2.0-flash",
        instruction=(
            "You are a helpful assistant with long-term memory. Use the "
            "load_memory tool when the user asks about something from a "
            "previous conversation."
        ),
        tools=[load_memory],
    )

    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
    )

    try:
        # -- turn 1: tell the agent something worth remembering -------------
        first_session_id = str(uuid.uuid4())
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=first_session_id
        )
        await _run_turn(
            runner,
            session_id=first_session_id,
            text="My favorite programming language is Rust.",
        )

        # Ending the session ingests it into InfoLang.
        first_session = await session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=first_session_id
        )
        await memory_service.add_session_to_memory(first_session)

        # -- turn 2: new session, ask the agent to recall it -----------------
        second_session_id = str(uuid.uuid4())
        await session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=second_session_id
        )
        await _run_turn(
            runner,
            session_id=second_session_id,
            text="What's my favorite programming language?",
        )
    finally:
        await memory_service.aclose()


async def _run_turn(runner: Runner, *, session_id: str, text: str) -> None:
    print(f"\nuser> {text}")
    message = types.Content(role="user", parts=[types.Part(text=text)])
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"{event.author}> {part.text}")


if __name__ == "__main__":
    asyncio.run(main())

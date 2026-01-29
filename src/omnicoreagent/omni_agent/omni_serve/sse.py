"""
OmniServe SSE (Server-Sent Events) Utilities.

Provides utilities for streaming agent events via SSE.
"""

import asyncio
import json
from typing import TYPE_CHECKING, Union, AsyncGenerator

from omnicoreagent.core.utils import logger

if TYPE_CHECKING:
    from omnicoreagent.omni_agent.agent import OmniCoreAgent
    from omnicoreagent.omni_agent.deep_agent import DeepAgent

AgentType = Union["OmniCoreAgent", "DeepAgent"]


def format_sse_event(event_type: str, data: dict) -> str:
    """
    Format data as an SSE event string.

    Args:
        event_type: The event type (e.g., 'message', 'tool_call', 'complete')
        data: The event data to send

    Returns:
        SSE-formatted string
    """
    json_data = json.dumps(data, default=str)
    return f"event: {event_type}\ndata: {json_data}\n\n"


def format_sse_data(data: dict) -> str:
    """
    Format data as an SSE data-only event.

    Args:
        data: The event data to send

    Returns:
        SSE-formatted string with just data field
    """
    json_data = json.dumps(data, default=str)
    return f"data: {json_data}\n\n"


async def run_agent_stream(
    agent: AgentType,
    query: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Run the agent and stream events via SSE.

    This function:
    1. Starts the agent.run() task
    2. Streams events as they occur
    3. Yields the final result when complete

    Args:
        agent: The agent to run (OmniCoreAgent or DeepAgent)
        query: The user query
        session_id: Session ID for the conversation

    Yields:
        SSE-formatted event strings
    """
    # Track completion
    run_complete = asyncio.Event()
    result = {"data": None, "error": None}

    async def run_agent():
        """Run the agent and capture result."""
        try:
            response = await agent.run(query, session_id=session_id)
            result["data"] = response
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"OmniServe SSE: Agent run error: {e}")
        finally:
            run_complete.set()

    # Start agent run as background task
    agent_task = asyncio.create_task(run_agent())

    # Yield session start event
    yield format_sse_event("session", {"session_id": session_id, "status": "started"})

    # Stream events while agent is running
    try:
        # Create event streaming task
        async def stream_events():
            try:
                async for event in agent.stream_events(session_id):
                    if hasattr(event, "model_dump"):
                        event_data = event.model_dump()
                    elif hasattr(event, "dict"):
                        event_data = event.dict()
                    elif isinstance(event, dict):
                        event_data = event
                    else:
                        event_data = {"data": str(event)}

                    event_type = event_data.get("type", "event")
                    if isinstance(event_type, str):
                        pass
                    elif hasattr(event_type, "value"):
                        event_type = event_type.value
                    else:
                        event_type = str(event_type)

                    yield format_sse_event(event_type, event_data)
            except Exception as e:
                logger.error(f"OmniServe SSE: Event streaming error: {e}")
                yield format_sse_event("error", {"error": str(e)})

        # Stream events with timeout checks
        async for sse_event in stream_events():
            yield sse_event
            # Check if run completed
            if run_complete.is_set():
                break

    except asyncio.CancelledError:
        agent_task.cancel()
        raise

    # Wait for agent to complete if not already
    if not run_complete.is_set():
        await asyncio.wait_for(run_complete.wait(), timeout=300)

    # Yield final result
    if result["error"]:
        yield format_sse_event(
            "error",
            {
                "error": result["error"],
                "session_id": session_id,
            },
        )
    else:
        yield format_sse_event(
            "complete",
            {
                "session_id": session_id,
                "response": result["data"].get("response", ""),
                "agent_name": result["data"].get("agent_name", ""),
                "metric": result["data"].get("metric"),
            },
        )

    yield format_sse_event("session", {"session_id": session_id, "status": "ended"})


async def stream_session_events(
    agent: AgentType,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream existing events for a session via SSE.

    Used for reconnecting to an existing session or
    replaying past events.

    Args:
        agent: The agent
        session_id: Session ID to stream events for

    Yields:
        SSE-formatted event strings
    """
    yield format_sse_event("session", {"session_id": session_id, "status": "streaming"})

    try:
        async for event in agent.stream_events(session_id):
            if hasattr(event, "model_dump"):
                event_data = event.model_dump()
            elif hasattr(event, "dict"):
                event_data = event.dict()
            elif isinstance(event, dict):
                event_data = event
            else:
                event_data = {"data": str(event)}

            event_type = event_data.get("type", "event")
            if hasattr(event_type, "value"):
                event_type = event_type.value

            yield format_sse_event(event_type, event_data)
    except Exception as e:
        logger.error(f"OmniServe SSE: Event replay error: {e}")
        yield format_sse_event("error", {"error": str(e)})

    yield format_sse_event("session", {"session_id": session_id, "status": "ended"})

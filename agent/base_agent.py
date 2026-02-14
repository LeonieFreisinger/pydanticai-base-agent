"""
Base Agent with streaming and Langfuse instrumentation.

Showcases:
- Generic BaseAgent[OutputType, DepsType] pattern
- Streaming with OutputData protocol
- Conditional Langfuse/OpenTelemetry instrumentation
- Provider-agnostic model instantiation
"""

import os
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
)

from .models import OutputCategory, OutputData

# Type variables for generic agent
OutputType = TypeVar("OutputType")
DepsType = TypeVar("DepsType")


def setup_langfuse_instrumentation(service_name: str = "sql-query-agent") -> bool:
    """
    Configure Langfuse instrumentation if enabled.

    Showcases:
    - Conditional observability setup
    - OpenTelemetry integration via logfire
    - Service naming for trace grouping

    Returns:
        True if instrumentation was enabled, False otherwise.
    """
    if not os.getenv("ENABLE_LANGFUSE", "false").lower() == "true":
        return False

    try:
        import logfire

        logfire.configure(
            service_name=service_name,
            send_to_logfire=False,  # We use Langfuse, not Logfire cloud
        )
        logfire.instrument_pydantic_ai()
        return True
    except ImportError:
        print("Warning: logfire not installed. Langfuse instrumentation disabled.")
        return False


class BaseAgent(Generic[OutputType, DepsType]):
    """
    Base agent class providing streaming and instrumentation.

    Showcases:
    - Generic type parameters for output and dependencies
    - Unified streaming interface via OutputData
    - Tool execution tracking
    - Chat history accumulation

    Type Parameters:
        OutputType: The type returned by agent.run() (str or Pydantic model)
        DepsType: The dependencies dataclass injected via RunContext
    """

    def __init__(
        self,
        model: Any,
        *,
        system_prompt: str = "",
        tools: list | None = None,
        instructions: list | None = None,
        output_type: type[OutputType] | None = None,
        retries: int = 3,
        instrument: bool = True,
        service_name: str = "sql-query-agent",
    ):
        """
        Initialize the base agent.

        Args:
            model: pydantic-ai model instance (from get_model())
            system_prompt: Static system prompt string
            tools: List of tool functions to register
            instructions: List of callables taking RunContext, returning str
            output_type: Pydantic model for structured output, or None for str
            retries: Maximum retry attempts for ModelRetry
            instrument: Whether to enable Langfuse instrumentation
            service_name: Service name for trace grouping
        """
        self.model = model
        self.tools = tools or []
        self.retries = retries

        # Setup instrumentation if requested
        if instrument:
            setup_langfuse_instrumentation(service_name)

        # Build the pydantic-ai Agent
        # Showcases: Agent construction with all configuration options
        agent_kwargs: dict[str, Any] = {
            "model": model,
            "retries": retries,
        }

        if system_prompt:
            agent_kwargs["system_prompt"] = system_prompt

        if output_type:
            agent_kwargs["output_type"] = output_type

        self._agent: Agent[DepsType, OutputType] = Agent(**agent_kwargs)

        # Register tools
        for tool in self.tools:
            self._agent.tool(tool)

        # Register instructions (dynamic prompts)
        # Showcases: instructions vs system_prompt - instructions are evaluated at runtime
        if instructions:
            for instruction_fn in instructions:
                self._agent.instructions(instruction_fn)

    async def run(
        self,
        prompt: str,
        deps: DepsType,
        message_history: list[ModelMessage] | None = None,
    ) -> OutputType:
        """
        Run the agent without streaming.

        Showcases:
        - Simple non-streaming execution
        - Message history for multi-turn conversations

        Args:
            prompt: User message
            deps: Dependencies to inject
            message_history: Previous conversation messages

        Returns:
            Agent output (str or structured type)
        """
        result = await self._agent.run(
            prompt,
            deps=deps,
            message_history=message_history,
        )
        return result.output

    async def stream(
        self,
        prompt: str,
        deps: DepsType,
        message_history: list[ModelMessage] | None = None,
    ) -> AsyncIterator[OutputData]:
        """
        Stream agent response with tool execution events.

        Showcases:
        - Streaming protocol with OutputData events
        - Text streaming with stream_text()
        - Chat history on final_result for persistence

        Yields:
            OutputData events as the agent processes the request
        """
        async with self._agent.run_stream(
            prompt,
            deps=deps,
            message_history=message_history,
        ) as result:
            # Track accumulated text
            accumulated_text = ""

            # Stream text deltas
            async for text_delta in result.stream_text(delta=True):
                accumulated_text += text_delta
                yield OutputData(
                    output_message=text_delta,
                    category=OutputCategory.MODEL_REQUEST,
                )

            # Final result with chat history
            # Showcases: Chat history for multi-turn conversation persistence
            chat_history = self._messages_to_dict(result.all_messages())

            yield OutputData(
                output_message=accumulated_text,
                category=OutputCategory.FINAL_RESULT,
                chat_history=chat_history,
            )

    async def reply_deterministic(
        self,
        content: str,
        chunk_size: int = 20,
    ) -> AsyncIterator[OutputData]:
        """
        Emit pre-computed content as if streaming from LLM.

        Showcases:
        - Deterministic response injection
        - Simulated streaming for UI consistency
        - Useful for cached/precomputed responses

        Args:
            content: Pre-computed response text
            chunk_size: Characters per chunk (simulates token streaming)

        Yields:
            OutputData events mimicking LLM streaming
        """
        # Chunk the content to simulate streaming
        for i in range(0, len(content), chunk_size):
            chunk = content[i : i + chunk_size]
            yield OutputData(
                output_message=chunk,
                category=OutputCategory.MODEL_REQUEST,
            )

        # Final result
        yield OutputData(
            output_message=content,
            category=OutputCategory.FINAL_RESULT,
            chat_history=[
                {
                    "role": "assistant",
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ],
        )

    def _messages_to_dict(self, messages: list[ModelMessage]) -> list[dict]:
        """Convert pydantic-ai messages to serializable dicts."""
        result = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                # User message
                for part in msg.parts:
                    if hasattr(part, "content"):
                        result.append(
                            {
                                "role": "user",
                                "content": part.content,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
            elif isinstance(msg, ModelResponse):
                # Assistant message
                text_parts = []
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        text_parts.append(part.content)

                if text_parts:
                    result.append(
                        {
                            "role": "assistant",
                            "content": "".join(text_parts),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

        return result

"""
Base Agent with streaming and Langfuse instrumentation.

Showcases:
- Generic BaseAgent[OutputType, DepsType] pattern
- Streaming with OutputData protocol
- Conditional Langfuse/OpenTelemetry instrumentation
- Provider-agnostic model instantiation
"""

import inspect
import json
import os
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic_ai import (
    Agent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    TextPartDelta,
)
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
)

from .models import OutputCategory, OutputData, ToolStatus, ToolStepInfo

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
        self.has_structured_output = output_type is not None

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
        Run the agent WITHOUT STREAMING.

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
        Stream the agent graph (LLM tokens + tool calls) as OutputData events.

        Showcases:
        - `Agent.iter()` node handling for MODEL_REQUEST / CALL_TOOLS / FINAL_RESULT
        - Structured-output shortcut: falls back to run() when `output_type` is set
        - Tool call lifecycle (STARTED, STREAMING, FINISHED, RETRY, ERROR) via ToolStepInfo
        - Chat history persistence on FINAL_RESULT so multi-turn chains stay in sync

        Yields:
            OutputData events as the agent processes the request
        """
        # For structured output (Level 1), use run() instead of streaming
        # since streaming works better with text responses
        if self.has_structured_output:
            result = await self._agent.run(
                prompt,
                deps=deps,
                message_history=message_history,
            )
            # Convert structured output to formatted string for display
            if hasattr(result.output, "model_dump_json"):
                output_text = result.output.model_dump_json(indent=2)
            else:
                output_text = str(result.output)

            # Yield the complete result as a single message
            yield OutputData(
                output_message=output_text,
                category=OutputCategory.MODEL_REQUEST,
            )

            # Final result with chat history
            chat_history = self._messages_to_dict(result.all_messages())
            yield OutputData(
                output_message=output_text,
                category=OutputCategory.FINAL_RESULT,
                chat_history=chat_history,
            )
            return

        accumulated_text = ""
        active_tool_calls: dict[str, ToolStepInfo] = {}

        try:
            async with self._agent.iter(
                prompt,
                deps=deps,
                message_history=message_history,
            ) as agent_run:
                async for node in agent_run:
                    if isinstance(node, ModelRequestNode):
                        async with node.stream(agent_run.ctx) as agent_stream:
                            final_result_started = False

                            async for stream_event in agent_stream:
                                if isinstance(stream_event, PartDeltaEvent) and isinstance(
                                    stream_event.delta, TextPartDelta
                                ):
                                    text_delta = stream_event.delta.content_delta
                                    accumulated_text += text_delta
                                    yield OutputData(
                                        output_message=text_delta,
                                        category=OutputCategory.MODEL_REQUEST,
                                    )
                                elif isinstance(stream_event, FinalResultEvent):
                                    final_result_started = True
                                    break

                            if final_result_started:
                                async for text_delta in agent_stream.stream_text(delta=True):
                                    accumulated_text += text_delta
                                    yield OutputData(
                                        output_message=text_delta,
                                        category=OutputCategory.MODEL_REQUEST,
                                    )
                    elif isinstance(node, CallToolsNode):
                        async with node.stream(agent_run.ctx) as tool_events:
                            async for tool_event in tool_events:
                                if isinstance(tool_event, FunctionToolCallEvent):
                                    tool_name = tool_event.part.tool_name
                                    tool_call_id = tool_event.part.tool_call_id
                                    tool_args = self._normalize_tool_args(tool_event.part.args)

                                    tool_step = ToolStepInfo(
                                        tool_name=tool_name,
                                        tool_call_id=tool_call_id,
                                        args=tool_args,
                                        status=ToolStatus.STARTED,
                                    )
                                    active_tool_calls[tool_call_id] = tool_step

                                    yield OutputData(
                                        output_message="",
                                        category=OutputCategory.CALL_TOOLS,
                                        tool_step_info=tool_step,
                                    )
                                elif isinstance(tool_event, FunctionToolResultEvent):
                                    tool_call_id = tool_event.tool_call_id

                                    if tool_call_id in active_tool_calls:
                                        tool_step = active_tool_calls[tool_call_id]
                                        result_content = self._get_tool_result_content(
                                            tool_event.result
                                        )

                                        if self._is_async_iterable(result_content):
                                            final_content = ""
                                            async for chunk in result_content:
                                                chunk_text = str(chunk)
                                                final_content += chunk_text

                                                yield OutputData(
                                                    output_message="",
                                                    category=OutputCategory.CALL_TOOLS,
                                                    tool_step_info=ToolStepInfo(
                                                        tool_name=tool_step.tool_name,
                                                        tool_call_id=tool_step.tool_call_id,
                                                        args=tool_step.args,
                                                        status=ToolStatus.STREAMING,
                                                        result=chunk_text,
                                                    ),
                                                )

                                            if hasattr(tool_event.result, "content"):
                                                tool_event.result.content = final_content

                                            tool_step.result = final_content
                                        else:
                                            tool_step.result = result_content

                                        tool_step.status = ToolStatus.FINISHED

                                        yield OutputData(
                                            output_message="",
                                            category=OutputCategory.CALL_TOOLS,
                                            tool_step_info=tool_step,
                                        )

                                        active_tool_calls.pop(tool_call_id, None)

                final_result = agent_run.result
                if final_result is None:
                    return

                chat_history = self._messages_to_dict(final_result.all_messages())
                final_output = final_result.output
                if hasattr(final_output, "model_dump_json"):
                    output_text = final_output.model_dump_json(indent=2)
                else:
                    output_text = str(final_output)

                if not accumulated_text:
                    accumulated_text = output_text

                yield OutputData(
                    output_message=accumulated_text,
                    category=OutputCategory.FINAL_RESULT,
                    chat_history=chat_history,
                )
        except GeneratorExit:
            # Client disconnected, stop iteration gracefully
            return

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

    def _normalize_tool_args(self, args: Any) -> dict[str, Any]:
        """Ensure tool arguments are represented as a dictionary."""
        if args is None:
            return {}

        if isinstance(args, dict):
            return args

        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except json.JSONDecodeError:
                return {"value": args}

            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}

        if hasattr(args, "model_dump"):
            return args.model_dump()

        if hasattr(args, "__dict__"):
            return {key: value for key, value in vars(args).items() if not key.startswith("_")}

        return {"value": args}

    def _get_tool_result_content(self, result: Any) -> Any:
        """Extract the tool result payload, preferring `content` if present."""
        if hasattr(result, "content"):
            return result.content
        return result

    def _is_async_iterable(self, value: Any) -> bool:
        """Return True if value can be iterated asynchronously."""
        return inspect.isasyncgen(value) or isinstance(value, AsyncIterator)

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

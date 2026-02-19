"""
Chainlit chat interface for SQL Query Agent.

Showcases:
- Real-time streaming with Chainlit's native support
- Tool execution visualization with cl.Step
- Multi-turn conversation persistence
- ModelRetry visualization (nested steps for retries)

Run with: chainlit run app.py
"""

from pathlib import Path
from uuid import uuid4

import chainlit as cl
from chainlit import ChatSettings
from chainlit.input_widget import Switch
from dotenv import load_dotenv

from agent.agent import SQLQueryAgent
from agent.deps import SQLAgentDeps, create_deps
from agent.models import OutputCategory, ToolStatus

# Load environment variables
load_dotenv()

# Database path
DB_PATH = Path(__file__).parent / "showcase.db"


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize agent and deps when chat starts.

    Showcases:
    - Session state management with cl.user_session
    - Deps initialization with conversation ID
    - Welcome message showing configuration
    """
    # Create agent
    agent = SQLQueryAgent()

    # Create deps with unique conversation ID
    conversation_id = uuid4()
    deps = await create_deps(DB_PATH, conversation_id=conversation_id)

    # Store in session for multi-turn persistence
    cl.user_session.set("agent", agent)
    cl.user_session.set("deps", deps)
    cl.user_session.set("message_history", [])
    settings = await ChatSettings(
        inputs=[
            Switch(
                id="stream_tool_output",
                label="Embed tool output in assistant stream",
                initial=False,
                description="When enabled, tool responses are merged into the assistant's streamed reply.",
            )
        ]
    ).send()
    cl.user_session.set("chat_settings", settings)
    cl.user_session.set(
        "stream_tool_output",
        settings.get("stream_tool_output", False),
    )

    # Send welcome message
    welcome = agent.get_welcome_message()
    await cl.Message(content=welcome).send()


@cl.on_chat_end
async def on_chat_end():
    """Clean up database connection when chat ends."""
    deps: SQLAgentDeps | None = cl.user_session.get("deps")
    if deps:
        await deps.close()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle user messages with streaming response.

    Showcases:
    - OutputData event handling
    - msg.stream_token() for text streaming
    - cl.Step for tool execution visualization
    - Nested steps for ModelRetry attempts
    - Chat history persistence for multi-turn
    """
    agent: SQLQueryAgent = cl.user_session.get("agent")
    deps: SQLAgentDeps = cl.user_session.get("deps")
    message_history = cl.user_session.get("message_history", [])
    stream_tool_output = cl.user_session.get("stream_tool_output", False)
    structured_output = bool(getattr(agent, "config", {}).get("structured_output"))

    # Create response message for streaming
    response_msg = cl.Message(content="")
    await response_msg.send()

    # Track active tool steps for proper nesting
    active_steps: dict[str, cl.Step] = {}
    inline_tool_meta: dict[str, dict[str, bool]] = {}

    structured_rendered = False

    try:
        # Stream agent response
        async for event in agent.stream(
            message.content,
            deps,
            message_history=message_history if message_history else None,
        ):
            if event.category == OutputCategory.MODEL_REQUEST:
                if structured_output and not structured_rendered:
                    response_msg.content = f"```json\n{event.output_message}\n```"
                    await response_msg.update()
                    structured_rendered = True
                else:
                    # Stream text tokens
                    await response_msg.stream_token(event.output_message)

            elif event.category == OutputCategory.CALL_TOOLS:
                tool_info = event.tool_step_info
                if not tool_info:
                    continue

                tool_name = tool_info.tool_name

                if tool_info.status == ToolStatus.STARTED:
                    # Create new step for tool execution
                    step = cl.Step(
                        name=tool_name,
                        type="tool",
                    )
                    step.input = format_tool_input(tool_info.args)
                    await step.send()
                    active_steps[tool_name] = step

                    # Stream tool execution start
                    await response_msg.stream_token(f"\n\n🔧 **Calling {tool_name}**")

                    # Show what the tool is doing
                    if tool_info.tool_representation:
                        step.output = f"⏳ {tool_info.tool_representation}"
                        await step.update()
                        await response_msg.stream_token(f"\n⏳ {tool_info.tool_representation}")

                elif tool_info.status == ToolStatus.STREAMING:
                    # Incrementally append tool output while it runs
                    chunk = str(tool_info.result or "")
                    step = active_steps.get(tool_name)
                    if step:
                        existing_output = step.output or ""
                        step.output = f"{existing_output}{chunk}"
                        await step.update()

                elif tool_info.status == ToolStatus.FINISHED:
                    # Update step with result
                    step = active_steps.get(tool_name)
                    if step:
                        step.output = format_tool_output(tool_info.result)
                        await step.update()
                    active_steps.pop(tool_name, None)

                    # Stream completion
                    await response_msg.stream_token(f"\n✅ {tool_name} completed")

                elif tool_info.status == ToolStatus.RETRY:
                    # Showcases: ModelRetry visualization
                    step = active_steps.get(tool_name)
                    if step:
                        # Show retry message
                        retry_msg = f"🔄 **Retry requested:**\n\n{tool_info.error or 'Retrying...'}"
                        step.output = retry_msg
                        await step.update()

                    # Stream retry message
                    await response_msg.stream_token(
                        f"\n🔄 Retry requested: {tool_info.error or 'Retrying...'}"
                    )

                elif tool_info.status == ToolStatus.ERROR:
                    step = active_steps.get(tool_name)
                    if step:
                        step.output = f"❌ Error: {tool_info.error}"
                        await step.update()

                    # Stream error
                    await response_msg.stream_token(f"\n❌ {tool_name} error: {tool_info.error}")

            elif event.category == OutputCategory.FINAL_RESULT:
                # Update message history for multi-turn
                # Showcases: Chat history persistence
                if event.chat_history:
                    # Add user message
                    message_history.append(
                        {
                            "role": "user",
                            "content": message.content,
                        }
                    )
                    # Add assistant messages from this turn
                    for hist_msg in event.chat_history:
                        if hist_msg.get("role") == "assistant":
                            message_history.append(hist_msg)

                    cl.user_session.set("message_history", message_history)

                    # Update deps chat_history for context injection
                    deps.chat_history = message_history

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        await response_msg.stream_token(f"\n\n{error_msg}")

    # Finalize the response message
    await response_msg.update()


def format_tool_input(args: dict) -> str:
    """Format tool arguments for display."""
    if not args:
        return "(no arguments)"

    lines = []
    for key, value in args.items():
        if key == "sql":
            # Format SQL nicely
            lines.append(f"**{key}:**\n```sql\n{value}\n```")
        elif isinstance(value, str) and len(value) > 100:
            lines.append(f"**{key}:** {value[:100]}...")
        else:
            lines.append(f"**{key}:** {value}")

    return "\n".join(lines)


def format_tool_output(result) -> str:
    """Format tool result for display."""
    if result is None:
        return "✅ Completed"

    # Handle Pydantic models
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif isinstance(result, dict):
        data = result
    elif isinstance(result, list):
        # List of models (e.g., TableInfo, ColumnInfo)
        if result and hasattr(result[0], "model_dump"):
            data = [item.model_dump() for item in result]
        else:
            data = result
    elif isinstance(result, str):
        # String result (templates, dates)
        if len(result) > 500:
            return f"```\n{result[:500]}...\n```"
        return f"```\n{result}\n```"
    else:
        return f"✅ {result}"

    # Format based on type
    if isinstance(data, list):
        if not data:
            return "✅ Empty result"

        # Check if it's table/column info
        if isinstance(data[0], dict) and "name" in data[0]:
            lines = ["| Name | Details |", "|------|---------|"]
            for item in data[:20]:  # Limit display
                name = item.get("name", "?")
                if "row_count" in item:
                    # TableInfo
                    lines.append(f"| {name} | {item.get('row_count', 0)} rows |")
                elif "data_type" in item:
                    # ColumnInfo
                    dtype = item.get("data_type", "?")
                    nullable = "NULL" if item.get("nullable") else "NOT NULL"
                    lines.append(f"| {name} | {dtype} {nullable} |")
                else:
                    lines.append(f"| {name} | - |")

            if len(data) > 20:
                lines.append(f"| ... | ({len(data) - 20} more) |")

            return "\n".join(lines)

    elif isinstance(data, dict):
        # QueryResult or similar
        if "sql" in data and "rows" in data:
            # QueryResult
            rows = data.get("rows", [])
            columns = data.get("columns", [])
            row_count = data.get("row_count", len(rows))
            exec_time = data.get("execution_time_ms", 0)

            output_lines = [
                f"**{row_count} rows** returned in {exec_time:.2f}ms",
                "",
            ]

            if rows and columns:
                # Build markdown table
                output_lines.append("| " + " | ".join(str(c) for c in columns) + " |")
                output_lines.append("|" + "|".join(["---"] * len(columns)) + "|")

                for row in rows[:15]:  # Limit to 15 rows
                    formatted_row = [str(v)[:30] if v is not None else "NULL" for v in row]
                    output_lines.append("| " + " | ".join(formatted_row) + " |")

                if len(rows) > 15:
                    output_lines.append(f"\n*... and {len(rows) - 15} more rows*")

            if data.get("truncated"):
                output_lines.append("\n⚠️ Results truncated to 100 rows")

            return "\n".join(output_lines)

    # Fallback: JSON-ish display
    import json

    try:
        return f"```json\n{json.dumps(data, indent=2, default=str)[:1000]}\n```"
    except Exception:
        return f"✅ {str(data)[:500]}"


# Custom CSS for better tool step display
@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates if needed."""
    pass


if __name__ == "__main__":
    # This allows running with: python app.py
    # But prefer: chainlit run app.py
    print("Run with: chainlit run app.py")

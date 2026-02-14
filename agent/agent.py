"""
SQL Query Agent with DEMO_LEVEL feature gating.

Showcases:
- Progressive feature exposure via DEMO_LEVEL
- Tool registration based on capability level
- Structured vs string output modes
- Full agent configuration with all pydantic-ai features
"""

import os
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage

from .base_agent import BaseAgent
from .config import get_model, get_model_settings
from .deps import SQLAgentDeps
from .models import StructuredQueryPlan
from .prompts import INSTRUCTIONS, get_demo_level_instructions
from .tools import get_tools_for_level


def get_demo_level() -> int:
    """
    Get current demo level from environment.

    Levels:
        1: Structured output only (no tools, no execution)
        2: Schema exploration (list_tables, describe_table)
        3: Full query execution (all tools)
        4: Full + ModelRetry + streaming + Langfuse
    """
    return int(os.getenv("DEMO_LEVEL", "4"))


class SQLQueryAgent:
    """
    SQL Query Agent with configurable capability levels.

    Showcases:
    - DEMO_LEVEL gating for progressive teaching
    - Structured output (Level 1) vs string output (Levels 2-4)
    - Dynamic tool registration
    - Integration of all pydantic-ai features

    Usage:
        agent = SQLQueryAgent()
        deps = await create_deps(db_path)

        # Non-streaming
        result = await agent.run("Show me all tables", deps)

        # Streaming
        async for event in agent.stream("Show me low stock products", deps):
            print(event)
    """

    def __init__(
        self,
        demo_level: int | None = None,
        provider: str | None = None,
        model_name: str | None = None,
    ):
        """
        Initialize the SQL Query Agent.

        Args:
            demo_level: Override DEMO_LEVEL env var (1-4)
            provider: LLM provider override ('openai', 'anthropic', 'google')
            model_name: Model name override
        """
        self.demo_level = demo_level or get_demo_level()
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4.1")

        # Get model instance (provider-agnostic)
        # Showcases: Provider abstraction via factory
        model = get_model(provider=provider, model_name=model_name)
        _model_settings = get_model_settings(self.model_name)  # noqa: F841

        # Get tools for this level
        tools = get_tools_for_level(self.demo_level)

        # Build system prompt with level-specific instructions
        system_prompt = self._build_system_prompt()

        # Determine output type: structured for Level 1, string for others
        # Showcases: Structured output vs free-form text
        output_type: type | None = None
        if self.demo_level == 1:
            output_type = StructuredQueryPlan

        # Create the base agent
        # Showcases: Full agent configuration
        self._agent: BaseAgent[Any, SQLAgentDeps] = BaseAgent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            instructions=INSTRUCTIONS if self.demo_level >= 2 else None,
            output_type=output_type,
            retries=3 if self.demo_level >= 4 else 1,
            instrument=self.demo_level >= 4,  # Langfuse only at Level 4
            service_name="sql-query-agent",
        )

        # Store config for display
        self.config = {
            "demo_level": self.demo_level,
            "model": self.model_name,
            "tools": [t.__name__ for t in tools],
            "structured_output": self.demo_level == 1,
            "langfuse_enabled": self.demo_level >= 4,
        }

    def _build_system_prompt(self) -> str:
        """Build system prompt with level-specific instructions."""
        base_prompt = """You are a SQL Query Agent - an expert database analyst that helps users
explore and query an e-commerce SQLite database.

The database contains:
- suppliers: Vendor information and reliability metrics
- products: Product catalog with inventory levels and pricing
- customers: Customer accounts and purchase history
- orders: Order records with status and totals
- order_items: Line items linking orders to products

Your goal is to help users get insights from this data through SQL queries.
"""
        # Add level-specific instructions
        level_instructions = get_demo_level_instructions(self.demo_level)

        return base_prompt + "\n" + level_instructions

    async def run(
        self,
        prompt: str,
        deps: SQLAgentDeps,
        message_history: list[ModelMessage] | None = None,
    ) -> Any:
        """
        Run the agent and return the result.

        For Level 1: Returns StructuredQueryPlan
        For Levels 2-4: Returns str

        Args:
            prompt: User question or request
            deps: Agent dependencies with database connection
            message_history: Previous conversation for multi-turn

        Returns:
            Agent output (structured or string based on level)
        """
        return await self._agent.run(prompt, deps, message_history)

    async def stream(
        self,
        prompt: str,
        deps: SQLAgentDeps,
        message_history: list[ModelMessage] | None = None,
    ):
        """
        Stream agent response with tool execution events.

        Yields OutputData events:
        - model_request: Text tokens from LLM
        - call_tools: Tool execution start/finish
        - final_result: Complete response with chat_history

        Args:
            prompt: User question or request
            deps: Agent dependencies
            message_history: Previous conversation

        Yields:
            OutputData events for real-time UI updates
        """
        async for event in self._agent.stream(prompt, deps, message_history):
            yield event

    def get_welcome_message(self) -> str:
        """Generate welcome message showing current configuration."""
        level_descriptions = {
            1: "Structured Output Only - generates query plans without execution",
            2: "Schema Exploration - can list tables and describe schemas",
            3: "Full Query Execution - can run SELECT queries",
            4: "Full Agent Mode - with error recovery, streaming, and Langfuse tracing",
        }

        tools_str = ", ".join(self.config["tools"]) if self.config["tools"] else "None"

        return f"""# SQL Query Agent Showcase

**Demo Level:** {self.demo_level} - {level_descriptions[self.demo_level]}
**Model:** {self.config["model"]}
**Tools:** {tools_str}
**Structured Output:** {self.config["structured_output"]}
**Langfuse Tracing:** {self.config["langfuse_enabled"]}

---

Try asking:
- "What tables are available?" (Level 2+)
- "Describe the products table" (Level 2+)
- "Show me products below reorder point" (Level 3+)
- "What were the top 5 customers last month?" (Level 3+)
- "Show me sales for table xyz" (Level 4 - triggers ModelRetry)
"""


# Convenience function for quick testing
async def create_agent_and_deps(
    db_path: Path | str | None = None,
) -> tuple[SQLQueryAgent, SQLAgentDeps]:
    """
    Create agent and deps for testing.

    Args:
        db_path: Path to database, defaults to showcase.db

    Returns:
        Tuple of (agent, deps)
    """
    from .deps import create_deps

    if db_path is None:
        db_path = Path(__file__).parent.parent / "showcase.db"

    agent = SQLQueryAgent()
    deps = await create_deps(db_path)

    return agent, deps

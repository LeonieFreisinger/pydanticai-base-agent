"""
Agent dependencies (Deps) for SQL Query Agent.

Showcases:
- Dependency injection pattern with pydantic-ai
- Mutable state that accumulates across tool calls
- Conversation persistence for multi-turn interactions
"""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import aiosqlite


@dataclass
class SQLAgentDeps:
    """
    Dependencies injected into every tool via RunContext.

    This is the agent's "working memory" - it starts minimal and gets
    hydrated by tools as the conversation progresses.

    Showcases:
    - RunContext[SQLAgentDeps] injection pattern
    - State that persists across tool calls within a conversation
    - Caching (schema_cache) to avoid redundant queries
    - History tracking for observability

    Attributes:
        db_path: Path to SQLite database file
        connection: Active database connection (created on first use)
        schema_cache: Cached table schemas to avoid re-fetching
        query_history: Log of all executed queries in this session
        conversation_id: Unique ID for this conversation (for persistence)
        chat_history: Accumulated message history for multi-turn context
    """

    db_path: Path
    connection: aiosqlite.Connection | None = None
    schema_cache: dict[str, list[dict]] = field(default_factory=dict)
    query_history: list[dict] = field(default_factory=list)
    conversation_id: UUID = field(default_factory=uuid4)
    chat_history: list[dict] = field(default_factory=list)

    async def get_connection(self) -> aiosqlite.Connection:
        """
        Get or create database connection.

        Showcases: Lazy initialization pattern - connection created on first use.
        """
        if self.connection is None:
            self.connection = await aiosqlite.connect(self.db_path)
            # Enable row factory for dict-like access
            self.connection.row_factory = aiosqlite.Row
        return self.connection

    async def close(self) -> None:
        """Close database connection if open."""
        if self.connection:
            await self.connection.close()
            self.connection = None

    def add_to_history(
        self,
        query: str,
        success: bool,
        row_count: int | None = None,
        error: str | None = None,
        execution_time_ms: float | None = None,
    ) -> None:
        """
        Record a query execution in history.

        Showcases: Deps mutation for state accumulation across tool calls.
        """
        self.query_history.append(
            {
                "query": query,
                "success": success,
                "row_count": row_count,
                "error": error,
                "execution_time_ms": execution_time_ms,
            }
        )


async def create_deps(db_path: Path | str, conversation_id: UUID | None = None) -> SQLAgentDeps:
    """
    Factory function to create agent dependencies.

    Showcases:
    - Deps factory pattern (matches production pattern with optional caching)
    - Async initialization when needed

    Args:
        db_path: Path to SQLite database
        conversation_id: Optional ID for conversation continuity

    Returns:
        Initialized SQLAgentDeps ready for agent use
    """
    return SQLAgentDeps(
        db_path=Path(db_path),
        conversation_id=conversation_id or uuid4(),
    )

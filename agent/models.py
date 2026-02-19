"""
Pydantic models for SQL Agent Showcase.

Showcases:
- Field(description=...) for self-documenting models that LLMs interpret
- Structured output types for validated extraction
- Streaming protocol models (OutputData, ToolStepInfo)
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Schema Information Models
# =============================================================================


class ColumnInfo(BaseModel):
    """
    Information about a database column.

    Showcases: Field descriptions that help the LLM understand data semantics.
    """

    name: str = Field(description="Column name as it appears in the database")
    data_type: str = Field(description="SQL data type (e.g., INTEGER, TEXT, REAL)")
    nullable: bool = Field(description="Whether the column allows NULL values")
    is_primary_key: bool = Field(default=False, description="Whether this is a primary key column")
    default_value: str | None = Field(default=None, description="Default value if specified")
    description: str | None = Field(
        default=None, description="Human-readable description of what this column represents"
    )


class TableInfo(BaseModel):
    """
    Information about a database table.

    Showcases: Nested Pydantic models with rich field descriptions.
    """

    name: str = Field(description="Table name in the database")
    row_count: int = Field(description="Approximate number of rows in the table")
    columns: list[ColumnInfo] = Field(
        default_factory=list, description="List of columns in this table with their metadata"
    )
    description: str | None = Field(
        default=None, description="Human-readable description of what this table stores"
    )


# =============================================================================
# Query Result Models
# =============================================================================


class QueryResult(BaseModel):
    """
    Result of executing a SQL query.

    Showcases:
    - Rich field descriptions for LLM interpretation
    - Structured data that's easy to display and reason about
    """

    sql: str = Field(description="The SQL query that was executed")
    columns: list[str] = Field(description="Column names in the result set")
    rows: list[list[Any]] = Field(
        description="Result rows as lists of values, matching column order"
    )
    row_count: int = Field(description="Number of rows returned")
    execution_time_ms: float = Field(description="Query execution time in milliseconds")
    truncated: bool = Field(
        default=False, description="Whether results were truncated due to size limits"
    )


# =============================================================================
# Structured Output for Level 1 (No Execution)
# =============================================================================


class QueryComplexity(str, Enum):
    """Estimated complexity of a SQL query."""

    SIMPLE = "simple"  # Single table, basic WHERE
    MODERATE = "moderate"  # JOINs or subqueries
    COMPLEX = "complex"  # Multiple JOINs, aggregations, window functions


class StructuredQueryPlan(BaseModel):
    """
    Structured output for Level 1: Agent generates query plan without execution.

    Showcases:
    - Pydantic output_type for structured extraction
    - Agent generates validated data instead of free-form text
    - Useful for query review/approval workflows
    """

    natural_language_request: str = Field(
        description="The user's original question in natural language"
    )
    sql_query: str = Field(description="The generated SQL query to answer the question")
    tables_used: list[str] = Field(description="List of table names referenced in the query")
    estimated_complexity: QueryComplexity = Field(
        description="Estimated complexity level of the query"
    )
    explanation: str = Field(description="Brief explanation of how the query answers the question")
    potential_issues: list[str] = Field(
        default_factory=list, description="Any potential issues or edge cases to be aware of"
    )


# =============================================================================
# Streaming Protocol Models
# =============================================================================


class ToolStatus(str, Enum):
    """Status of a tool execution."""

    STARTED = "started"
    STREAMING = "streaming"
    FINISHED = "finished"
    RETRY = "retry"
    ERROR = "error"


class ToolStepInfo(BaseModel):
    """
    Information about a tool execution step.

    Showcases: Structured tool execution tracking for UI visualization.
    """

    tool_name: str = Field(description="Name of the tool being executed")
    tool_call_id: str | None = Field(
        default=None, description="Unique identifier for the tool call"
    )
    args: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool")
    status: ToolStatus = Field(description="Current status of the tool execution")
    result: Any = Field(default=None, description="Tool result (when finished)")
    error: str | None = Field(default=None, description="Error message if failed")
    tool_representation: str | None = Field(
        default=None, description="Human-readable description of what the tool is doing"
    )
    duration_ms: float | None = Field(
        default=None, description="Execution duration in milliseconds"
    )


class OutputCategory(str, Enum):
    """Category of streaming output event."""

    MODEL_REQUEST = "model_request"  # Text tokens from LLM
    CALL_TOOLS = "call_tools"  # Tool execution events
    FINAL_RESULT = "final_result"  # Complete response


class OutputData(BaseModel):
    """
    Unified streaming output model.

    Showcases:
    - Single protocol for all streaming events
    - Categories distinguish text vs tool vs final events
    - Matches production BaseAgent.stream() output pattern
    """

    output_message: str = Field(
        default="", description="Text content (for model_request and final_result)"
    )
    category: OutputCategory = Field(description="Type of output event")
    tool_step_info: ToolStepInfo | None = Field(
        default=None, description="Tool execution details (for call_tools category)"
    )
    chat_history: list[dict] | None = Field(
        default=None, description="Full conversation history (only in final_result)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When this event was generated"
    )

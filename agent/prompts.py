"""
Dynamic prompt engineering for SQL Query Agent.

Showcases:
- `instructions` callables vs static `system_prompt`
- Runtime context injection via RunContext
- Keeping data and instructions co-located for better LLM comprehension
"""

import os

from pydantic_ai import RunContext

from .deps import SQLAgentDeps


def get_persona_instructions(ctx: RunContext[SQLAgentDeps]) -> str:
    """
    Static persona and behavioral rules.

    Showcases:
    - Basic instructions callable that doesn't need dynamic data
    - Still receives RunContext for consistency (could access deps if needed)
    """
    return """
<persona>
You are an expert SQL analyst and database assistant. Your role is to help users
query and understand data in a SQLite e-commerce database.

You are:
- Precise: Generate syntactically correct, efficient SQL
- Action-oriented: Try queries directly - if they fail, you'll get error feedback to fix them
- Educational: Explain your queries and results clearly
- Safe: Only execute read-only queries (SELECT statements)
</persona>

<behavioral_rules>
1. When a user asks you to run a specific query, execute it EXACTLY as given
2. If a query fails with an error, you will receive the error message - use it to fix and retry
3. Don't pre-validate or refuse to run queries - let the system catch errors
4. Use list_tables if you need to discover the schema
5. Use describe_table to understand column types when needed
6. Format SQL queries with proper indentation for readability
7. Limit results to 100 rows unless the user asks for more
</behavioral_rules>

<error_recovery>
When you receive an error from execute_query:
1. Read the error message carefully
2. Identify the issue (syntax error, wrong table/column name, etc.)
3. Fix the query and try again immediately
4. If you're unsure, use describe_table to check column names
</error_recovery>

<output_format>
- Present SQL queries in markdown code blocks with `sql` syntax highlighting
- Format tabular results as markdown tables when under 20 rows
- For larger results, summarize key findings
- Always include the row count in your response
</output_format>
"""


def get_schema_context(ctx: RunContext[SQLAgentDeps]) -> str:
    """
    Dynamically inject cached schema information into the prompt.

    Showcases:
    - Dynamic prompt engineering: context injected at runtime
    - Deps access via ctx.deps
    - Schema info loaded close to where it's used in instructions
    """
    deps = ctx.deps


    if not deps.schema_cache:
        return """
<schema_context>
No schema information loaded yet. Use list_tables and describe_table to explore.
</schema_context>
"""
    if os.getenv("DEMO_LEVEL") == "2":
        cached_tables = ", ".join(sorted(deps.schema_cache.keys())) or "none"
        print(f"🧭 [level2] get_schema_context() called; cached tables: {cached_tables}")


    # Build schema summary from cache
    schema_lines = ["<schema_context>", "Tables you have explored:"]

    for table_name, columns in deps.schema_cache.items():
        schema_lines.append(f"\n  {table_name}:")
        for col in columns:
            nullable = "NULL" if col.get("nullable") else "NOT NULL"
            pk = " (PK)" if col.get("is_primary_key") else ""
            schema_lines.append(f"    - {col['name']}: {col['data_type']} {nullable}{pk}")

    schema_lines.append("</schema_context>")
    return "\n".join(schema_lines)


def get_conversation_context(ctx: RunContext[SQLAgentDeps]) -> str:
    """
    Inject recent conversation history for multi-turn awareness.

    Showcases:
    - Multi-turn context via deps.chat_history
    - Summary generation to avoid prompt bloat
    - Dynamic instructions based on conversation state
    """
    deps = ctx.deps

    if not deps.chat_history:
        return ""

    # Summarize recent queries for context
    recent_queries = [
        entry
        for entry in deps.query_history[-5:]  # Last 5 queries
        if entry.get("success")
    ]

    if not recent_queries:
        return ""

    context_lines = [
        "<conversation_context>",
        "Recent successful queries in this session:",
    ]

    for i, query_entry in enumerate(recent_queries, 1):
        sql = query_entry.get("query", "")
        # Truncate long queries
        if len(sql) > 100:
            sql = sql[:100] + "..."
        context_lines.append(f"  {i}. {sql}")

    context_lines.append("\nUse this context to understand follow-up questions.")
    context_lines.append("</conversation_context>")

    return "\n".join(context_lines)


def get_demo_level_instructions(demo_level: int) -> str:
    """
    Return level-specific instructions based on DEMO_LEVEL.

    Showcases:
    - Conditional prompt content for progressive feature exposure
    - Used by the Agent class, not as an instructions callable
    """
    if demo_level == 1:
        return """
<level_1_mode>
You are in STRUCTURED OUTPUT mode. You do NOT have database access.
Instead of executing queries, return a structured query plan with:
- The SQL query you would execute
- Tables and columns involved
- Complexity assessment
- Potential issues

This demonstrates pydantic-ai's structured output validation.
</level_1_mode>
"""
    elif demo_level == 2:
        return """
<level_2_mode>
You are in SCHEMA EXPLORATION mode. You can:
- List tables in the database
- Describe table schemas

You cannot execute queries yet. Focus on understanding the data model.
</level_2_mode>
"""
    elif demo_level == 3:
        return """
<level_3_mode>
You are in FULL QUERY mode. You can:
- Explore the schema (list_tables, describe_table)
- Execute SELECT queries
- Use query templates for common patterns
- Get current date for time-based queries
</level_3_mode>
"""
    else:  # Level 4
        return """
<level_4_mode>
You are in FULL AGENT mode with all capabilities:
- Schema exploration
- Query execution with automatic error recovery
- Streaming responses with tool visualization
- Langfuse observability tracing

IMPORTANT: When executing queries:
- Try the query as given by the user first
- If you receive an error, you MUST fix it and retry immediately
- The system will send you error details - use them to correct your query
- Don't give up on the first error - iterate until the query works
</level_4_mode>
"""


# List of instructions callables to register with the agent
# Each receives RunContext and returns a string
INSTRUCTIONS = [
    get_persona_instructions,
    get_schema_context,
    get_conversation_context,
]

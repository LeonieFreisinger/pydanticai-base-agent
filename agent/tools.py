"""
SQL Agent tools with ModelRetry error recovery.

Showcases:
- RunContext dependency injection
- ModelRetry for LLM self-correction
- Pydantic models as return types with Field(description=...)
- Deps mutation for state accumulation
- Tool docstrings as LLM schema

IMPORTANT: Tool docstrings and parameter annotations become the schema
that the LLM sees. Write them clearly and include all relevant context.
"""

import logging
import os
import time
from datetime import datetime

from pydantic_ai import ModelRetry, RunContext

from .deps import SQLAgentDeps
from .models import ColumnInfo, QueryResult, TableInfo

# Set up logging for retry visibility
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

# Tables the agent is allowed to access (for demo: access control via ModelRetry)
ALLOWED_TABLES = {
    "suppliers",
    "products",
    "customers",
    "orders",
    "order_items",
}

# Maximum rows to return (safety limit)
MAX_RESULT_ROWS = 100


# =============================================================================
# Tool: list_tables
# =============================================================================


async def list_tables(ctx: RunContext[SQLAgentDeps]) -> list[TableInfo]:
    """
    List all accessible tables in the database with row counts.

    Returns a list of TableInfo objects containing table names and row counts.
    Use this tool first to understand what data is available before querying.

    Showcases:
    - RunContext[SQLAgentDeps] injection (ctx is auto-provided by pydantic-ai)
    - Returning Pydantic models with rich descriptions
    - Async database operations
    """
    logger.info("🛠️  Tool called: list_tables()")
    conn = await ctx.deps.get_connection()

    tables = []
    for table_name in sorted(ALLOWED_TABLES):
        # Get row count
        cursor = await conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        row = await cursor.fetchone()
        count = row[0] if row else 0

        tables.append(
            TableInfo(
                name=table_name,
                row_count=count,
                description=_get_table_description(table_name),
            )
        )

    return tables


def _get_table_description(table_name: str) -> str:
    """Return human-readable description for known tables."""
    descriptions = {
        "suppliers": "Vendors who supply products, with contact info and reliability metrics",
        "products": "Product catalog with pricing, inventory levels, and reorder points",
        "customers": "Customer accounts with type (regular/premium/enterprise) and spend history",
        "orders": "Customer orders with status, dates, and totals",
        "order_items": "Line items for each order linking products to orders",
    }
    return descriptions.get(table_name, "")


# =============================================================================
# Tool: describe_table
# =============================================================================


async def describe_table(
    ctx: RunContext[SQLAgentDeps],
    table_name: str,
) -> list[ColumnInfo]:
    """
    Get detailed schema information for a specific table.

    Args:
        table_name: Name of the table to describe (e.g., 'products', 'orders')

    Returns:
        List of ColumnInfo objects with column names, types, and constraints.

    Use this tool to understand column names and types before writing queries.
    Results are cached for efficiency in subsequent queries.

    Showcases:
    - ModelRetry for invalid input recovery
    - Deps mutation (caching schema in deps.schema_cache)
    - Parameter validation with helpful error messages
    """
    logger.info(f"🛠️  Tool called: describe_table(table_name='{table_name}')")
    # Validate table access
    if table_name not in ALLOWED_TABLES:
        # Showcases: ModelRetry sends error back to LLM for self-correction
        logger.warning(f"🔄 ModelRetry: Invalid table '{table_name}' - not in allowed list")
        raise ModelRetry(
            f"Table '{table_name}' is not accessible. "
            f"Allowed tables: {', '.join(sorted(ALLOWED_TABLES))}. "
            f"Please choose from these tables."
        )

    # Check cache first
    if table_name in ctx.deps.schema_cache:
        cached = ctx.deps.schema_cache[table_name]
        return [ColumnInfo(**col) for col in cached]

    conn = await ctx.deps.get_connection()

    # Get column info using PRAGMA
    cursor = await conn.execute(f"PRAGMA table_info({table_name})")
    rows = await cursor.fetchall()

    columns = []
    for row in rows:
        col_info = ColumnInfo(
            name=row[1],  # column name
            data_type=row[2],  # type
            nullable=not row[3],  # notnull flag (inverted)
            is_primary_key=bool(row[5]),  # pk flag
            default_value=row[4],  # default value
            description=_get_column_description(table_name, row[1]),
        )
        columns.append(col_info)

    # Cache in deps for future use
    # Showcases: Deps mutation - state accumulates across tool calls
    ctx.deps.schema_cache[table_name] = [col.model_dump() for col in columns]

    if os.getenv("DEMO_LEVEL") == "2":
        logger.info("[level2] schema context hydrated for table '%s'", table_name)

    return columns


def _get_column_description(table_name: str, column_name: str) -> str:
    """Return description for known columns (for teaching purposes)."""
    descriptions = {
        ("products", "stock_level"): "Current inventory quantity on hand",
        ("products", "reorder_point"): "Stock level that triggers reorder alert",
        ("products", "safety_stock"): "Minimum stock to maintain as buffer",
        ("products", "unit_price"): "Selling price per unit",
        ("products", "cost_price"): "Purchase cost per unit from supplier",
        ("customers", "customer_type"): "Customer tier: regular, premium, or enterprise",
        ("customers", "total_spent"): "Cumulative order total for this customer",
        ("orders", "status"): "Order status: pending, confirmed, shipped, delivered, cancelled",
        ("suppliers", "reliability_score"): "Historical on-time delivery rate (0.0 to 1.0)",
        ("suppliers", "lead_time_days"): "Typical days from order to delivery",
    }
    return descriptions.get((table_name, column_name), "")


# =============================================================================
# Tool: execute_query
# =============================================================================


async def execute_query(
    ctx: RunContext[SQLAgentDeps],
    sql: str,
) -> QueryResult:
    """
    Execute a read-only SQL query and return results.

    Args:
        sql: A SELECT statement to execute. Must be read-only (no INSERT/UPDATE/DELETE).
             Include appropriate LIMIT clause for large tables.

    Returns:
        QueryResult with columns, rows (up to 100), execution time, and truncation flag.

    If the query fails, you'll receive the error message and can try again with corrections.
    If results are empty, consider broadening your WHERE conditions.

    Showcases:
    - Multiple ModelRetry scenarios (syntax error, empty results)
    - Query history tracking via deps mutation
    - Execution timing for observability
    """
    logger.info("🛠️  Tool called: execute_query()")
    logger.debug(f"SQL: {sql[:100]}...")
    # Validate read-only
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        logger.warning(f"🔄 ModelRetry: Non-SELECT query attempted: {sql_upper[:50]}...")
        raise ModelRetry(
            "Only SELECT queries are allowed. "
            f"Your query starts with: {sql_upper.split()[0] if sql_upper else 'empty'}. "
            "Please rewrite as a SELECT statement."
        )

    # Check for dangerous keywords
    dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
    for keyword in dangerous:
        if keyword in sql_upper:
            logger.warning(f"🔄 ModelRetry: Dangerous keyword '{keyword}' in query")
            raise ModelRetry(
                f"Query contains forbidden keyword: {keyword}. "
                "Only read-only SELECT queries are allowed."
            )

    conn = await ctx.deps.get_connection()
    start_time = time.perf_counter()

    try:
        cursor = await conn.execute(sql)
        rows_raw = await cursor.fetchall()

        execution_time = (time.perf_counter() - start_time) * 1000

        # Get column names
        columns = [desc[0] for desc in cursor.description] if cursor.description else []

        # Convert rows to lists
        rows = [list(row) for row in rows_raw]
        total_rows = len(rows)

        # Check for empty results
        if total_rows == 0:
            # Log the attempt
            ctx.deps.add_to_history(
                query=sql,
                success=True,
                row_count=0,
                execution_time_ms=execution_time,
            )
            # Showcases: ModelRetry for empty results with helpful suggestion
            logger.warning(f"🔄 ModelRetry: Query returned 0 rows - SQL: {sql[:80]}...")
            raise ModelRetry(
                "Query returned no results. This could mean:\n"
                "1. The WHERE conditions are too restrictive\n"
                "2. The data doesn't exist\n"
                "3. There's a typo in column values\n\n"
                "Try:\n"
                "- Removing some WHERE conditions\n"
                "- Using LIKE with % wildcards for text matching\n"
                "- Checking the actual values with a simpler query first"
            )

        # Truncate if needed
        truncated = total_rows > MAX_RESULT_ROWS
        if truncated:
            rows = rows[:MAX_RESULT_ROWS]

        # Record success in history
        ctx.deps.add_to_history(
            query=sql,
            success=True,
            row_count=total_rows,
            execution_time_ms=execution_time,
        )

        logger.info(f"✅ Query succeeded: {total_rows} rows in {execution_time:.2f}ms")

        return QueryResult(
            sql=sql,
            columns=columns,
            rows=rows,
            row_count=total_rows,
            execution_time_ms=round(execution_time, 2),
            truncated=truncated,
        )

    except ModelRetry:
        # Re-raise ModelRetry as-is
        raise
    except Exception as e:
        execution_time = (time.perf_counter() - start_time) * 1000

        # Log the failure
        ctx.deps.add_to_history(
            query=sql,
            success=False,
            error=str(e),
            execution_time_ms=execution_time,
        )

        # Showcases: ModelRetry with error details for LLM self-correction
        logger.warning(f"🔄 ModelRetry: SQL error - {e}")
        raise ModelRetry(
            f"SQL execution failed with error:\n\n"
            f"```\n{e}\n```\n\n"
            f"Your query was:\n"
            f"```sql\n{sql}\n```\n\n"
            f"Please fix the syntax error and try again."
        )


# =============================================================================
# Tool: get_current_date
# =============================================================================


async def get_current_date(ctx: RunContext[SQLAgentDeps]) -> str:
    """
    Get the current date and time for time-based queries.

    Returns:
        Current datetime in ISO format (YYYY-MM-DDTHH:MM:SS)

    Use this when writing queries that filter by date, such as:
    - Orders from the last 30 days
    - Year-over-year comparisons
    - Finding recent activity

    Showcases:
    - Simple utility tool
    - Tool docstring becomes LLM's understanding of when to use it
    """
    logger.info("🛠️  Tool called: get_current_date()")
    return datetime.now().isoformat()


# =============================================================================
# Tool: get_stock_query_template
# =============================================================================


async def get_stock_query_template(ctx: RunContext[SQLAgentDeps]) -> str:
    """
    Get a pre-built SQL template for common stock analysis queries.

    Returns:
        A SQL query template with placeholders for stock level analysis.

    This template helps analyze inventory status including:
    - Products below reorder point
    - Stock value calculations
    - Supplier lead time considerations

    Replace {{condition}} with your specific WHERE clause.

    Showcases:
    - Retriever pattern: pre-built templates for common tasks
    - Injection of domain knowledge via tools
    """
    logger.info("🛠️  Tool called: get_stock_query_template()")
    template = """
-- Stock Analysis Template
-- Replace {{condition}} with specific filters

SELECT 
    p.sku,
    p.name,
    p.category,
    p.stock_level,
    p.reorder_point,
    p.safety_stock,
    p.stock_level - p.reorder_point AS stock_vs_reorder,
    CASE 
        WHEN p.stock_level <= p.safety_stock THEN 'CRITICAL'
        WHEN p.stock_level <= p.reorder_point THEN 'LOW'
        WHEN p.stock_level > p.reorder_point * 3 THEN 'OVERSTOCK'
        ELSE 'OK'
    END AS stock_status,
    p.unit_price * p.stock_level AS stock_value,
    s.name AS supplier_name,
    s.lead_time_days,
    s.reliability_score
FROM products p
JOIN suppliers s ON p.supplier_id = s.id
WHERE p.is_active = 1
  AND {{condition}}
ORDER BY stock_vs_reorder ASC;

-- Example conditions:
-- p.stock_level < p.reorder_point  -- Products needing reorder
-- p.category = 'Electronics'        -- Filter by category  
-- s.lead_time_days > 10            -- Long lead time items
"""
    return template


# =============================================================================
# Tool Registration Helpers
# =============================================================================


def get_tools_for_level(level: int) -> list:
    """
    Get the list of tools to register based on DEMO_LEVEL.

    Showcases:
    - Progressive tool exposure for teaching
    - Feature gating via environment variable

    Levels:
        1: No tools (structured output only)
        2: Schema exploration (list_tables, describe_table)
        3: Full query (+ execute_query, get_current_date, get_stock_query_template)
        4: Full + ModelRetry demonstrations (same tools, errors enabled)
    """
    if level == 1:
        return []
    elif level == 2:
        return [list_tables, describe_table]
    else:  # Level 3 and 4
        return [
            list_tables,
            describe_table,
            execute_query,
            get_current_date,
            get_stock_query_template,
        ]


# Human-readable tool representations for UI display
TOOL_REPRESENTATIONS = {
    "list_tables": lambda args: "Listing database tables...",
    "describe_table": lambda args: f"Describing table: {args.get('table_name', 'unknown')}",
    "execute_query": lambda args: "Executing SQL query...",
    "get_current_date": lambda args: "Getting current date/time",
    "get_stock_query_template": lambda args: "Loading stock analysis template",
}


def get_tool_representation(tool_name: str, args: dict) -> str:
    """Get human-readable description of a tool call for UI display."""
    formatter = TOOL_REPRESENTATIONS.get(tool_name)
    if formatter:
        return formatter(args)
    return f"Running {tool_name}..."

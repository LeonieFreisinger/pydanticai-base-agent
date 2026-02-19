# SQL Query Agent - Building Agents with pydantic-ai

This document provides a comprehensive walkthrough of the SQL Query Agent showcase, designed for showcasing how to build agents with pydantic-ai concepts.

---

## Table of Contents

1. [Core Components](#1-core-components)
2. [Progressive Feature Walkthrough](#2-progressive-feature-walkthrough)
3. [Pydantic-AI Features Map](#3-pydantic-ai-features-map)
4. [Production Comparison](#4-production-comparison)
5. [Key Takeaways](#5-key-takeaways)

---

<details>
<summary>1. Core Components</summary>

## 1. Core Components

### 2.1 Base Agent (`base_agent.py`)

**What to point out:**
- Generic type parameters `BaseAgent[OutputType, DepsType]`
- Conditional Langfuse setup via `instrument` flag
- `stream()` yields `OutputData` events with three categories
- Message history conversion for persistence

```python
class BaseAgent(Generic[OutputType, DepsType]):
    """Base agent providing streaming and instrumentation."""

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
        self.model = model
        self.tools = tools or []
        self.retries = retries
        self.has_structured_output = output_type is not None

        # Setup instrumentation if requested
        if instrument:
            setup_langfuse_instrumentation(service_name)

        # Build the pydantic-ai Agent
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
        if instructions:
            for instruction_fn in instructions:
                self._agent.instructions(instruction_fn)
```

### 2.2 Dependencies (`deps.py`)

**What to point out:**
- Dataclass with both immutable config (`db_path`) and mutable state (`schema_cache`, `query_history`)
- `async def get_connection()` shows lazy initialization and row factory setup
- `add_to_history()` method demonstrates deps mutation for state accumulation

```python
@dataclass
class SQLAgentDeps:
    """
    Dependencies injected into every tool via RunContext.
    This is the agent's "working memory" across tool calls.
    """
    db_path: Path
    connection: aiosqlite.Connection | None = None
    schema_cache: dict[str, list[dict]] = field(default_factory=dict)
    query_history: list[dict] = field(default_factory=list)
    conversation_id: UUID = field(default_factory=uuid4)
    chat_history: list[dict] = field(default_factory=list)

    async def get_connection(self) -> aiosqlite.Connection:
        """Lazy initialization - connection created on first use."""
        if self.connection is None:
            self.connection = await aiosqlite.connect(self.db_path)
            self.connection.row_factory = aiosqlite.Row
        return self.connection
```

### 2.3 Pydantic Models (`models.py`)

**What to point out:**
- `Field(description=...)` makes models self-documenting for LLMs
- The LLM reads these descriptions and understands data semantics without extra prompting
- Enums provide constrained choices

```python
class ColumnInfo(BaseModel):
    """Information about a database column."""
    
    name: str = Field(description="Column name as it appears in the database")
    data_type: str = Field(description="SQL data type (e.g., INTEGER, TEXT, REAL)")
    nullable: bool = Field(description="Whether the column allows NULL values")
    is_primary_key: bool = Field(default=False, description="Whether this is a primary key")
    default_value: str | None = Field(default=None, description="Default value if specified")
    description: str | None = Field(
        default=None,
        description="Human-readable description of what this column represents"
    )
```

### 2.4 Tools (`tools.py`)

**What to point out:**
- First parameter is always `RunContext[DepsType]` - auto-injected by pydantic-ai
- Docstrings become the schema the LLM sees - write them clearly!
- `ModelRetry` sends errors back to the LLM for self-correction
- Tools return Pydantic models, not dicts or strings

```python
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
    """
    # Validate table access
    if table_name not in ALLOWED_TABLES:
        # ModelRetry: Error sent back to LLM for correction
        raise ModelRetry(
            f"Table '{table_name}' is not accessible. "
            f"Allowed tables: {', '.join(sorted(ALLOWED_TABLES))}."
        )
    
    # ... execute query ...
    
    # Cache in deps for future use (deps mutation)
    ctx.deps.schema_cache[table_name] = [col.model_dump() for col in columns]
    
    return columns
```

### 2.5 Dynamic Prompts (`prompts.py`)

**What to point out:**
- `instructions` callables vs static `system_prompt`
- Runtime access to deps via `ctx.deps`
- Dynamic context injection keeps data close to instructions

```python
def get_schema_context(ctx: RunContext[SQLAgentDeps]) -> str:
    """
    Dynamically inject cached schema information into the prompt.
    
    Showcases: Runtime context injection via deps.
    """
    deps = ctx.deps
    
    if not deps.schema_cache:
        return "<schema_context>No schema information loaded yet. Use list_tables and describe_table to explore.</schema_context>"
    
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
```

### 2.6 Provider Configuration (`config.py`)

**What to point out:**
- Single factory function abstracts all providers
- `get_model_settings()` maps settings per model (reasoning models use `reasoning_effort`)
- Easy to add new providers

```python
def get_model(
    provider: str | None = None,
    model_name: str | None = None,
) -> Model:
    """Factory function for provider-agnostic model creation."""
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    default_models: dict[str, str] = {
        "openai": "gpt-4.1",
        "anthropic": "claude-sonnet-4-20250514",
        "google": "gemini-2.0-flash",
    }
    model_name = model_name or os.getenv("MODEL_NAME", default_models.get(provider, "gpt-4.1"))
    
    if provider == "openai":
        return OpenAIModel(model_name)
    elif provider == "anthropic":
        return AnthropicModel(model_name)
    elif provider == "google":
        return GeminiModel(model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use: openai, anthropic, google")
```

---

</details>

<details>
<summary>2. Progressive Feature Walkthrough</summary>

## 2. Progressive Feature Walkthrough

<details>
<summary>Level 1: Structured Output Only</summary>

### Level 1: Structured Output Only

**Configuration:**
```bash
DEMO_LEVEL=1
```

**What's Active:**
- No tools
- `output_type=StructuredQueryPlan`
- Agent generates SQL plans without execution

**Try This:**
```
"Generate a query to find the top 5 customers by total spending"
```
Expected: Returns `StructuredQueryPlan` with `natural_language_request`, `sql_query`, `tables_used`, `estimated_complexity`, `explanation`, `potential_issues`

**Learning Focus:**
- Pydantic structured output validation
- How LLM generates typed, validated data
- `Field(description=...)` enriches schema for the LLM, clarifying intent/semantics of each field

See the output_type assignment in [agent/agent.py](agent/agent.py#L88-L92) and the StructuredQueryPlan model in [agent/models.py](agent/models.py#L94-L115).



---

</details>

<details>
<summary>Level 2: Dependency Injection and Dynamic Prompting</summary>

### Level 2: Dependency Injection and Dynamic Prompting

**Configuration:**
```bash
DEMO_LEVEL=2
```

**What's Active:**
- `list_tables` tool
- `describe_table` tool
- Dynamic instructions that inject cached schema into the prompt


**Try This:**
```
1. "What tables are in the database?"
2. "Describe the products table"
3. "What columns does the orders table have?"
```
Expected: Agent explores schema progressively. After step 2, the cached products schema appears in the LLM prompt for step 3, enabling smarter follow-up questions.

**Learning Focus:**
- **RunContext Dependency Injection:** Tools receive `ctx: RunContext[SQLAgentDeps]` as their first parameter, auto-injected by pydantic-ai. This gives them access to the agent's working memory.
- **State Mutation Across Tool Calls:** `ctx.deps.schema_cache` is mutable. When `describe_table` is called, it populates the cache. Future tool calls see this accumulated data.
- **Dynamic Prompt Engineering:** The `get_schema_context()` instruction callable is re-evaluated before each LLM turn, injecting the cached schema into the prompt so the LLM sees what's already been explored.



**The Available Tools:**

1. **`list_tables(ctx)`** ([tools.py](agent/tools.py#L47-55))
    - Queries the database for all accessible table names and row counts
    - Returns `list[TableInfo]` with metadata
    - Does NOT cache (it's a lightweight read-only query)
    - Called first to discover what data exists

2. **`describe_table(ctx, table_name)`** ([tools.py](agent/tools.py#L98-140))
    - Gets detailed column information (names, types, nullability, primary keys, defaults)
    - Uses SQLite's `PRAGMA table_info()` for schema
    - **Crucially:** Mutates `ctx.deps.schema_cache[table_name]` to cache results
    - Validates input with `ModelRetry`—invalid table names are rejected with suggestions, letting the LLM self-correct


#### How It Works Under the Hood

**1. Dependency Injection via RunContext**

Pydantic-ai automatically inspects tool function signatures. When it sees a parameter `ctx: RunContext[SQLAgentDeps]`, it:
1. Creates a `RunContext` object at request time
2. Attaches the current `SQLAgentDeps` instance to it
3. Passes it to the tool as the first argument

Code links: [agent/tools.py](agent/tools.py#L99) for `describe_table()`

```python
# In tools.py
async def describe_table(
    ctx: RunContext[SQLAgentDeps],  # ← pydantic-ai injects this automatically
    table_name: str,
) -> list[ColumnInfo]:
    deps = ctx.deps  # Access the injected SQLAgentDeps instance
    # deps contains: connection, schema_cache, query_history, etc.
```

This means all tools in the same conversation share the same `deps` object. When one tool mutates it, the changes are visible to the next tool.

**2. State Mutation Across Tool Calls**

Schema caching is the accumulation of data across multiple tool calls:

**Call 1:** User asks "Describe the products table"
- `describe_table(ctx, 'products')` executes
- Queries `PRAGMA table_info('products')`  
- Gets columns: `[{'name': 'id', 'data_type': 'INTEGER', ...}, ...]`
- **Mutates:** `ctx.deps.schema_cache['products'] = [...]`  ← State saved in deps

**Call 2:** User asks "Describe the orders table"
- `describe_table(ctx, 'orders')` executes  
- Queries `PRAGMA table_info('orders')`
- Gets columns: `[{'name': 'order_id', 'data_type': 'INTEGER', ...}, ...]`
- **Mutates:** `ctx.deps.schema_cache['orders'] = [...]`  ← New data added

**Now `ctx.deps.schema_cache` contains:**
```python
{
    'products': [{'name': 'id', 'data_type': 'INTEGER', ...}, ...],
    'orders': [{'name': 'order_id', 'data_type': 'INTEGER', ...}, ...],
}
```

**Why mutations matter:** Without mutation, each tool would have a fresh, empty cache. With mutation, subsequent tools and prompts see the accumulated knowledge. This is how the agent builds up understanding of the database over multiple turns.

**3. Dynamic Prompt Engineering**

The magic happens with `instructions` callables. These are functions that receive `RunContext` and are re-evaluated **before each LLM turn**:

Code links: [agent/prompts.py](agent/prompts.py#L62) for `get_schema_context()`.

```python
# In prompts.py
def get_schema_context(ctx: RunContext[SQLAgentDeps]) -> str:
    """Dynamically build schema info from cached data."""
    deps = ctx.deps
    
    if not deps.schema_cache:
        return "<schema_context>No schema information loaded yet. Use list_tables and describe_table to explore.</schema_context>"
    
    # Build a formatted string from the cache
    schema_lines = ["<schema_context>", "Tables you explored:"]
    for table_name, columns in deps.schema_cache.items():
        schema_lines.append(f"\n  {table_name}:")
        for col in columns:
            nullable = "NULL" if col.get("nullable") else "NOT NULL"
            pk = " (PK)" if col.get("is_primary_key") else ""
            schema_lines.append(f"    - {col['name']}: {col['data_type']} {nullable}{pk}")
    
    schema_lines.append("</schema_context>")
    return "\n".join(schema_lines)
```

**In agent/agent.py:**
```python
# Register instructions (dynamic prompts) with the agent
instructions=INSTRUCTIONS if self.demo_level >= 2 else None,
```

Where `INSTRUCTIONS` is:
```python
INSTRUCTIONS = [
    get_persona_instructions,      # Static rules
    get_schema_context,            # Dynamic! Re-evaluated each turn
    get_conversation_context,      # Dynamic! Uses deps.query_history
]
```

**Timeline of what happens:**

```
User: "Describe products"
  ↓
LLM sees prompt with:
  - system_prompt
  - get_persona_instructions() → static text
  - get_schema_context() → evaluates NOW
    → deps.schema_cache is EMPTY
    → returns "No schema information loaded yet. Use list_tables and describe_table to explore."
  ↓
Tool executes: describe_table(ctx, 'products')
  ↓
ctx.deps.schema_cache['products'] = [...]  ← DATA NOW IN CACHE
  ↓

User: "What about orders table?"
  ↓
LLM sees prompt with:
  - system_prompt
  - get_persona_instructions() → static text
  - get_schema_context() → evaluates AGAIN (re-evaluated!)
    → deps.schema_cache CONTAINS products data
    → returns:
        <schema_context>
        Tables you explored:
          products:
            - id: INTEGER
            - name: TEXT
            ...
        </schema_context>
  ↓
LLM now has context about what was already explored!
```

**Key insight:** The prompt is NOT static. `get_schema_context()` is called fresh before each LLM turn, so it sees the current state of the cache. This creates **adaptive prompts** that evolve as the conversation progresses.

**Final note:**  `deps.schema_cache` is the structured state used by tools/code, while `get_schema_context()` renders a human-readable summary for the LLM prompt so it can reuse known schema without extra tool calls.


---

</details>

<details>
<summary>Level 3: Tool Orchestration</summary>

### Level 3: Tool Orchestration

**Configuration:**
```bash
DEMO_LEVEL=3
```

**What's Active:**
- All Level 2 tools (`list_tables`, `describe_table`)
- `execute_query` tool (new)
- `get_current_date` tool (new)
- `get_stock_query_template` tool (new)
- Query history tracking in deps
- All schema caching from Level 2

**Try This:**
```
1. "Show me products that are below their reorder point"
2. "What were the top selling products last month?"
3. "Use the stock template to find critical inventory"
```
Expected: Agent executes real SQL queries, results displayed in markdown tables


**Learning Focus:**
- **Tool Orchestration:** Agent reasons about what it needs and calls tools in the right sequence
- **Query History Accumulation:** Each executed query is tracked in `ctx.deps.query_history` for context

**Code Location:** [agent/tools.py](agent/tools.py#L95-L300)

#### Tool Docstrings as LLM Schema

Before diving into tool orchestration, it's important to understand how tools describe themselves. Tool docstrings and Pydantic model field descriptions are **not just documentation**—they are the schema the LLM reads to understand what tools do and what data means.

When you register a tool with the agent, pydantic-ai extracts the docstring and includes it in the LLM's understanding:

```python
# In tools.py
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
    """
```

The LLM sees this docstring and understands: this tool executes SELECT queries, has safety constraints, returns structured data with metadata. This is critical because the LLM needs to know **what it can and cannot do** before attempting a query.

Level 3 demonstrates **self-documenting tools**—clear docstrings enable the LLM to be smart about tool usage without exhaustive prompting.

**The Three New Tools:**

1. **`execute_query(ctx, sql)`** ([tools.py](agent/tools.py#L195-250))
    - Executes a read-only SQL SELECT statement against the database
    - Returns `QueryResult` with columns, rows (max 100), execution time, and truncation flag
    - **Validates** input: rejects anything not `SELECT`, blocks dangerous keywords (INSERT, UPDATE, DELETE, DROP)
    - **On error:** Raises `ModelRetry` with SQL errors or empty-result guidance so the LLM can self-correct
    - **Tracks history:** Calls `ctx.deps.add_to_history()` to log the query for context on next turn
    - This is where the actual database work happens

2. **`get_current_date(ctx)`** ([tools.py](agent/tools.py#L260-275))
    - Returns current date/time in ISO format
    - Used by the LLM when constructing time-based queries ("What happened last month?")
    - Ensures queries are grounded in actual time, not stale assumptions
    - Simple but essential for temporal queries

3. **`get_stock_query_template(ctx)`** ([tools.py](agent/tools.py#L280-300))
   Returns a **parameterized SQL template** for common inventory analysis.

   Example (uses a `{{condition}}` placeholder and joins suppliers):
   ```sql
   SELECT
       p.sku,
       p.name,
       p.stock_level,
       p.reorder_point,
       s.name AS supplier_name
   FROM products p
   JOIN suppliers s ON p.supplier_id = s.id
   WHERE p.is_active = 1
     AND {{condition}}
   ORDER BY p.stock_level - p.reorder_point ASC
   ```

   LLM can use this as a reference or plug in a specific condition.
   Demonstrates the **template/retriever pattern**: pre-built queries that agents can reference, understand, and customize.


#### How It Works Under the Hood

**1. Tool Orchestration: The Agent Chooses What to Do**

Unlike Level 2 where the user explicitly asks "describe products", at Level 3 the agent must decide which tools to call.

When you ask: **"Show me products that are below their reorder point"**

The agent thinks:
```
Do I know the products table schema? 
  → Check schema_cache... NO (not yet cached)
  → Call: describe_table("products")

Now I have the schema. Can I write a query?
  → YES, I know the columns (stock_level, reorder_point)
  → Call: execute_query("SELECT ... WHERE stock_level < reorder_point")

Display the results to the user
```

This is **autonomous tool selection**: the agent reasons about its information gaps and fills them with appropriate tools. It doesn't call every tool every time—it's smart about what's needed.

**The orchestration happens in the LLM's reasoning**, not in code. The LLM sees:
- Available tools: `list_tables`, `describe_table`, `execute_query`, `get_current_date`, `get_stock_query_template`
- Cached schema in the prompt (from `get_schema_context()`)
- Your question
- → It decides the optimal sequence

**2. Query History Accumulation**

Every time `execute_query` runs, it logs the query:

```python
# In tools.py execute_query()
ctx.deps.add_to_history(
    query=sql,
    success=True,
    row_count=total_rows,
    execution_time_ms=execution_time,
)
```

This populates `ctx.deps.query_history` with entries like:
```python
[
    {
        "query": "SELECT * FROM products WHERE stock_level < 10",
        "success": True,
        "row_count": 5,
        "execution_time_ms": 12.3,
    },
    {
        "query": "SELECT * FROM orders WHERE customer_id = 42",
        "success": True,
        "row_count": 8,
        "execution_time_ms": 8.5,
    },
]
```

**Why this matters:** The `get_conversation_context()` instruction callable reads from `query_history`:

```python
# In prompts.py
def get_conversation_context(ctx: RunContext[SQLAgentDeps]) -> str:
    """Inject recent query history for context."""
    if not ctx.deps.chat_history:
        return ""
    recent_queries = [
        entry
        for entry in ctx.deps.query_history[-5:]  # Last 5 queries
        if entry.get("success")
    ]
    
    if not recent_queries:
        return ""
    
    context_lines = ["<conversation_context>", "Recent successful queries:"]
    for i, query_entry in enumerate(recent_queries, 1):
        sql = query_entry.get("query", "")[:100]  # Truncate
        context_lines.append(f"  {i}. {sql}")
    context_lines.append("\nUse this context to understand follow-up questions.")
    context_lines.append("</conversation_context>")
    return "\n".join(context_lines)
```

So on the **next user turn**, the LLM sees a prompt like:
```
<conversation_context>
Recent successful queries in this session:
  1. SELECT * FROM products WHERE stock_level < 10
  2. SELECT * FROM orders WHERE customer_id = 42
</conversation_context>
```

This helps the LLM understand the ongoing investigation and make better follow-up decisions.


Notice how each turn:
- Has access to schema_cache (doesn't re-fetch product columns)
- Has access to query_history (understands prior investigation)
- Chooses the right tool (doesn't call every tool every time)

---

</details>

<details>
<summary>Level 4: Full Agent Mode with Streaming</summary>

### Level 4: Full Agent Mode with Streaming

**Configuration:**
```bash
DEMO_LEVEL=4
```

**What's Active:**
- All Level 3 tools
- ModelRetry error recovery (3 retries)
- Streaming with real-time response delivery
- Tool visualization in Chainlit UI

**Try This:**
```
1. "Show me data from the secret_table"
   → Watch: Invalid table error, agent self-corrects
   
2. "SELECT * FROM products WEHRE stock < 10"
   → Watch: SQL syntax error, agent fixes typo
   
3. "SELECT * FROM products LIMIT 100000"
   → Watch: Large result set truncated, agent handles gracefully
   
4. "SELECT secret_column FROM products"
   → Watch: Column doesn't exist error, agent asks to describe table first
```

Expected: Agent receives error, logs it with 🔄 emoji, corrects itself, and retries automatically. Check console output for logs.


**Learning Focus:**
- Error recovery patterns with ModelRetry
- Streaming protocol architecture and benefits
- Real-time response handling

**Code Location:** [agent/tools.py](agent/tools.py#L124-L126), [agent/base_agent.py](agent/base_agent.py#L80-L320)

#### ModelRetry: The Error Recovery Pattern

ModelRetry is a pydantic-ai feature that enables **LLM self-correction**. When a tool raises a `ModelRetry` error, the LLM receives the error message and gets another attempt to fix the problem. This is fundamentally different from a regular exception—it's a dialogue between the tool and the LLM.

**How it works:**

1. **Tool invokes ModelRetry**
   ```python
   # In tools.py describe_table()
   if table_name not in ALLOWED_TABLES:
       logger.warning(f"🔄 ModelRetry: Invalid table '{table_name}' - not in allowed list")
       raise ModelRetry(
           f"Table '{table_name}' is not accessible. "
           f"Allowed tables: {', '.join(sorted(ALLOWED_TABLES))}. "
           f"Please choose from these tables."
       )
   ```

2. **Pydantic-ai catches it** and sends the error message back to the LLM

3. **LLM receives the error** in its reasoning context and decides to retry

4. **LLM calls the tool again** with corrected input

5. **Up to 3 retries** are allowed (configured via `retries=3` in `agent.py`)


**Example Conversation with ModelRetry:**

```
User: "Show me data from the secret_table"
↓
LLM calls → describe_table("secret_table")
↓
Tool logs: 🔄 ModelRetry: Invalid table 'secret_table' - not in allowed list
Tool raises: ModelRetry("Table 'secret_table' is not accessible. Allowed tables: ..., please choose from these tables")
↓
Pydantic-ai intercepts error, sends message back to LLM
↓
LLM reasoning: "Oh, secret_table doesn't exist. Let me call list_tables() first or pick from the allowed list"
LLM calls → list_tables()
↓
Tool returns: [suppliers, products, customers, orders, order_items]
↓
LLM displays result: "I found these tables available: suppliers, products, ..."
(Retry was transparent to the user)
```


#### Streaming: Real-Time Response Delivery

**Core Concept:**

- LLM output and tool activity stream as typed events in real time.
- Text and tool deltas emit immediately when Agent.iter() yields — no UI blocking.
- Chainlit shows assistant messages alongside tool activity and can insert tool output directly into the reply.

**Why Streaming Matters:**

1. **Faster perceived latency** – time-to-first-token consistently under a few hundred milliseconds.
2. **Explainability** – every tool call is visible as it happens, not retroactively dumped.
3. **Non-blocking** messages don't block.


**Try This:**
```
With streaming enabled (Chainlit UI):
1. "Show me products below reorder point"
   → Watch text appear token-by-token
   → See tool steps appear as Steps
   → Watch tool outputs update in real-time
   
2. "JOIN products with orders and show top 10 by sales"
   → Multiple tool calls visible in sequence
   → Table results streamed as they're generated
   → Entire flow visible in nested Steps
```



**Streaming pipeline (agent/base_agent.py)**
   - Entry point: [`BaseAgent.stream()`](agent/base_agent.py#L160-L252) opens `Agent.iter()` and routes each node type.
    - Model text streaming: inside the `ModelRequestNode` branch we call `node.stream_text(delta=True)` and immediately yield `OutputData(category=MODEL_REQUEST, output_message=token)` so GPT tokens hit the UI as soon as they exist.
    - Tool streaming: inside the `CallToolsNode` branch we handle `FunctionToolCallEvent` (STARTED) and `FunctionToolResultEvent`. When the result content is an async iterable we stream each chunk with `ToolStatus.STREAMING`, accumulate the final string, then emit a `FINISHED` update with the completed payload.
    - Tool lifecycle statuses emitted by `BaseAgent.stream()` are `STARTED`, `STREAMING`, and `FINISHED`.
   - Final result: once the iterator completes we read `run.result`, serialize structured outputs/chat history, and emit one `OutputData(category=FINAL_RESULT, chat_history=...)` so multi-turn state stays in sync.

   ```python
   async def stream(...):
       async with self._agent.iter(...) as agent_run:
           async for node in agent_run:
               if isinstance(node, ModelRequestNode):
                   async with node.stream(agent_run.ctx) as agent_stream:
                       async for stream_event in agent_stream:
                           if isinstance(stream_event, PartDeltaEvent):
                               text_delta = stream_event.delta.content_delta
                               yield OutputData(
                                   output_message=text_delta,
                                   category=OutputCategory.MODEL_REQUEST,
                               )
               elif isinstance(node, CallToolsNode):
                   async with node.stream(agent_run.ctx) as tool_events:
                       async for tool_event in tool_events:
                            if isinstance(tool_event, FunctionToolCallEvent):
                               yield OutputData(... ToolStatus.STARTED ...)

                            elif isinstance(tool_event, FunctionToolResultEvent):
                                tool_step = active_tool_calls[tool_call_id]
                                result_content = self._get_tool_result_content(tool_event.result)

                                if self._is_async_iterable(result_content):
                                    final_content = ""
                                    async for chunk in result_content:
                                        chunk_text = str(chunk)
                                        final_content += chunk_text
                                        yield OutputData(
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
                                    category=OutputCategory.CALL_TOOLS,
                                    tool_step_info=tool_step,
                                )
                                active_tool_calls.pop(tool_call_id, None)
   ```

**Event Flow (token-level example):**

```
User asks: "Show me products below reorder point"
    ↓
MODEL_REQUEST tokens stream
    • "Looking", " at ", "inventory", " tables..."
    • Chainlit renders these in the assistant bubble immediately
    ↓
CALL_TOOLS: describe_table
    • STARTED → cl.Step titled describe_table(products)
    • STREAMING chunks → "Column", " name", " = product_id", ...
    • FINISHED → schema table locked in, optional inline mirror if toggle on
    ↓
CALL_TOOLS: execute_query
    • STARTED → cl.Step: execute_query(low_stock_query)
    • STREAMING chunks → "Product A — qty 3", "\nProduct B — qty 1"
    • FINISHED → final markdown table + duration
    ↓
MODEL_REQUEST (closing thoughts)
    • Tokens: "Found", " 2 products", " below", " reorder", " point."
    • Same assistant bubble continues streaming
    ↓
FINAL_RESULT
    • No new UI element, but spinner stops and chat_history is persisted for the next turn
```

**Further Details on Agent.iter()**

**Graph** – Each agent run compiles into a directed graph of nodes (`ModelRequestNode`, `CallToolsNode`, final result). Nodes know how to execute and how they connect, so the runtime can reason about retries and dependencies.
**`Agent.iter()`** – Calling `async with agent.iter(...) as run:` is how you “open” that graph. It returns a `GraphRun` context manager so you can watch the execution instead of waiting for the final answer.
**`GraphRun`** – The object from `Agent.iter()`. It is an async iterable (`async for node in run:`) that yields every node in order while exposing `run.ctx` (current RunContext) and `run.result` (final structured output after iteration completes).
**Why it matters** – Iterating gives you hooks for streaming: you can emit MODEL_REQUEST tokens as soon as the LLM sends `PartDeltaEvent`s, mirror tool progress by reacting to `FunctionToolCallEvent` / `FunctionToolResultEvent`, inject observability, or stop cleanly if the client disconnects. That’s exactly what `BaseAgent.stream()` does when it turns graph nodes into the uniform `OutputData` protocol consumed by Chainlit.

Think of `run()` as “just give me the final message” and `iter()` as “let me sit inside the execution loop.” If you need interleaved tool+LLM streaming or fine-grained logging, `iter()` is the API to reach for.

---

</details>

</details>

<details>
<summary>3. Pydantic-AI Features Map</summary>

## 3. Pydantic-AI Features Map

| Feature | File | Lines | What to Point Out |
|---------|------|-------|-------------------|
| **RunContext Injection** | [agent/tools.py](agent/tools.py) | [L49-L50](agent/tools.py#L49-L50) | First param `ctx: RunContext[SQLAgentDeps]` auto-injected |
| **ModelRetry** | [agent/tools.py](agent/tools.py) | [L125-L126](agent/tools.py#L125-L126) | `raise ModelRetry(message)` sends error to LLM |
| **Field(description=...)** | [agent/models.py](agent/models.py) | [L28-L30](agent/models.py#L28-L30) | Self-documenting models for LLM interpretation |
| **Structured Output** | [agent/agent.py](agent/agent.py) | [L88-L92](agent/agent.py#L88-L92) | `output_type=StructuredQueryPlan` |
| **instructions Callables** | [agent/prompts.py](agent/prompts.py) | [L62-L80](agent/prompts.py#L62-L80) | Runtime prompt generation with deps access |
| **system_prompt vs instructions** | [agent/agent.py](agent/agent.py) | [L86-L100](agent/agent.py#L86-L100) | Static vs dynamic prompt patterns |
| **Tool Registration** | [agent/base_agent.py](agent/base_agent.py) | [L135](agent/base_agent.py#L135) | `self._agent.tool(tool)` loop |
| **Streaming** | [agent/base_agent.py](agent/base_agent.py) | [L171-L320](agent/base_agent.py#L171-L320) | `async with agent.iter()` pattern |
| **OutputData Protocol** | [agent/models.py](agent/models.py) | [L164-L177](agent/models.py#L164-L177) | Unified event model with categories |
| **Provider Factory** | [agent/config.py](agent/config.py) | [L73-L110](agent/config.py#L73-L110) | `get_model()` abstracts providers |
| **Langfuse Setup** | [agent/base_agent.py](agent/base_agent.py) | [L40-L63](agent/base_agent.py#L40-L63) | Conditional `logfire.instrument_pydantic_ai()` |
| **Deps Mutation** | [agent/tools.py](agent/tools.py) | [L156](agent/tools.py#L156) | `ctx.deps.schema_cache[table_name] = ...` |
| **Chat History** | [agent/deps.py](agent/deps.py) | [L45](agent/deps.py#L45) | Multi-turn via `chat_history` field |
| **Message Conversion** | [agent/base_agent.py](agent/base_agent.py) | [L418](agent/base_agent.py#L418) | `_messages_to_dict()` for persistence |

---

</details>

<details>
<summary>4. Production Comparison</summary>

## 4. Production Comparison

### What makes this Showcase more Production-Ready

| Aspect | Showcase | Production |
|--------|----------|-------------------|
| Database | SQLite (file-based) | PostgreSQL |
| Deps Creation | Simple factory | Factory with Redis caching |
| Prompt System | Simple `instructions` | Manifest + template + sections |
| Tools | Static list | Agent-specific tool mapping |
| Streaming | Chainlit `stream_token` | WebSocket routes |
| Monitoring | nothing | Langfuse |

---

</details>

<details>
<summary>5. Key Takeaways</summary>

## 5. Key Takeaways

### Design Principles

1. **Deps as Working Memory** — Cache schema and history in deps so state persists across turns.
2. **Structured Output** — Use `output_type` models to validate responses when execution is off.
3. **Self-Documenting Models** — `Field(description=...)` gives the LLM reliable schema semantics.
4. **Dynamic Context Injection** — `instructions` recompute prompt context from current deps each turn.
5. **Tool Orchestration + ModelRetry** — Let the agent recover from tool errors and iteratively fix queries.
6. **Streaming Protocol** — `OutputData` events enable real-time UI updates and tool visualization.
7. **Provider Abstraction** — Centralized model factory makes swapping LLMs/configs trivial.

---

</details>

## Resources

- [pydantic-ai Documentation](https://ai.pydantic.dev/)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Chainlit Documentation](https://docs.chainlit.io/)


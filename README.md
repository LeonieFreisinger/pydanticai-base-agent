# SQL Agent Showcase

A standalone demonstration of building AI agents with **pydantic-ai**, designed for teaching core concepts progressively.

## Quick Start

```bash
# 1. Install dependencies and create database
make setup

# 2. Set your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run the chat interface
make run
```

Open http://localhost:8000 in your browser.

## Demo Levels

Control which features are active via `DEMO_LEVEL` environment variable:

| Level | Features | Teaching Focus |
|-------|----------|----------------|
| 1 | Structured output only | Pydantic output validation |
| 2 | + Schema exploration | RunContext, dependency injection |
| 3 | + Full query execution | Tool orchestration, deps mutation |
| 4 | + ModelRetry + Langfuse | Error recovery, observability |

```bash
# Run with specific level
make run-level1  # Structured output
make run-level2  # Schema exploration
make run-level3  # Full queries
make run-level4  # Everything (default)
```

## Project Structure

```
sql_agent_showcase/
├── agent/
│   ├── agent.py       # SQLQueryAgent with DEMO_LEVEL gating
│   ├── base_agent.py  # BaseAgent with streaming + Langfuse
│   ├── config.py      # Provider-agnostic model factory
│   ├── deps.py        # SQLAgentDeps (dependency injection)
│   ├── models.py      # Pydantic models with Field descriptions
│   ├── prompts.py     # Dynamic instructions callables
│   └── tools.py       # SQL tools with ModelRetry
├── database/
│   ├── init_db.py     # Database seeding script
│   └── schema.sql     # Table definitions
├── app.py             # Chainlit chat interface
├── agent.md           # Detailed documentation
├── pyproject.toml     # Dependencies
└── Makefile           # Quick commands
```

## Key pydantic-ai Features Demonstrated

1. **Dependency Injection** - `RunContext[SQLAgentDeps]` in tools
2. **Dynamic Prompts** - `instructions` callables vs static `system_prompt`
3. **Structured Output** - `output_type=StructuredQueryPlan`
4. **ModelRetry** - Error recovery with LLM self-correction
5. **Streaming** - `OutputData` protocol with tool visualization
6. **Observability** - Langfuse integration via logfire

## Try These Demos

### ModelRetry in Action
```
"Show me data from the secret_table"
→ Triggers ModelRetry: table not in allowed list

"SELECT * FROM products WEHRE stock < 10"  
→ Triggers ModelRetry: SQL syntax error, LLM corrects and retries
```

### Multi-Turn Conversation
```
1. "What tables are in the database?"
2. "Tell me more about the products table"
3. "Show me products with low stock"
4. "Which suppliers provide those products?"
```

### Structured Output (Level 1)
```
"Generate a query to find top customers by revenue"
→ Returns StructuredQueryPlan with SQL, tables, complexity assessment
```

## Configuration

See `.env.example` for all options. Key settings:

```bash
# Provider: openai, anthropic, or google
LLM_PROVIDER=openai
MODEL_NAME=gpt-4.1

# Feature level
DEMO_LEVEL=4

# Observability
ENABLE_LANGFUSE=false
```

## Documentation

See [agent.md](agent.md) for comprehensive documentation including:
- Architecture diagrams
- Code walkthrough with "what to point out"
- Pydantic-AI features mapping
- Production comparison with Hugo agent

## License

MIT - Use freely for teaching and learning.

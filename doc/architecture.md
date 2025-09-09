# Architecture

Hobnob is built on top of [LangGraph](https://langchain-ai.github.io/langgraph/) for workflow orchestration and [LangChain](https://python.langchain.com) for LLM integration.

## Core Components

- **FlowRunner** – Orchestrates execution of steps and manages state.
- **Executors** – Pluggable step implementations such as LLM calls or user input.
- **Routers** – Evaluate transition conditions using safe expression engines like JMESPath.
- **TriStore** – Optional persistence layer backed by PostgreSQL with the AGE extension. It stores flows, instances, checkpoints, and events in a graph model for replay and analysis.

The TriStore uses a graph-first schema with vertices like `Flow`, `Step`, and `Instance`, connected by edges such as `HAS_STEP` and `NEXT`. Relational tables and JSONB blobs augment the graph for idempotency and arbitrary metadata.

# Hobnob

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Hobnob** is a declarative workflow engine for building LLM-powered applications. Define complex multi-step workflows using JSON configuration and let Hobnob handle the execution, state management, and flow control.

## Overview

Hobnob enables you to create sophisticated AI workflows without writing complex orchestration code. Simply define your workflow steps, transitions, and conditions in a JSON configuration, and Hobnob will execute them using LangGraph under the hood.

## Key Features

- **Declarative Configuration**: Define workflows in JSON rather than code
- **LLM Integration**: Seamless integration with LangChain LLMs
- **State Management**: Automatic state tracking and updates between steps
- **State Inference**: Optional `infer_state` flag builds the state schema from prompts or sample state
- **Conditional Flow Control**: Dynamic routing based on state conditions with safe expression evaluators
- **User Interaction**: Built-in support for user input steps
- **Structured Output**: Automatic JSON parsing from LLM responses
- **Rich Prompting**: Support for system prompts, context, examples, and instructions
- **Security**: Safe condition evaluation using JMESPath instead of unsafe eval()
- **Observability**: Step-level logs for inputs, outputs, and errors
- **Resilience**: Configurable retry/backoff policies

## Installation

Hobnob uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package and project management.

```bash
uv sync
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Quick Start

```python
from typing import TypedDict, List, Optional
import logging
from langchain_openai import ChatOpenAI
from hobnob import FlowRunner

logging.basicConfig(level=logging.INFO)

# Define your state schema
class MyState(TypedDict):
    count: int
    message: str
    done: bool

# Define your workflow
flow_definition = {
    "system_prompt": "You are a helpful assistant that counts numbers.",
    "steps": [
        {
            "name": "count_step",
            "context": "Count from the current number and provide a message.",
            "instructions": "Increment the count by 1 and create a friendly message.",
            "output_format": "Return JSON with: count, message, done",
            "prompt": "Current count: {count}. Increment by 1 and create a message. Set done=true if count >= 5."
        }
    ],
    "transitions": [
        {"from": "count_step", "to": "count_step", "condition": "not done"},
        {"from": "count_step", "to": None, "condition": "done"}
    ],
    "initial_step": "count_step"
}

# Run the workflow
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
runner = FlowRunner(flow_definition, llm, MyState)

initial_state = {"count": 0, "message": "", "done": False}
final_state = runner.run(initial_state)
```

### TriStore (AGE/Cypher) Persistence

- Graph-first persistence for flows, steps, instances, checkpoints, and events using PostgreSQL + AGE.
- Optional and off by default. Enable by passing `persist_flow=True` and a `TriStore` instance.

```python
from hobnob.tristore import TriStore
from langchain_openai import ChatOpenAI

# Flow must include a stable identity
flow_definition |= {"name": "my_flow", "semver": "1.0.0"}

llm = ChatOpenAI(model="gpt-4o", temperature=0)
store = TriStore()  # reads DSN from HOBNOB_DATABASE_URL or DATABASE_URL
runner = FlowRunner(flow_definition, llm, infer_state=True, persist_flow=True, store=store)

final_state = runner.run(initial_state, tenant="demo")
```

Replay deterministically (temperature=0, same model):

```bash
hobnob replay --instance <uuid>
```

#### Database Setup (AGE/Cypher)

1) Install PostgreSQL + the AGE extension. Ensure the extension is available in your DB.
2) Initialize the graph and tables (one-time):

```bash
export HOBNOB_DATABASE_URL=postgres://user:pass@host:5432/db
hobnob graph-init --graph hob
# or apply the SQL directly:
psql "$HOBNOB_DATABASE_URL" -f sql/001_tristore.sql
```

DSN examples:

```bash
# Local default database
export HOBNOB_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/postgres"

# Custom database/user
export HOBNOB_DATABASE_URL="postgresql://hob_user:s3cret@127.0.0.1:5432/hob_db"
```

Verify AGE is available and working:

```bash
psql "$HOBNOB_DATABASE_URL" -c "CREATE EXTENSION IF NOT EXISTS age;"
psql "$HOBNOB_DATABASE_URL" -c "LOAD 'age';"
psql "$HOBNOB_DATABASE_URL" -c "SELECT create_graph('hob');"
psql "$HOBNOB_DATABASE_URL" -c "SELECT * FROM cypher('hob', $$ RETURN 1 $$) AS (r agtype);"
```

Installing Apache AGE (quick notes):

- Requires PostgreSQL headers and build tools. Extension creation typically needs superuser privileges. On managed services, ensure AGE is supported and enabled by your provider.

Ubuntu/Debian (PG 15):

```bash
sudo apt-get update
sudo apt-get install -y postgresql-15 postgresql-server-dev-15 build-essential git
git clone https://github.com/apache/age.git
cd age
make PG_CONFIG=/usr/lib/postgresql/15/bin/pg_config
sudo make install PG_CONFIG=/usr/lib/postgresql/15/bin/pg_config
# then in the target database
psql "$HOBNOB_DATABASE_URL" -c "CREATE EXTENSION age;"
psql "$HOBNOB_DATABASE_URL" -c "LOAD 'age';"
```

macOS (Homebrew, PG 15):

```bash
brew install postgresql@15 git
brew services start postgresql@15
export PATH="$(brew --prefix)/opt/postgresql@15/bin:$PATH"
git clone https://github.com/apache/age.git
cd age
make PG_CONFIG=$(which pg_config)
make install PG_CONFIG=$(which pg_config)
psql "$HOBNOB_DATABASE_URL" -c "CREATE EXTENSION age;"
psql "$HOBNOB_DATABASE_URL" -c "LOAD 'age';"
```

### Prompt-to-Flow Generation (default authoring)

You can create flow definitions directly from natural language using `from_prompt`.

```python
import json
from langchain_openai import ChatOpenAI
from hobnob import from_prompt

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
flow = from_prompt("A workflow that greets a user and then asks for feedback", llm=llm)
print(json.dumps(flow, indent=2))
```

Refine by sending the JSON back with more instructions:

```python
refined = from_prompt(
    "Add a final step that thanks the user and ends the session. "
    "Here is the current flow:\n" + json.dumps(flow),
    llm=llm,
)
```

### Schema-Free Prompt-Driven Agents

Skip defining a `TypedDict` by relying on prompt placeholders or example state. When `infer_state=True` (default), `FlowRunner` constructs the state schema and warns on unexpected fields.

```python
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
runner = FlowRunner(flow_definition, llm, infer_state=True)
```

### Publishing and Versioning

To guarantee deterministic replay, publish flows with a stable `name@semver` and freeze that spec in the graph.

Python API:

```python
from hobnob import publish_flow
from hobnob.tristore import TriStore

store = TriStore()
summary = publish_flow(flow_definition, store, name="fib_limericks", semver="1.0.0")
print(summary)  # includes flow_id, flow_hash, status
```

CLI (prompt or file):

```bash
# From prompt
hobnob publish --prompt "Fibonacci with limericks" --name fib_limericks --semver 1.0.0

# From JSON file
hobnob publish --file flow.json --name fib_limericks --semver 1.0.0
```

Semver is immutable: attempting to republish a different spec under the same version will fail. Bump `semver` for any change.

### One-shot: Generate + Publish + Run

```bash
hobnob quickrun \
  --prompt "Fibonacci with limericks" \
  --name fib_limericks \
  --semver 1.0.0 \
  --model gpt-4o \
  --temperature 0.0 \
  --tenant demo \
  --state '{"fib_sequence":[1,1],"last_number":1,"done":false,"user_continue":"yes"}'
```

You can also use `--file flow.json` and `--state-file state.json`.

## Step Types

### LLM Steps
Default step type that sends prompts to the LLM and processes JSON responses:

```python
{
    "name": "analyze_data",
    "context": "Background information about the task",
    "instructions": "Specific instructions for the LLM", 
    "output_format": "Expected JSON format description",
    "examples": [{"input": {...}, "output": {...}}],
    "prompt": "Process this data: {data}"
}
```

### User Input Steps
Interactive steps that prompt the user for input:

```python
{
    "name": "get_user_choice",
    "type": "user_input",
    "question": "Would you like to continue? (yes/no): "
}
```

### Web Search Steps
Built-in executor that performs a DuckDuckGo query and stores the summary in state:

```python
{
    "name": "lookup_topic",
    "type": "web_search",
    "query_key": "topic",        # read query from state["topic"]
    "result_key": "summary"       # store results in state["summary"]
}
```

### API Call Steps
Generic HTTP request executor:

```python
{
    "name": "get_joke",
    "type": "api_call",
    "url": "https://api.chucknorris.io/jokes/random",
    "result_key": "joke"
}
```

### Custom Executors
Register domain-specific step types using the `ExecutorRegistry`:

```python
from hobnob import ExecutorRegistry

def add_one_factory(cfg, _runner):
    def _step(state):
        return {**state, "n": state["n"] + 1}
    return _step

ExecutorRegistry.register("add_one", add_one_factory)

# Then reference it in your flow definition:
{"name": "bump", "type": "add_one"}
```

## Flow Configuration

### System Prompt
Optional system-level instructions sent with every LLM request:

```python
{
    "system_prompt": "You are an expert data analyst. Always be precise and thorough."
}
```

### Transitions
Define how the workflow moves between steps:

```python
"transitions": [
    {"from": "step1", "to": "step2"},  # Unconditional
    {"from": "step2", "to": "step3", "condition": "status == 'success'"},  # Conditional
    {"from": "step3", "to": None}  # End workflow
]
```

### Conditional Logic
Hobnob uses safe expression evaluators for conditional transitions:

```python
"condition": "count > 10 and user_continue == 'yes'"
"condition": "not done and len(results) < 5"
"condition": "analysis_score >= 0.8"
```

**Router System**: Hobnob includes multiple expression evaluators for security and flexibility:

- **JMESPathRouter** (default): Uses [JMESPath](https://jmespath.org/) for safe JSON querying
- **EvalRouter**: Legacy evaluator, disabled by default for security
- **Custom Routers**: Register domain-specific condition logic via `RouterRegistry`

```python
from hobnob import RouterRegistry, EvalRouter

# Safe by default - uses JMESPath
flow_definition = {
    "transitions": [
        {"from": "step1", "to": "step2", "condition": "count > 10"},
        {"from": "step2", "to": None, "condition": "done"}
    ]
}

# For development/legacy compatibility - enable unsafe eval
EvalRouter.enabled = True  # Not recommended for production

# Register custom router for domain-specific logic
class CustomRouter:
    def check(self, condition: str, state: dict) -> bool:
        # Your custom logic here
        return True

RouterRegistry.register("custom", CustomRouter())
```

### Retries & Backoff

Configure automatic retries for any step to handle transient failures:

```python
{
    "name": "fetch_data",
    "retry": {"max_attempts": 5, "backoff": 2}
}
```

`max_attempts` controls how many times the step is attempted. `backoff` sets the base delay (in seconds) for exponential backoff between attempts.

## Examples

### Fibonacci with Limericks
A complete example that generates Fibonacci numbers with creative limericks:

```python
from typing import TypedDict, List, Optional
from langchain_openai import ChatOpenAI
from hobnob import FlowRunner

class FibState(TypedDict):
    fib_sequence: List[int]
    last_number: int
    done: bool
    user_continue: str
    limerick: Optional[str]

flow_definition = {
    "system_prompt": "You are a creative mathematician and poet.",
    "steps": [
        {
            "name": "fib_and_limerick",
            "context": "The Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, ...",
            "instructions": "Create a limerick and calculate the next Fibonacci number",
            "output_format": "JSON with: limerick, fib_sequence, last_number, done",
            "prompt": (
                "Current sequence: {fib_sequence}\\n"
                "Last number: {last_number}\\n\\n"
                "1. Write a limerick about {last_number}\\n"
                "2. Calculate next Fibonacci number\\n"
                "3. Set done=true if new number > 10"
            )
        },
        {
            "name": "ask_continue",
            "type": "user_input",
            "question": "Continue? (yes/no): "
        }
    ],
    "transitions": [
        {"from": "fib_and_limerick", "to": "ask_continue"},
        {"from": "ask_continue", "to": "fib_and_limerick", 
         "condition": "user_continue == 'yes' and not done"},
        {"from": "ask_continue", "to": None, 
         "condition": "user_continue == 'no' or done"}
    ],
    "initial_step": "fib_and_limerick"
}

# Run with step callback for monitoring
def print_step(step_name, state):
    print(f"\\n--- {step_name} ---")
    print(f"Sequence: {state.get('fib_sequence', [])}")
    if state.get('limerick'):
        print(f"Limerick:\\n{state['limerick']}")

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
runner = FlowRunner(flow_definition, llm, FibState, on_step=print_step)

initial_state = {
    "fib_sequence": [1, 1],
    "last_number": 1,
    "done": False,
    "user_continue": "yes",
    "limerick": None
}

final_state = runner.run(initial_state)
```

To persist and replay this flow, see `examples/fib_tristore.py` and the TriStore/Publish sections above.

## Architecture

Hobnob is built on top of:
- **LangGraph**: For workflow orchestration and state management
- **LangChain**: For LLM integration and prompt management
- **TypedDict**: For type-safe state schemas

### Core Components

- **FlowRunner**: Main orchestrator that builds and executes workflows
- **PromptRenderer**: Handles enhanced prompt formatting with context and examples
- **JsonParser**: Extracts structured data from LLM responses
- **LLMStep**: Executes LLM-powered workflow steps
- **UserInputStep**: Handles interactive user input

## Use Cases

- **Data Processing Pipelines**: Multi-step analysis and transformation workflows
- **Decision Trees**: Complex conditional logic with user input
- **Multi-Agent Conversations**: Orchestrated interactions between different AI agents
- **Interactive Applications**: Chatbots and assistants with complex state management
- **Content Generation**: Multi-stage content creation with review cycles


## Fixes/Improvements

The current implementation works and demonstrates the dynamic prompt-driven flow pattern, but the following fixes and enhancements are recommended for production use:

### 1. **Better LLM Output Parsing**

* **Problem:** LLM step output parsing is currently forgiving; it silently swallows parsing errors and stores the output as a raw string. This can mask problems or inconsistencies in the LLM response.
* **Solution:** Make parsing strict—fail fast on invalid JSON or unexpected fields. Optionally, support more robust JSON extraction (e.g., via regex for code blocks) and allow for structured error handling or user intervention.

### 2. **Extensible Step Executors**

* **Problem:** Currently, only LLM and user-input steps are implemented. Custom step types (API calls, database operations, function tools, etc.) require code changes.
* **Solution:** Expose an executor registry or plugin system so new step types can be added without modifying core code. Document how to register custom executors.

### 3. **Step/Post-Step Callbacks**

* **Enhancement:** The new `on_step` callback mechanism is essential for traceability, debugging, and user experience. Document its use clearly in the README, and consider supporting hooks for pre-step, post-step, and error events.

### 4. **Improved Error Handling & Logging**

* **Problem:** Minimal error handling is present, and runtime errors are not surfaced in a user-friendly way.
* **Solution:** Add robust logging, clear error messages, and options for graceful recovery, especially for user interaction and LLM failures.

### 5. **Security & Sandboxing**

* **Enhancement:** If user-supplied or LLM-generated flows are supported, sandbox execution (especially any form of evaluation or code execution) to prevent privilege escalation or resource abuse.

### 6. **Validation & Schema Enforcement**

* **Enhancement:** Enforce state schemas more rigorously (with Pydantic or similar) to prevent key errors and maintain predictable step input/output contracts.

### 7. **Testing & Examples**

* **Enhancement:** Add comprehensive tests for each executor, parser, and the overall runner. Include example flows in the repo for common use cases.


## Monitoring & Recovery

Long-running workflows benefit from proactive monitoring and recovery mechanisms:

- Enable logging at `INFO` level to capture step inputs, outputs, and errors.
- Ship logs to a central system for alerting and audit trails.
- Use `retry` policies with exponential backoff to handle transient LLM or tool failures.
- Persist intermediate state so workflows can resume after interruption.
- Run workflows under external supervisors or schedulers that can restart from the last checkpoint.

## Development

This project requires Python 3.13+ and uses modern type hints throughout.

```bash

# Run tests
pytest

# Type checking
mypy hobnob/

# Linting
ruff check hobnob/
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## License and Copyright

MIT License. See [LICENSE](LICENSE.txt) for details.

Copyright © 2025 Iwan van der Kleijn

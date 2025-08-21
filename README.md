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
- **Conditional Flow Control**: Dynamic routing based on state conditions with safe expression evaluators
- **User Interaction**: Built-in support for user input steps
- **Structured Output**: Automatic JSON parsing from LLM responses
- **Rich Prompting**: Support for system prompts, context, examples, and instructions
- **Security**: Safe condition evaluation using JMESPath instead of unsafe eval()

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
from langchain_openai import ChatOpenAI
from hobnob import FlowRunner

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

## Prompt-to-Flow Generation

You can create flow definitions directly from natural language using the
`from_prompt` helper. This allows rapid prototyping of workflows without
hand-writing JSON.

```python
import json
from langchain_openai import ChatOpenAI
from hobnob import from_prompt

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
flow = from_prompt("A workflow that greets a user and then asks for feedback", llm=llm)
print(json.dumps(flow, indent=2))
```

### Refining a Flow

The generated JSON can be refined by sending it back to `from_prompt` with
additional instructions:

```python
refined = from_prompt(
    "Add a final step that thanks the user and ends the session. "
    "Here is the current flow:\n" + json.dumps(flow),
    llm=llm,
)
```

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

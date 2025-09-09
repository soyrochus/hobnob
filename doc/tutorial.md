# Tutorial

This tutorial walks through a minimal Hobnob flow that counts numbers.

```python
from typing import TypedDict
from langchain_openai import ChatOpenAI
from hobnob import FlowRunner

class MyState(TypedDict):
    count: int
    message: str
    done: bool

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

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
runner = FlowRunner(flow_definition, llm, MyState)

initial_state = {"count": 0, "message": "", "done": False}
final_state = runner.run(initial_state)
```

The runner executes the step repeatedly until `done` becomes `True`.

"""
pip install langchain-openai langgraph python-dotenv
export OPENAI_API_KEY=...
"""

import json
from typing import TypedDict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from hobnob import FlowRunner
from hobnob import RouterRegistry, EvalRouter


load_dotenv()

# Enable unsafe eval router (for dev only!)
EvalRouter.enabled = True


class GraphState(TypedDict):
    fib_sequence: List[int]
    last_number: int
    done: bool
    user_continue: str
    limerick: Optional[str]


flow_definition = {
    "system_prompt": (
        "You are a creative mathematician and poet. "
        "You excel at both mathematical calculations and writing entertaining limericks."
    ),
    "steps": [
        {
            "name": "fib_and_limerick",
            "context": "The Fibonacci sequence starts with 1, 1 and each subsequent number "
                       "is the sum of the two preceding ones (1, 1, 2, 3, 5, 8, 13, ...).",
            "instructions": "Create an entertaining limerick and calculate the next Fibonacci number accurately",
            "output_format": "Return valid JSON with exactly these fields: limerick, fib_sequence, last_number, done",
            "examples": [
                {
                    "input": {"fib_sequence": [1, 1], "last_number": 1},
                    "output": {
                        "limerick": "Sample limerick",
                        "fib_sequence": [1, 1, 2],
                        "last_number": 2,
                        "done": False
                    }
                }
            ],
            "prompt": (
                "Current Fibonacci sequence: {fib_sequence}\n"
                "Last number: {last_number}\n\n"
                "TASKS:\n"
                "1. Write a creative limerick about the number {last_number}\n"
                "2. Calculate the next Fibonacci number (sum of last two numbers in sequence)\n"
                "3. Append the new number to the sequence\n"
                "4. IMPORTANT: Set 'done' to true if the NEW number you calculated > 10, otherwise false\n\n"
                "Return your response as valid JSON."
            )
        },
        {
            "name": "ask_user_continue",
            "type": "user_input",
            "question": "Should we continue generating Fibonacci numbers? (yes/no): "
        }
    ],
    "transitions": [
        {"from": "fib_and_limerick", "to": "ask_user_continue"},
        {"from": "ask_user_continue",
         "to": "fib_and_limerick",
         "condition": "user_continue == 'yes' and not done"},
        {"from": "ask_user_continue",
         "to": None,
         "condition": "user_continue == 'no' or done"}
    ],
    "initial_step": "fib_and_limerick"
}

def print_step(step_name, state):
    print(f"\n--- STEP: {step_name} ---")
    if step_name == "fib_and_limerick":
        print("Fibonacci sequence so far:", state["fib_sequence"])
        print("Last number:", state["last_number"])
        print("Limerick:\n", state.get("limerick"))
    elif step_name == "ask_user_continue":
        print(f"User chose: {state['user_continue']}")
    print("-----------------------\n")

if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    initial_state: GraphState = {
        "fib_sequence": [1, 1],
        "last_number": 1,
        "done": False,
        "user_continue": "yes",
        "limerick": None
    }


    # To use eval (for fast iteration or Python-style conditions):
    runner_eval = FlowRunner(
        flow_definition,
        llm,
        GraphState,
        on_step=print_step,
        condition_router=RouterRegistry.get("eval"),
    )

    final_state = runner_eval.run(initial_state)

    print("\n==== FINAL STATE ====")
    print(json.dumps(final_state, indent=2))

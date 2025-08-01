import json
from typing import Dict, Any, Optional, TypedDict, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph import END
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# State schema for LangGraph
class GraphState(TypedDict):
    fib_sequence: List[int]
    last_number: int
    done: bool
    user_continue: str
    limerick: Optional[str]

# ---- JSON configuration (as a Python dict here) ----
flow_definition = {
    "steps": [
        {
            "name": "fib_and_limerick",
            "prompt": (
                "Current Fibonacci sequence: {fib_sequence}. "
                "Last number: {last_number}.\n"
                "Write a silly limerick about {last_number}.\n"
                "Then, calculate the next Fibonacci number and append it to the sequence.\n"
                "If {last_number} > 10, set 'done' to true. Otherwise, set 'done' to false.\n"
                "Return JSON with: limerick, fib_sequence, last_number, done."
            )
        },
        {
            "name": "ask_user_continue",
            "prompt": ""  # This will be handled by direct user input
        }
    ],
    "transitions": [
        {"from": "fib_and_limerick", "to": "ask_user_continue"},
        {"from": "ask_user_continue", "to": "fib_and_limerick",
         "condition": "user_continue == 'yes' and not done"},
        {"from": "ask_user_continue", "to": None,
         "condition": "user_continue == 'no' or done"}
    ],
    "initial_step": "fib_and_limerick"
}

# ---- Core generic runner ----

def render_prompt(prompt: str, state: Dict[str, Any]) -> str:
    # Very basic rendering (Python's str.format)
    return prompt.format(**state)

def parse_json_from_llm_output(output: str) -> Dict[str, Any]:
    # Assume LLM always returns JSON on first code block or line
    import re
    try:
        # Try to extract JSON from code block
        m = re.search(r'\{.*\}', output, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        # Fallback, just try whole output
        return json.loads(output)
    except Exception:
        # As fallback, treat everything as string and return
        return {"output": output.strip()}

def build_graph(flow_definition, llm):
    # Map step names to step configs
    step_map = {s["name"]: s for s in flow_definition["steps"]}
    transitions = flow_definition["transitions"]

    def make_node(step_cfg):
        def node(state):
            step_name = step_cfg['name']
            
            # Special handling for user interaction steps
            if step_name == "ask_user_continue":
                print(f"\nCurrent Fibonacci sequence: {state['fib_sequence']}")
                print(f"Last number: {state['last_number']}")
                
                while True:
                    user_input = input("Should we continue? (yes/no): ").lower().strip()
                    if user_input in ['yes', 'y']:
                        return {**state, "user_continue": "yes"}
                    elif user_input in ['no', 'n']:
                        return {**state, "user_continue": "no"}
                    else:
                        print("Please enter 'yes' or 'no'")
            else:
                # Regular LLM-powered step
                prompt = render_prompt(step_cfg["prompt"], state)
                print(f"\n>>> PROMPT ({step_cfg['name']}):\n{prompt}\n")
                result = llm.invoke(prompt)
                print(f"<<< LLM Output:\n{result.content}\n")
                updates = parse_json_from_llm_output(result.content)
                state = {**state, **updates}
                return state
        return node

    # Build LangGraph StateGraph
    sg = StateGraph(GraphState)
    for step in flow_definition["steps"]:
        sg.add_node(step["name"], make_node(step))

    # Add transitions - group by source node to handle multiple conditions
    transitions_by_source = {}
    for t in transitions:
        src = t["from"]
        if src not in transitions_by_source:
            transitions_by_source[src] = []
        transitions_by_source[src].append(t)
    
    for src, src_transitions in transitions_by_source.items():
        if len(src_transitions) == 1 and not src_transitions[0].get("condition"):
            # Simple unconditional edge
            tgt = src_transitions[0]["to"]
            sg.add_edge(src, tgt if tgt else END)
        else:
            # Conditional edges - create a single routing function
            def make_router(transitions_list):
                def router(state):
                    for t in transitions_list:
                        cond = t.get("condition")
                        tgt = t["to"]
                        if cond:
                            try:
                                if eval(cond, {}, state):
                                    return tgt if tgt else END
                            except Exception as e:
                                print(f"Condition eval error: {e}")
                        else:
                            # Unconditional fallback
                            return tgt if tgt else END
                    return END
                return router
            
            # Get all possible targets for this source
            targets = set()
            for t in src_transitions:
                tgt = t["to"]
                targets.add(tgt if tgt else END)
            
            sg.add_conditional_edges(src, make_router(src_transitions), list(targets))

    sg.set_entry_point(flow_definition["initial_step"])
    return sg.compile()

# ---- Execution ----

def main():
    # Replace with your model or local model
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    

    # Initial state (Fibonacci 1, 1)
    state: GraphState = {
        "fib_sequence": [1, 1],
        "last_number": 1,
        "done": False,
        "user_continue": "yes",
        "limerick": None
    }

    graph = build_graph(flow_definition, llm)
    # Run until END or user stops
    final_state = graph.invoke(state)
    print("\n==== FINAL STATE ====")
    print(final_state)
    print("\nFull Fibonacci sequence:", final_state["fib_sequence"])
    print("\nLast limerick:", final_state.get("limerick"))

if __name__ == "__main__":
    main()

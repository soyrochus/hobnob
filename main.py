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
    "system_prompt": "You are a creative mathematician and poet. You excel at both mathematical calculations and writing entertaining limericks. Always be accurate with math and creative with poetry.",
    "steps": [
        {
            "name": "fib_and_limerick",
            "context": "The Fibonacci sequence starts with 1, 1 and each subsequent number is the sum of the two preceding ones (1, 1, 2, 3, 5, 8, 13, ...).",
            "instructions": "Create an entertaining limerick and calculate the next Fibonacci number accurately",
            "output_format": "Return valid JSON with exactly these fields: limerick, fib_sequence, last_number, done",
            "examples": [
                {
                    "input": {"fib_sequence": [1, 1], "last_number": 1},
                    "output": {
                        "limerick": "There once was a number named one,\nWho started the sequence for fun,\nWith another one too,\nThey made something new,\nAnd Fibonacci's tale had begun!",
                        "fib_sequence": [1, 1, 2],
                        "last_number": 2,
                        "done": False
                    }
                },
                {
                    "input": {"fib_sequence": [1, 1, 2, 3, 5, 8], "last_number": 8},
                    "output": {
                        "limerick": "There once was an eight so great,\nWho knew thirteen would be his fate,\nWith five as his friend,\nThey'd sum to the end,\nAnd crossing ten was their date!",
                        "fib_sequence": [1, 1, 2, 3, 5, 8, 13],
                        "last_number": 13,
                        "done": True
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
            "type": "user_input"  # Mark this as a user interaction step
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

def render_prompt(step_cfg: Dict[str, Any], state: Dict[str, Any], system_prompt: str = "") -> str:
    """Enhanced prompt rendering with system context, examples, and structured formatting"""
    full_prompt = ""
    
    # Add system prompt if provided
    if system_prompt:
        full_prompt += f"SYSTEM: {system_prompt}\n\n"
    
    # Add context if provided
    if step_cfg.get("context"):
        full_prompt += f"CONTEXT: {step_cfg['context']}\n\n"
    
    # Add examples if provided
    if step_cfg.get("examples"):
        full_prompt += "EXAMPLES:\n"
        for i, ex in enumerate(step_cfg["examples"], 1):
            full_prompt += f"Example {i}:\n"
            full_prompt += f"Input: {json.dumps(ex['input'], indent=2)}\n"
            full_prompt += f"Output: {json.dumps(ex['output'], indent=2)}\n\n"
    
    # Add instructions if provided
    if step_cfg.get("instructions"):
        full_prompt += f"INSTRUCTIONS: {step_cfg['instructions']}\n\n"
    
    # Add output format if provided
    if step_cfg.get("output_format"):
        full_prompt += f"OUTPUT FORMAT: {step_cfg['output_format']}\n\n"
    
    # Add the main prompt/task
    if step_cfg.get("prompt"):
        full_prompt += f"CURRENT TASK:\n{step_cfg['prompt'].format(**state)}"
    
    return full_prompt

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
            if step_cfg.get("type") == "user_input" or step_name == "ask_user_continue":
                print(f"\n=== USER INTERACTION ===")
                print(f"Current Fibonacci sequence: {state['fib_sequence']}")
                print(f"Last number: {state['last_number']}")
                if state.get('limerick'):
                    print(f"Latest limerick:\n{state['limerick']}")
                
                while True:
                    user_input = input("\nShould we continue generating Fibonacci numbers? (yes/no): ").lower().strip()
                    if user_input in ['yes', 'y']:
                        return {**state, "user_continue": "yes"}
                    elif user_input in ['no', 'n']:
                        return {**state, "user_continue": "no"}
                    else:
                        print("Please enter 'yes' or 'no'")
            else:
                # Regular LLM-powered step with enhanced prompting
                system_prompt = flow_definition.get("system_prompt", "")
                prompt = render_prompt(step_cfg, state, system_prompt)
                print(f"\n>>> ENHANCED PROMPT ({step_cfg['name']}):")
                print("="*50)
                print(prompt)
                print("="*50)
                
                result = llm.invoke(prompt)
                print(f"\n<<< LLM Output:\n{result.content}\n")
                updates = parse_json_from_llm_output(result.content)
                new_state = {**state, **updates}
                
                # Show what was updated
                print(f"State updates: {updates}")
                return new_state
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

"""CLI example for generating Hobnob flows from natural language prompts."""

import argparse
import json

from langchain_openai import ChatOpenAI

from hobnob import from_prompt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Hobnob flow from a natural language description",
    )
    parser.add_argument("prompt", help="Description of the workflow to build")
    args = parser.parse_args()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    flow = from_prompt(args.prompt, llm=llm)
    print(json.dumps(flow, indent=2))


if __name__ == "__main__":
    main()

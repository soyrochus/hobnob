"""CLI example for generating Hobnob flows from natural language prompts."""

import argparse
import json

# When running this example as a top-level script, the package root might
# not be on sys.path (depending on the runner). Add the project root so the
# local `hobnob` package can be imported without installing the package.
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def load_env_file(env_path: Path) -> bool:
    """Load simple KEY=VALUE pairs from an env file into os.environ.

    Prefer python-dotenv if available. Return True if the file existed and
    something was loaded, False otherwise.
    """
    p = Path(env_path)
    if not p.exists():
        return False

    try:
        # If python-dotenv is installed use it (it handles many edge cases).
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=str(p), override=False)
        return True
    except Exception:
        # fall back to minimal loader to avoid adding a hard dependency
        import os

        text = p.read_text(encoding="utf-8")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            # strip surrounding quotes
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            os.environ.setdefault(key, val)
        return True


def load_envs_candidates() -> list[Path]:
    """Return a list of candidate .env file paths to load (in preference order)."""
    candidates: list[Path] = []
    here = Path(__file__).resolve().parent
    project_root = Path(__file__).resolve().parents[1]
    home = Path.home()

    candidates.append(here / ".env")
    candidates.append(project_root / ".env")
    candidates.append(project_root / ".env.local")
    candidates.append(home / ".env")

    return candidates


def load_any_envs() -> list[Path]:
    """Try loading env files from candidate locations and return loaded paths."""
    loaded: list[Path] = []
    for p in load_envs_candidates():
        try:
            if load_env_file(p):
                loaded.append(p)
        except Exception:
            # don't fail on env parsing errors
            continue
    return loaded



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Hobnob flow from a natural language description",
    )
    parser.add_argument("prompt", help="Description of the workflow to build")
    args = parser.parse_args()
    # Load .env files (example-level, project-level, home) so OPENAI_API_KEY
    # and other settings are available when constructing the LLM.
    loaded_envs = load_any_envs()
    if loaded_envs:
        print(f"Loaded env files: {', '.join(str(p) for p in loaded_envs)}")
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover - friendly runtime message
        raise ImportError(
            "The 'langchain_openai' package is required to run this example. "
            "Install with your package manager (e.g. pip install langchain-openai)"
        ) from exc

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    try:
        from hobnob import from_prompt
    except Exception as exc:  # pragma: no cover - friendly runtime message
        raise ImportError(
            "The local 'hobnob' package (and its dependencies) could not be imported. "
            "If you're running this from the repository, ensure the project root is on PYTHONPATH or install the package. "
            "Also install the package dependencies (e.g. 'langgraph' and 'langchain-openai')."
        ) from exc

    flow = from_prompt(args.prompt, llm=llm)
    print(json.dumps(flow, indent=2))


if __name__ == "__main__":
    main()

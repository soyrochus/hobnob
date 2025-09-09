# Concepts

Hobnob executes workflows described in JSON. A workflow is composed of **steps** and **transitions** that operate on a shared **state**.

- **Step** – A unit of work. Most steps are powered by an LLM, but custom executors and user input steps are also supported.
- **State** – Typed key/value store passed between steps. Hobnob can infer the state schema from prompts or accept an explicit `TypedDict`.
- **Transition** – Routes execution from one step to the next based on boolean conditions evaluated against the current state.
- **Flow** – A collection of steps and transitions with an initial step. The flow definition can be persisted to a TriStore backend for replay and analysis.

Hobnob focuses on declarative configuration so that complex flows can be expressed without imperative code.

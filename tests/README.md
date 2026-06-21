# Testing the Workspace Agent

To maintain strict reproducibility and a highly legible test output, all contributions to the `langgraph-workspace-agent` test suite must adhere to the conventions outlined below. This repository contains three distinct testing tiers: **Unit**, **Integration**, and **Evaluations (Evals)**.

---

## 1. Global Testing Conventions

These rules apply universally across the test suite, specifically targeting our `pytest` executions (`unit` and `integration`).

### Strict Naming Convention
Apply the `test_<target>_<result>_<scenario>` pattern.
* **Why:** This automatically groups related successes and failures together in alphabetical test runner outputs (like `pytest -v`), making it immediately obvious which execution paths are failing. Because the `<target>` explicitly tracks the structural function/component, we are free to organize the file layout behaviorally (see below).
* **Unit Example:** `test_get_execution_llm_success_tier_1`
* **Integration Example:** `test_github_webhook_error_invalid_signature`

### Topological Ordering (Green -> Edge -> Red)
Within every test file, class, or behavioral block, tests must be strictly ordered by execution path:
1. **Green (Happy Path):** The standard, expected successful executions.
2. **Edge (Graceful Fallbacks):** Retries, circuit breakers, or acceptable alternate paths.
3. **Red (Critical Failures):** Terminal errors, validation blocks, and explicit exception raising.

### Behavioral Grouping
Tests must be clustered by the feature, workflow, or logical event they represent, rather than strictly mapping 1:1 to function names.
* **Why:** Behavioral grouping allows the test suite to act as living documentation for the system's workflows. It also makes the tests highly resilient to internal refactoring; breaking a large function into smaller internal helpers shouldn't require tearing apart your test file's structure.
* **Implementation:** Use visual code comments to clearly demarcate these workflow blocks (e.g., `# ========================================== \n# Workflow: Human-in-the-Loop Interrupts`).
* **Exception:** For pure, stateless utility modules (e.g., simple mathematical helpers or file-system wrappers), structural grouping by function name remains acceptable.

---

## 2. Directory-Specific Guidelines

### `tests/unit/`
Unit tests validate isolated functions, routing logic, and state transitions without hitting external services or spinning up the sandbox.
* **Mocking:** All LLM calls, filesystem operations, and Git operations *must* be mocked.
* **Focus:** Validate the orchestrator's graph routing (`route_after_llm`, `route_after_intent`) and error handling logic.

### `tests/integration/`
Integration tests validate the boundaries between the agent and external services (e.g., Telegram, GitHub Webhooks).
* **Scope:** These tests ensure that HTTP payloads are parsed correctly, webhook signatures are validated, and the system can properly convert external events into the internal `AgentState`.
* **Statefulness:** Ensure any persistent memory or database instances (like Redis checkpointers) are mocked or pointed to an ephemeral, isolated test database to prevent state bleeding.

### `tests/evals/`
Evaluations are deterministic scripts used to measure the NLU accuracy and pipeline health of the LLMs against ground-truth datasets using Langfuse.
* **Execution:** Evals are not run via standard `pytest`. They are executed as standalone scripts (e.g., `python tests/evals/run_router_evals.py --eval`).
* **Dataset Prefixing:** To guarantee strict reproducibility and ordering in the evaluation pipeline, all JSON datasets must utilize a consistent numerical prefix system (e.g., `01_router_nlu.json`, `02_e2e_pipeline.json`). 
* **Thresholds:** Eval scripts must contain explicit success thresholds (e.g., `TARGET_THRESHOLD = 100.0` for core intents). If an eval drops below this threshold, the script must exit with a non-zero status code to fail CI.

---

## 3. Running the Tests

**Run all standard tests (Unit & Integration):**
```bash
pytest tests/unit tests/integration -v
```

**Sync and run an evaluation pipeline:**
> **⚠️ Prerequisite:** The evaluation suite requires a local observability stack. Ensure your Langfuse instance and Redis memory store have been spun up (typically via `docker compose up`) and that the Langfuse web UI is actively accepting connections at `http://localhost:3000` before running these scripts.
```
python -m tests.evals.run_router_evals --sync
python -m tests.evals.run_e2e_evals --sync

python -m tests.evals.run_router_evals --eval
python -m tests.evals.run_e2e_evals --eval
```
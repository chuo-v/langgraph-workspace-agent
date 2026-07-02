# 05. Development and Evaluation

This guide is for developers who want to contribute to the LangGraph Workspace Agent, add new MCP tools, or tune the probabilistic LLM routing engine.

Because this orchestrator blends deterministic infrastructure (FastAPI, Docker) with non-deterministic logic (LLM reasoning, routing), the development workflow is split into two distinct phases: **Standard Testing** (Unit/Integration) and **Agentic Evaluation** (Evals).

---

## 1. Setting Up the Local Development Environment

It is recommended to develop on the host machine directly (e.g., macOS) rather than inside the production container to allow for rapid debugging and IDE integration.

**1. Clone and Configure Python:**
Ensure you are using Python 3.11+ (matching the production Docker container).
```bash
git clone git@github.com:chuo-v/langgraph-workspace-agent.git
cd langgraph-workspace-agent
python -m venv venv
source venv/bin/activate

```

**2. Install Development Dependencies:**
Maintain a strict separation between production and development packages. Install the core requirements alongside the testing suite (`pytest`, `ruff`, `pytest-mock`).

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt

```

**3. Test Environment Variables:**
You do not need active API keys or Cloudflare tunnels to run the deterministic tests. You can duplicate `.env.example` to `.env.test` and fill it with dummy data (e.g., `TELEGRAM_BOT_TOKEN=test_token_123`).

---

## 2. Code Quality and Linting

`ruff` is used to guarantee the codebase remains immaculately formatted and highly performant. Before submitting a Pull Request, ensure your code passes the strict formatting checks defined in `pyproject.toml`.

```bash
# Check for linting errors
ruff check .

# Automatically format code
ruff format .


```

*Note: The GitHub Actions CI pipeline will automatically fail if `ruff` detects unformatted code.*

---

## 3. Adding Custom MCP Tools

To expand the agent's capabilities, custom tools can be added to the execution environment.

1. **Create the Tool:** Define the new tool logic in a Python file within the [`src/workspace_agent/tools/`](../src/workspace_agent/tools/) directory. Wrap the primary function using LangChain's `@tool` decorator. A clear, highly descriptive docstring is strictly required, as the execution LLM relies entirely on this docstring to understand when and how to invoke the tool.
2. **Register the Tool:** Once defined, import the new tool and append it to the active list inside [`src/workspace_agent/tools/registry.py`](../src/workspace_agent/tools/registry.py). This step is mandatory; it ensures the orchestrator successfully binds the tool to the LLM during the execution node's lifecycle.

---

## 4. Running the Test Suite (Unit & Integration)

The test suite validates the deterministic components of the agent: webhook ingress, tool execution fallbacks, configuration parsing, and dependency injection.

To execute the standard test suite, run:

```bash
pytest tests/unit/ tests/integration/ -v

```

### Mocking External Infrastructure

Because the agent physically manipulates the host machine and communicates with remote APIs, the test suite utilizes rigorous mocking via `pytest-mock` and `conftest.py`:

* **Telegram Webhooks ([`test_telegram.py`](../tests/integration/test_telegram.py) & [`test_webhook.py`](../tests/integration/test_webhook.py)):** We use `FastAPI.TestClient` to programmatically simulate incoming Telegram JSON payloads. This allows us to test authentication failures (missing `X-Telegram-Bot-Api-Secret-Token`) and concurrency locks without an active internet connection.
* **GitHub Operations ([`test_github_webhook.py`](../tests/integration/test_github_webhook.py) & [`test_github.py`](../tests/unit/test_github.py)):** The `GitPython` CLI bindings and REST API calls are mocked to return simulated JSON responses (e.g., mimicking a `201 Created` status for PR generation) to prevent the tests from generating dummy Pull Requests.
* **Vector Memory Singleton:** In [`conftest.py`](../tests/conftest.py), the `vector_memory` ChromaDB client is swapped with a lightweight dictionary mock. This ensures the `get_hybrid_context` utility can be tested in milliseconds without requiring a live embedding service (cloud or local) or a running database.

---

## 5. Executing Agentic Evaluations (Evals)

Standard unit tests cannot reliably validate non-deterministic LLM logic. If you modify the system prompts in [`prompts.yaml`](../src/workspace_agent/core/prompts.yaml) or change the `Tier 1` routing logic, you must run the **Evaluation Pipeline** to ensure you haven't introduced regressions (e.g., the agent hallucinating workspaces or entering runaway tool loops).

### Evaluation Datasets

Static datasets are maintained in the [`tests/evals/datasets/`](../tests/evals/datasets) directory:

* [`01_router_nlu.json`](../tests/evals/datasets/01_router_nlu.json): Tests the Tier 1 router's ability to correctly classify complex human instructions into the right `intent_category` and map them to the correct `inferred_workspace`.
* [`02_e2e_pipeline.json`](../tests/evals/datasets/02_e2e_pipeline.json): Contains full execution scenarios designed to test edge cases, disambiguation loops, and tracebacks.

> **💡 Note on Dataset Prompts & Local Execution:** You will notice that the prompts within these datasets explicitly append the `/use:standard` command. The Standard Tier (Tier 2) strikes the ideal balance for automated testing: it provides the robust, deterministic reasoning capabilities required for reliable benchmarking, without incurring the premium API costs of the Frontier Tier (Tier 3). Furthermore, standardizing on this tier prevents the evaluation variance that can occur if developers are using different ultra-lightweight cloud models or resource-constrained local models in their Base Tier.
> The evaluation pipeline is deliberately designed to run on the host machine rather than relying on automated GitHub Action runners. This is the easiest approach to maintain when someone else forks the repository and tries to set up the agent for development—it requires zero additional manual setup. It bypasses the need to securely inject sensitive API keys into GitHub Secrets, avoids resource limits on free cloud runners, and gives the developer immediate, zero-latency access to their local Langfuse dashboard (`http://localhost:3000`) for rapid trace inspection.


### Running the Evals

To run the automated scoring pipeline, ensure your local Docker Compose stack is running (specifically the Postgres and Langfuse containers).

```bash
# Evaluate the Tier 1 Router Logic
python tests/evals/run_router_evals.py

# Evaluate the End-to-End Tool Execution
python tests/evals/run_e2e_evals.py

```

### How Evals Work (Langfuse Integration)

1. **Execution:** The script iterates through the prompts in the dataset, injecting them into the LangGraph state machine. (Physical side-effects like file writing and git pushing are mocked out).
2. **Assertion:** The script programmatically compares the agent's final routing decisions and tool choices against the expected outcomes defined in the JSON dataset.
3. **Telemetry & Scoring:** Binary scores (1 for Pass, 0 for Fail) are pushed back to your local Langfuse instance.

### Tuning & Trace Inspection

If an evaluation fails, open your local Langfuse dashboard (`http://localhost:3000`).

1. Navigate to the **Traces** tab.
2. Locate the failed prompt and inspect the visual trace tree.
3. Look at the exact inputs and outputs of the LLM step to identify why it made a poor decision.
4. Adjust the system instructions in `prompts.yaml` or tweak the few-shot examples, then re-run the evaluation script to confirm the regression is fixed.
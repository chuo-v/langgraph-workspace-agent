# 03. Architecture Deep Dive

The LangGraph Workspace Agent is not a simple linear script; it is a fault-tolerant finite state machine designed to run persistently as a background daemon. This document breaks down the core architectural pillars: the LangGraph orchestration, the dual-layer memory system, the Docker-out-of-Docker sandbox, and the local telemetry stack.

---

## 1. The LangGraph Finite State Machine

At the heart of the agent is a Directed Acyclic Graph (DAG) built with LangGraph. Instead of relying on a single, massive LLM context window to handle an entire engineering workflow, the system decomposes tasks into discrete, heavily monitored nodes.

### State Management (`AgentState` & `PRState`)
The agent maintains a strictly typed dictionary (`AgentState`) as its single source of truth during execution. It tracks conversational history using LangGraph's `add_messages` reducer, but also maintains critical metadata outside the message history, such as:
* `intent_category` and `inferred_workspace` (for routing).
* `execution_retry_count` (for sandbox loop termination).
* `t1_base_calls`, `t2_standard_calls`, `t3_frontier_calls` (for telemetry).
* `active_agent_branch` and `pending_pr_url` (for persistent Git lifecycle tracking).

### Node Progression & Key Components

1. **Ingress & Triage (`parse_intent_node`):**
   When a webhook is received, this node intercepts the payload and isolates the most recent conversational turns. It invokes a localized chain that probabilistically determines the user's intent and inferred workspace. It also applies the **Upward Escalation Rule**: if a localized Tier 1 model times out or fails, a custom `ThreadPoolExecutor` gracefully catches the failure and seamlessly escalates the request to a Tier 2 API.
2. **The Execution Loop (`execute_task_node` & `workspace_tools_node`):**
   * **Context Assembly:** `execute_task_node` compiles a hybrid context array, deliberately filtering out past orchestrator success markers to prevent LLM hallucination and context poisoning.
   * **Sequential Execution:** Unlike standard web-search agents that can run tools asynchronously, `workspace_tools_node` enforces **strict sequential tool execution**. This architectural choice actively prevents race conditions when multiple file-write or search-and-replace tools attempt to interact with the same local file simultaneously.
   * **API Resilience:** The execution node wraps API calls in a retry block, injecting system-level recovery messages (e.g., catching JSON truncation errors) rather than crashing the daemon.
3. **Reflection & Verification (`evaluate_diff_node`):**
   Before opening a GitHub Pull Request, the agent invokes an LLM critic. It generates a git diff and asks the critic to verify if the instructions were actually fulfilled. If the critic detects an empty diff or a hallucinated success, it bounces the graph back to the execution node with a system error.
4. **Map-Reduce GitOps (`review_pr_node`):**
   For large repository refactors, standard LLMs fail to summarize massive git diffs. This node dynamically chunks the incremental diff into 40,000-character segments, runs a map-reduce summarization pipeline across the chunks, and synthesizes a highly accurate Pull Request title and body before calling the GitHub REST API.

---

## 2. Dual-Layer Persistence Architecture

Long-running chat interactions inevitably suffer from transformer recency bias. As the context window grows, the agent begins to "forget" earlier constraints or instructions. The system mitigates this via a decoupled memory model:

### Layer 1: Redis Checkpointing (Short-Term Thread State)
The orchestrator uses a local `redis:alpine` container as a LangGraph checkpointer.
* **State Suspension:** When the graph hits a `human_node` or requires disambiguation (`clarification_node`), the precise state of the graph is serialized to Redis. The execution thread terminates, freeing up system resources.
* **Asynchronous Resumption:** When the user replies via Telegram hours later, the orchestrator retrieves the exact graph state from Redis and resumes the loop exactly where it left off.

### Layer 2: Hybrid Long-Term Memory (Store & Vector)
In the background, the asynchronous `update_memory_node` constantly observes the conversation and extracts structured facts using `with_structured_output`.
* **Entity Memory (LangGraph Store):** Explicit user preferences, active projects, and core interests are saved to the local `InMemoryStore` and mirrored to `user_profile.json`.
* **Episodic Memory (ChromaDB):** As the agent interacts with sandboxes, it autonomously discovers systemic rules or codebase constraints (e.g., "The authentication service in Repo A requires a specific mocked token for local testing"). These **"Operational Insights"** are embedded and saved to a local ChromaDB instance. The `get_hybrid_context` utility searches this vector database ahead-of-time and injects relevant historical insights directly into the system prompt for future tasks.

---

## 3. Security: The Ephemeral Execution Sandbox

Standard developer agents execute raw code or shell commands directly on the host machine. This presents a critical vulnerability to prompt-injection attacks. This orchestrator utilizes a **Docker-out-of-Docker (DooD)** model to enforce an immutable security boundary.

* **The Mechanism:** When the LangGraph worker needs to compile a LaTeX document or execute a Python script, it uses the host's Docker socket to programmatically spawn a temporary, ephemeral container (e.g., using `Dockerfile.sandbox`).
* **Path Resolution & Whitelisting:** The filesystem MCP server aggressively sanitizes inputs. It uses Python's modern `pathlib` module (`.resolve()` and `.is_relative_to()`) to strictly verify that requested mounts reside within the `allowed_paths` defined in `config.yaml`. Any attempt to traverse upward (`../../etc/passwd`) immediately raises a fatal security exception.
* **Controlled Side Effects:** The container bind-mounts *only* the specific authorized workspace directory. The agent executes the command, captures `stdout` and `stderr` (returning tracebacks to the LLM for self-correction), and then the container is instantly destroyed. Any modified files naturally persist on the host drive via the volume mount.

---

## 4. Telemetry and Observability Stack

Because the daemon runs headless, robust observability is critical for debugging routing failures or token bloat. The project includes a dedicated, containerized telemetry stack using **Langfuse**.

* **The Stack:** A local PostgreSQL instance stores the relational traces, while ClickHouse handles high-volume analytical queries for the Langfuse web UI (`http://localhost:3000`).
* **Trace Injection:** Using LangChain's callback system (`callbacks.py`), every LLM interaction, prompt wrapper, and tool execution is silently intercepted.
* **Metrics Tracking:** The `AgentState` explicitly tracks `t1_base_calls`, `t2_standard_calls`, and `t3_frontier_calls`. This allows the agent to transparently report the exact computational cost of an execution loop to the user via Telegram upon task completion, while providing the developer with deep visual trace trees in the local Langfuse dashboard to pinpoint prompt degradation.
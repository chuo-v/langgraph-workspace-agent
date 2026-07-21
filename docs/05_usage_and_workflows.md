# 05. Usage and Workflows

Once your background daemon is running and the Telegram webhook is active, the agent acts as your detached, mobile-first engineering partner. This guide covers how to communicate with the agent, explicitly control its reasoning engines, manage conversational threads, connect external IDEs over MCP, and understand the lifecycle of a task from initial prompt to merged code.

## Interacting via the Telegram Interface

The entire user interface lives within your private Telegram chat with the bot. Because the ingress router uses a Natural Language Understanding (NLU) layer, you do not need to use rigid CLI commands.

### 1. Conversational Queries vs. Code Execution
The Tier 1 routing node evaluates your input to decide if you are asking a general question or requesting a file operation.
* **Conversational Bypass:** If you ask, *"What is a Kolmogorov-Arnold Network?"*, the router intercepts this, bypasses all Git tools, and replies natively in the chat to save tokens and prevent accidental workspace modifications.
* **Workspace Inference:** If you ask, *"Add a new retry mechanism to the API client,"* the router maps this to the `inferred_workspace` defined in your `config.yaml` based on context, syncs the repository, and begins execution.

### 2. Disambiguation (Human-in-the-Loop)
If your request is ambiguous (e.g., *"Fix the typo in the README"* when you have three configured workspaces), the agent's graph suspends execution. It will send a Telegram message listing your available workspaces or matching files and wait for you to reply with a clarification before proceeding.

### 3. File Attachments (Text-Based Context)
The agent supports reading text-based file attachments directly from the Telegram chat (e.g., logs, scripts, or PDFs) to inject additional context into the execution loop. Note that while the Telegram app allows you to send multiple files at once, **the agent gateway is designed to process only one attached file per prompt turn**. To ensure the attached context binds correctly to your instruction without triggering concurrency locks, please send files individually rather than in a multi-document batch.

---

## Explicit Execution Tier Overrides

While the router automatically assesses `task_complexity` and escalates to higher tiers autonomously, you maintain ultimate manual control over the compute tier used for the execution loop.

You can force the agent to use a specific model tier by appending a unified `/use:` flag anywhere in your Telegram message. The parser strips the flag cleanly before passing the instruction to the LLM.

* `/use:standard` **Flag:** Forces the orchestrator to use the Tier 2 API (e.g., DeepSeek). Best for moderate tasks spanning multiple files.
  > *"Update the telemetry counters in the routing node /use:standard"*
* `/use:frontier` **Flag:** Forces the orchestrator to use the Tier 3 API (e.g., Gemini 1.5 Pro or Claude 3.5 Sonnet). Reserved for complex architectural refactors or highly abstract reasoning.
  > *"Refactor the entire GitHub webhook integration to use Pydantic models for payload validation instead of raw dicts /use:frontier"*
* `/use:model_alias` **Flag:** Bypasses the tier logic entirely to explicitly lock execution to a specific model alias mapped in your `config.yaml`.
  > *"Translate these comments to Japanese /use:gemini_pro"*

*(Note: If you do not provide a flag, the system defaults to Tier 1 unless the internal router flags the prompt as highly complex.)*

---

## Thread Management & Task Control

The agent uses Redis checkpointer states to maintain conversational context, while offering control commands to halt execution or abandon dirty states.

* **/reset Command:** Clears the active LangGraph thread state and short-term memory in Redis. Use this when the agent gets stuck on stale conversation context and you want a fresh chat session without touching local Git branches.
* **/cleanup Command:** Aborts the active workflow, clears short-term memory, switches the local repository back to the target default branch (e.g., `main`), and forcefully deletes the temporary agent branch (`agent/*`). Use this when you want to abandon a task entirely.
* **In-Flight Interruption (`/stop`, `/abort`, `/cancel`, `/halt`):** If the agent is currently executing tools in a long-running loop, sending any of these flags in Telegram instructs the orchestrator to pause execution cleanly at the next tool checkpoint without corrupting repository state.

---

## Model Context Protocol (MCP) IDE Integration

In addition to autonomous Telegram ingress, the agent exposes its native tool engine (filesystem operations, sandbox execution, and Git controls) via a standalone **Model Context Protocol (MCP)** server (`src/workspace_agent/tools/mcp_server.py`). This allows external desktop AI tools (e.g., Claude Desktop, Cursor, or Zed) to directly leverage the agent's whitelisted sandbox tools.

### Claude Desktop Configuration
Add the workspace agent MCP server to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "workspace-agent-tools": {
      "command": "/path/to/langgraph-workspace-agent/.venv/bin/python",
      "args": [
        "-m",
        "src.workspace_agent.tools.mcp_server"
      ],
      "env": {
        "PYTHONPATH": "/path/to/langgraph-workspace-agent",
        "ALLOWED_PATHS": "/Users/username/git",
        "WORKSPACE_AGENT_CONFIG_PATH": "/path/to/langgraph-workspace-agent/config.yaml"
      }
    }
  }
}
```

When connected, your desktop AI client can invoke sandboxed test runners, LaTeX compilers, and structured search tools over standard I/O (`stdio`).

---

## The Lifecycle of a Task

When you request a code change, the LangGraph state machine moves through a strict, self-healing pipeline. Here is exactly what happens under the hood during a typical request:

### Phase 1: Preparation & Execution
1. **Sync Lock:** The agent immediately locks the conversational thread (`is_busy: true`) and performs a fast-forward `git pull` on your target branch to ensure it is working with the latest code.
2. **Context Injection:** It queries ChromaDB for relevant "Operational Insights" and injects them into the system prompt alongside the explicit boundaries of your `allowed_paths`.
3. **Sequential Tool Loop:** The execution LLM explores the codebase, reads files, and executes edits sequentially to prevent race conditions. If a sandbox tool (like a Python execution or LaTeX compilation) throws a traceback, the agent automatically captures the `stderr` and attempts to self-heal the code up to `max_sandbox_retries`.

### Phase 2: Reflection & Evaluation
Before it even considers touching your Git history, the agent runs a **Reflection Node**.
1. It generates a raw `git diff` of the uncommitted changes.
2. An independent Tier 1 "Critic" model compares this diff against your original Telegram instruction.
3. If the critic detects that the instructions were ignored or the diff is empty (hallucinated success), it bounces the agent back to the execution phase with a harsh "System Rejection" message.

### Phase 3: Pull Request Generation
Once the critic approves the changes:
1. **Map-Reduce Summarization:** For large changes, the agent chunks the git diff and uses a map-reduce pipeline to generate a highly semantic PR title and description.
2. **Branch & Commit:** It commits the changes locally to a generated branch (e.g., `agent/add-retry-logic-a1b2`).
3. **Push & Notify:** It pushes the branch, opens the GitHub Pull Request, and pings your chat with the URL.
   > *"✅ **Execution Complete.** I have pushed the changes to a new branch and opened a Pull Request for your review: [🔗 Link]"*

### Phase 4: Agentic CI/CD & Iteration
1. **Test Execution:** If you configured `ci_suites` in your `config.yaml`, the agent automatically runs those shell commands in a sandboxed subprocess.
2. **Status Checks:** It posts the Pass/Fail results directly to the GitHub PR via the REST API, including collapsible logs in a PR comment.
3. **Human Iteration:** You can review the PR on your phone. If you want changes, simply reply in the chat (e.g., *"Change the timeout from 30s to 60s"*). The agent uses the `active_agent_branch`, applies the fix, and silently patches the existing PR.

### Phase 5: Cleanup
Once you manually merge the Pull Request via the GitHub UI, the configured GitHub webhook autonomously notifies the agent. The agent's `pr_merged_node` activates in the background, cleanly resetting your local host machine back to the target branch and safely deleting the temporary agent branch to prevent repository clutter. Once the cleanup is complete, the agent will send a final confirmation message in the chat.
# 02. Configuration

The LangGraph Workspace Agent is designed to be highly secure and cost-efficient. To achieve this, it relies on strict configuration parameters defined in your `config.yaml` file. This guide explains how to configure your operational settings, map your repositories, define your LLM routing tiers, and enforce security boundaries for the execution sandbox.

## The `config.yaml` Breakdown

Your `config.yaml` acts as the single source of truth for the agent's environmental awareness. It tells the agent which repositories exist on the host machine, what models to use for different tasks, and which directories it is physically allowed to access.

To get started, copy the template file in your deployment repository:
```bash
cp config.example.yaml config.yaml
```

### Example Configuration

Here is a comprehensive example based on a typical workspace layout:

```yaml
# ==========================================
# AGENT SETTINGS
# ==========================================
agent:
  # The default branch the agent will sync and branch off from
  target_branch: "main"
  # List of GitHub usernames authorized to trigger the agent via PRs or ChatOps comments
  allowed_github_users:
    - "your_github_username"
  # How many times the agent can retry fixing a tracebacked sandbox execution
  max_sandbox_retries: 3
  # Appends a footer to Telegram messages showing LLM API call counts
  show_telemetry: true
  # Text prepended to PR titles, PR comments, and commit messages (leave empty for none)
  agent_prefix: "🤖"

# ==========================================
# LLM ROUTING & TIERS
# ==========================================
llm:
  base_tier:
    default_model: "gemini_flash"
    available_models:
      gemini_flash:
        provider: "gemini"
        model_name: "gemini-2.5-flash"
      qwen_local:
        provider: "ollama"
        model_name: "qwen2.5:32b"

  standard_tier:
    default_model: "deepseek_flash"
    available_models:
      deepseek_flash:
        provider: "deepseek"
        model_name: "deepseek-v4-flash"

  frontier_tier:
    default_model: "claude"
    available_models:
      claude:
        provider: "anthropic"
        model_name: "claude-sonnet-4-6"
      deepseek_pro:
        provider: "deepseek"
        model_name: "deepseek-v4-pro"
      gemini_pro:
        provider: "gemini"
        model_name: "gemini-2.5-pro"

# ==========================================
# FILESYSTEM SANDBOX
# ==========================================
allowed_paths:
  - "/Users/vernon/git/langgraph-workspace-agent-dev"
  - "/Users/vernon/git/langgraph-workspace-agent-deployment"
  - "/Users/vernon/git/langgraph-workspace-agent-sandbox"

# ==========================================
# WORKSPACE MAPPINGS
# ==========================================
workspaces:
  langgraph-workspace-agent-dev:
    description: "The core LangGraph orchestration architecture and webhook. This is a development clone of the main langgraph-workspace-agent repository."
    path: "/Users/vernon/git/langgraph-workspace-agent-dev"
    target_branch: "develop"
    pre_commit_suites:
      - name: "Ruff Format"
        command: "docker run --rm -v $(pwd):/app -w /app ghcr.io/astral-sh/ruff:latest format ."
        timeout_seconds: 30
      - name: "Ruff Check & Fix"
        command: "docker run --rm -v $(pwd):/app -w /app ghcr.io/astral-sh/ruff:latest check --fix"
        timeout_seconds: 45
    ci_suites:
      - name: "NLU Component Evaluator"
        command: "python -m tests.evals.run_router_evals --eval"
        timeout_seconds: 120
      - name: "E2E Pipeline Evaluator"
        command: "python -m tests.evals.run_e2e_evals --eval"
        timeout_seconds: 600

  langgraph-workspace-agent-deployment:
    description: "The private GitOps deployment repository for the workspace agent's live configuration and secrets."
    path: "/Users/vernon/git/langgraph-workspace-agent-deployment"

  langgraph-workspace-agent-sandbox:
    description: "A dedicated private sandbox for functional testing and validating agent capabilities across file editing, GitOps workflows, and sandboxed script execution."
    path: "/Users/vernon/git/langgraph-workspace-agent-sandbox"
    target_branch: "develop"
    pre_commit_suites:
      - name: "Ruff Format"
        command: "docker run --rm -v $(pwd):/app -w /app ghcr.io/astral-sh/ruff:latest format ."
        timeout_seconds: 30
      - name: "Ruff Check & Fix"
        command: "docker run --rm -v $(pwd):/app -w /app ghcr.io/astral-sh/ruff:latest check --fix"
        timeout_seconds: 45
    ci_suites:
      - name: "Data Processor Unit Tests"
        command: "python tests/test_data_processor.py"
        timeout_seconds: 120
      - name: "Documentation Validation"
        command: "python tests/test_documentation.py"
        timeout_seconds: 120
```

### Understanding Key Settings

* **Authorization (`allowed_github_users`):** This is a critical security boundary. When managing public repositories, anyone can fork your code and submit a Pull Request. This whitelist ensures that only you (and authorized collaborators) can trigger the local daemon to execute code via GitHub webhooks or `/retest` ChatOps comments.
* **Agent Identity (`agent_prefix`):** This prefix (e.g., an emoji like `🤖` or a tag like `[Agent]`) is prepended to any commit messages, PR titles, and PR comments generated by the agent, making it easy to distinguish autonomous actions from human development.
* **Pre-commit vs. CI Suites:**
  * `pre_commit_suites`: These run *before* the agent creates a commit. They are ideal for formatters (e.g., Ruff, Black) and type-checkers (e.g., MyPy) to ensure the agent's code adheres to your repository's style guidelines. Unlike relying solely on remote GitHub Actions, running these locally allows the agent to capture any error tracebacks and autonomously self-heal the code *before* pushing the branch.
  * `ci_suites`: These run *after* the agent completes a task to evaluate functional success (e.g., unit tests, E2E pipelines). If a CI suite fails, the agent will attempt to self-correct up to the `max_sandbox_retries` limit.



## Configuring LLM Routing & Tiers

To optimize costs and speed, the orchestrator divides operations across three compute tiers defined in the `llm` block.

* **Base Tier (Tier 1):** This tier expects a fast, low-cost model (e.g., `gemini-2.5-flash` via a cloud provider, or `qwen2.5:32b` via local Ollama). The agent exclusively uses this tier for initial intent parsing, conversational chat, memory extraction, and simple PR evaluation.
* **Standard Tier (Tier 2):** Fast, moderate-cost API execution layer. While any supported cloud provider can be mapped to this tier, it is typically populated with low-latency, balanced models (such as `deepseek-v4-flash` or `gpt-5.4-mini`) to handle iterative, multi-file execution tasks cleanly without consuming the high-cost tokens of the frontier layer.
* **Frontier Tier (Tier 3):** Premium-tier complex reasoning layer. Reserved strictly for high-complexity tasks, repository-wide architectural refactors, or advanced structural reasoning. Any supported cloud provider can be configured here based on preference, though it is typically populated with top-tier reasoning engines (such as `claude-sonnet-4-6`, `gemini-2.5-pro`, or `deepseek-v4-pro`).

### How Tier Escalation Works

Unlike systems that require manual model selection for every prompt, the ingress router handles this autonomously:

1. **Dynamic Task Classification:** When a message arrives, the Tier 1 model classifies the `task_complexity`. If the task is deemed highly complex, the execution phase is automatically escalated to the Frontier Tier.
2. **Upward Fault Tolerance & Fallbacks:** The orchestrator utilizes a hardcoded "Upward Escalation Rule" to prevent the daemon from hanging due to missing configurations or routing timeouts. If a tier lacks the required API keys, or if the base model times out during the initial intent parsing phase, the system automatically falls back to the next available tier (Tier 2). *(Note: This automatic fallback protects the ingress routing and initialization phases; sudden runtime connection drops during the actual task execution will trigger a safety retry loop rather than an automatic tier escalation)*.
3. **Manual Overrides & Specific Model Selection:** You can manually force a tier escalation via Telegram by including `/use:standard` or `/use:frontier` anywhere in your prompt. Alternatively, if you want to bypass the tier system entirely and target a specific model defined in your config, you can use the `/use:alias` syntax (e.g., `/use:claude` or `/use:deepseek`).
4. **Human-in-the-Loop Disambiguation:** If the Tier 1 router determines your request is too ambiguous to map to a workspace, it suspends execution and sends a clarifying question back to Telegram before spending tokens on an execution model.

## Whitelisting Repository Paths (The Sandbox Boundary)

Allowing an autonomous agent to execute Python code or compile documents directly on your host machine introduces massive security risks. This architecture mitigates this using a **Docker-out-of-Docker (DooD)** execution engine.

The `allowed_paths` array in your configuration serves as an important constraint within this isolation model:

1. **Strict Mounts:** The agent's filesystem tools use the `pathlib` module to strictly resolve file requests. The ephemeral Docker containers will **only** bind-mount the specific absolute paths declared in `allowed_paths`.
2. **Traversal Prevention:** If the LLM generates a script attempting to access files outside the workspace (e.g., reading `../../../../etc/passwd` or your global `~/.ssh/` keys), the system intercepts the attempt and raises a permission error before execution occurs.
3. **PR Gates:** The agent limits its operational reach to whitelisted repositories and relies on Pull Request creation rather than direct upstream branch commits. While not a definitive silver bullet for security, keeping a human in the loop for code approvals significantly balances operational utility with code oversight.

> **⚠️ OS Pathing Caveat (macOS vs. Linux/WSL)**
> Because the sandbox utilizes the host's Docker socket to spawn containers, the absolute paths defined in your `config.yaml` **must perfectly match** the actual paths mounted in your `docker-compose.yml`.
> * The examples above use macOS paths (e.g., `/Users/username/...`).
> * If the agent is deployed on **Linux or Windows (WSL)**, ensure all paths in `config.yaml` and `.env` are updated to reflect `/home/username/...` instead.
> 
> 
> If the configured paths do not exactly match the host OS structure, the ephemeral sandbox will fail to resolve the volume bindings, resulting in empty directory mounts and immediate "file not found" errors during tool execution. Always use absolute paths to guarantee reliable path resolution.
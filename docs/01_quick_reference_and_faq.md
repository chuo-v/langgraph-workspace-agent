# 01. Quick Reference & FAQ

## ⚡ Common Scenarios: How Do I...?

### I would like to update a file in one of my GitHub repositories.

First, make sure the target repository is properly whitelisted in your `config.yaml` under both `allowed_paths` and `workspaces`. Then, simply open your Telegram chat and tell the agent what you want to do using natural language (e.g., *"Update the retry logic in the api_client.py file in the example-project"*). The agent will pull the latest code, make the edits in an isolated sandbox, and open a Pull Request for your review.

### I need the agent to perform a massive, highly complex architectural refactor.
By default, the agent uses a probabilistic router to assign tasks to cost-efficient models. To force the agent to use your most powerful premium reasoning model for a complex task, append the `/use:frontier` flag anywhere in your Telegram message.

### I want to iterate on a Pull Request the agent just opened.

Reply directly in the Telegram chat with your feedback (e.g., *"Change the timeout from 30s to 60s"*). The agent will update its active branch and seamlessly patch the existing PR.

### I want to pause or stop an active task that is currently executing.

If the agent is in the middle of a long execution loop and you want to interrupt it without losing your branch context, send `/stop`, `/abort`, `/cancel`, or `/halt` in Telegram. The orchestrator will safely halt execution at the next checkpoint and wait for your next command.

### I want to completely abandon a task and clean up temporary local branches.

Send `/cleanup` in Telegram. Unlike `/reset`, which only wipes conversational memory, `/cleanup` aborts the current task, switches your local Git repository back to the target default branch, and forcefully deletes the temporary agent branch (`agent/*`).

### I want to retrigger the agentic CI/CD tests on a Pull Request.

Comment `@agent retest` or `@agent /retest` (substituting the "agent" part for your `chatops_name` if you configured a custom one) on the PR in GitHub to trigger the agentic CI pipeline to re-evaluate the code against your test suites.

**💡 Multi-Agent Tip:** If multiple team members run independent instances of the agent on a shared repository, you can prevent command collisions by targeting your specific instance using your username syntax: `@agent-{github_username} retest`.

### I want to force the agent to use a specific model for a single prompt.
If you have configured specific model aliases in your `config.yaml`, you can bypass the tier system entirely by using the alias flag (e.g., appending `/use:gemini_pro` or `/use:qwen_local` to your message).

---

## 🛠️ Troubleshooting & FAQ

### The agent is looping, hallucinating, or incorrectly responding to previous requests instead of my current one. How do I fix this?
The agent maintains deep conversational memory using a Redis checkpointer. If it gets confused or stuck on stale context:
* Send **`/reset`** to wipe short-term thread memory and start a completely fresh conversational session.
* Send **`/cleanup`** if you also want to abandon the current task, delete the temporary Git branch, and revert your local workspace state.

### The agent aborted the workflow and gave me a "Security Exception" or "PermissionError". Why?
To protect your host machine, the agent operates in an immutable Docker-out-of-Docker sandbox that can only access explicitly authorized directories. Check your `config.yaml` and ensure the exact absolute path of your target file is covered by the `allowed_paths` array. The agent cannot traverse upward or access hidden system files like `.ssh` or `.aws`.

### I commented on a PR to trigger the agent, but it completely ignored me.
The agent employs a strict security whitelist to prevent unauthorized users from executing code on your host machine. Verify that your exact GitHub handle is listed in the `allowed_github_users` array within your `config.yaml`.

### The agent replied with "Execution Aborted: Circuit breaker triggered". What does this mean?
The agent is restricted by the `max_consecutive_tool_steps` limit to prevent runaway loops (the default is 60 steps). If this triggers, the task was likely too complex for the current model tier to solve in one go. Try breaking your request into smaller, sequential steps, or escalate the task using the `/use:frontier` flag.

### My tests keep failing in the CI suite, but they pass locally. Why?
The agent runs CI suites in completely isolated, ephemeral `/tmp` workspaces and builds a fresh virtual environment using `uv`. Ensure that your `requirements.txt` or `requirements-dev.txt` is fully up-to-date and contains all necessary dependencies for your project, as the agent relies on these files to construct the sandbox environment.

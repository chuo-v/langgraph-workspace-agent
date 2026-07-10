import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid

import psutil
import redis
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.workspace_agent.core.config import settings
from src.workspace_agent.core.prompt_manager import PromptManager
from src.workspace_agent.core.state import AgentState, PRState
from src.workspace_agent.orchestrator.router import (
    TIER_BASE,
    get_execution_llm_sequence,
)
from src.workspace_agent.tools.github import (
    cleanup_local_branch,
    comment_on_pull_request,
    create_branch_and_commit,
    get_git_diff,
    get_git_diff_blueprint,
    is_diff_empty,
    open_pull_request,
    set_commit_status,
    update_pull_request,
)
from src.workspace_agent.tools.registry import execute_tool_call

from .execution import _build_fallback_chain

# ==========================================
# Module Initialization & Constants
# ==========================================
__all__ = [
    "compile_node",
    "run_pre_commit_node",
    "evaluate_diff_node",
    "agentic_ci_node",
    "review_pr_node",
    "pr_merged_node",
]

MAX_DIFF_LENGTH = 40000
MAX_GITHUB_COMMENT_LENGTH = 60000

try:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = redis.Redis.from_url(redis_url)
except Exception as e:
    print(f"Warning: Failed to initialize Redis in nodes.py: {e}")
    redis_client = None


# ==========================================
# Validation & Pre-Commit Nodes
# ==========================================


def compile_node(state: PRState, config: RunnableConfig = None) -> dict:
    """
    Automated graph node that forcefully compiles any modified LaTeX files.
    Provides an autonomous feedback loop to the LLM if compilation fails.

    State Transitions:
    - On success: Clears 'modified_tex_files' and nullifies 'latest_traceback_error'.
    - On failure: Injects error messages back to the LLM and increments
      'execution_retry_count', or aborts if max retries are exceeded.
    """
    files_to_compile = state.get("modified_tex_files", [])
    retry_count = state.get("execution_retry_count", 0)

    errors = []
    for file_path in files_to_compile:
        tc = {"name": "compile_latex_document", "args": {"tex_file_path": file_path}}
        res = execute_tool_call(tc, config)

        if "Compilation Finished (Exit Code: 0)" not in res:
            errors.append(f"File {file_path}:\n{res}")

    if errors:
        retry_count += 1
        error_str = "\n\n".join(errors)

        if retry_count >= settings.agent.max_sandbox_retries:
            abort_msg = (
                f"⚠️ **Compilation Failed**\n"
                f"I was unable to fix the LaTeX syntax after {retry_count} attempts.\n\n"
                f"**Latest Error:**\n`{error_str}`\n\n"
                f"*Workflow safely aborted.*"
            )
            return {
                "messages": [AIMessage(content=abort_msg)],
                "is_aborted": True,
                "latest_traceback_error": None,
                "execution_retry_count": retry_count,
            }
        else:
            msg = PromptManager.get("compilation", "latex_rejection", error_str=error_str).strip()

            return {
                "messages": [HumanMessage(content=msg)],
                "latest_traceback_error": "latex_compilation_error",
                "execution_retry_count": retry_count,
            }

    return {
        "modified_tex_files": [],  # Successfully compiled, clear the queue
        "latest_traceback_error": None,
    }


def run_pre_commit_node(state: PRState, config: RunnableConfig = None) -> dict:
    """
    Executes pre-commit tools (e.g., formatters, linters, type-checkers).
    If a tool fails and cannot auto-fix, it routes the error back to the LLM.

    State Transitions:
    - On success: Nullifies 'latest_traceback_error'.
    - On failure: Injects subprocess tracebacks as HumanMessages and increments
      'execution_retry_count', aborting if exceeded.
    """
    target_path = state.get("workspace_absolute_path")
    retry_count = state.get("execution_retry_count", 0)

    workspace_config = next(
        (ws for ws in settings.workspaces.values() if ws.path == target_path), None
    )

    # Bypass if not configured for this specific workspace
    if not workspace_config or not workspace_config.pre_commit_suites:
        return {"latest_traceback_error": None}

    expanded_target_path = os.path.expanduser(target_path)
    errors = []

    for suite in workspace_config.pre_commit_suites:
        try:
            process = subprocess.run(
                suite.command,
                cwd=expanded_target_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=suite.timeout_seconds,
                check=False,
            )

            # Non-zero exit code indicates an unfixable error (e.g., strict typing issue)
            if process.returncode != 0:
                errors.append(f"Suite '{suite.name}' failed:\n{process.stdout}\n{process.stderr}")

        except Exception as e:
            errors.append(f"Suite '{suite.name}' OS Error: {str(e)}")

    if errors:
        retry_count += 1
        error_str = "\n\n".join(errors)

        if retry_count >= settings.agent.max_sandbox_retries:
            abort_msg = (
                f"⚠️ **Pre-Commit Checks Failed**\n"
                f"I was unable to satisfy the repository's pre-commit rules after "
                f"{retry_count} attempts.\n\n"
                f"**Latest Error:**\n`{error_str}`\n\n"
                f"*Workflow safely aborted.*"
            )
            return {
                "messages": [AIMessage(content=abort_msg)],
                "is_aborted": True,
                "latest_traceback_error": None,
                "execution_retry_count": retry_count,
            }

        # Inject the pre-commit feedback directly into the agent's context
        msg = PromptManager.get("evaluation", "pre_commit_rejection", error_str=error_str).strip()
        return {
            "messages": [HumanMessage(content=msg)],
            "latest_traceback_error": "pre_commit_error",
            "execution_retry_count": retry_count,
        }

    # Successfully passed all checks
    return {"latest_traceback_error": None}


# ==========================================
# Evaluation Nodes
# ==========================================


def evaluate_diff_node(state: PRState, config: RunnableConfig = None) -> dict:
    """
    Reflection Node: Uses git diff and an LLM critic to verify the agent actually
    completed the requested task before opening a PR.

    State Transitions:
    - On valid pass: Clears traceback error; converts intents to read_only if an
      escape hatch was explicitly used.
    - On verification failure: Appends rejection reasoning to messages and
      increments 'execution_retry_count' and 't1_base_calls'.
    """
    config = config or {}
    target_path = state.get("workspace_absolute_path")
    target_branch = state.get("target_branch", "main")
    retry_count = state.get("execution_retry_count", 0)
    messages = state.get("messages", [])

    # Scan the most recent messages to see if the agent explicitly used the escape hatch
    explicit_escape = False
    for msg in reversed(messages):
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            if any(tc["name"] == "mark_task_already_completed" for tc in msg.tool_calls):
                explicit_escape = True
                # The agent correctly identified the task was already done.
                # Convert the intent to read_only to gracefully bypass PR creation!
                return {"latest_traceback_error": None, "intent_category": "workspace_read_only"}
            break  # Only evaluate the single most recent AI action

    raw_diff, incremental_diff = _get_evaluation_diffs(target_path, target_branch)
    if raw_diff is None:
        # If git fails locally, pass it through; review_pr_node has fallback handlers
        return {"latest_traceback_error": None}

    is_empty_diff = is_diff_empty(raw_diff)
    is_empty_incremental = is_diff_empty(incremental_diff)

    # Extract the true user request (ignoring our injected SYSTEM REJECTIONS)
    latest_human_msg = next(
        (
            m.content
            for m in reversed(state.get("messages", []))
            if m.type == "human"
            and not str(m.content).startswith("SYSTEM REJECTION")
            and not str(m.content).startswith("SYSTEM ERROR")
        ),
        "",
    )
    original_instruction = state.get("original_instruction", "")
    instruction_context = latest_human_msg if latest_human_msg else original_instruction

    # Extract the agent's final summary from the message history
    agent_summary = next(
        (str(m.content) for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        "No summary provided.",
    )

    critic_prompt = PromptManager.get(
        "evaluation",
        "critic_prompt",
        instruction_context=instruction_context,
        agent_summary=agent_summary,
        raw_diff=raw_diff if not is_empty_diff else "[NO CUMULATIVE CHANGES TO REPOSITORY]",
        incremental_diff=incremental_diff
        if not is_empty_incremental
        else "[NO NEW UNCOMMITTED CHANGES]",
    ).strip()

    try:
        # Use the base tier LLM for fast, reliable evaluation, wrapping it in fallbacks
        llms = get_execution_llm_sequence(requested_tier=TIER_BASE, temperature=0.0)
        eval_chain = _build_fallback_chain(llms)
        eval_result = eval_chain.invoke(critic_prompt, config=config).content.strip()
    except Exception:
        # Degrade gracefully if API fails and pass it through
        return {"latest_traceback_error": None}

    if eval_result.startswith("PASS"):
        return _handle_evaluation_pass(
            state, is_empty_diff, is_empty_incremental, explicit_escape, retry_count
        )

    return _handle_evaluation_fail(eval_result, state, retry_count)


def _get_evaluation_diffs(target_path: str, target_branch: str) -> tuple[str | None, str | None]:
    """
    Helper to stage intent-to-add files and fetch git diffs for evaluation.
    Ensures untracked files are captured so the LLM critic sees the complete scope of modifications.
    """
    try:
        try:
            # Stage intent-to-add for all untracked files so they appear in git diff
            # Set check=True and capture_output to force an exception on failure and
            # read the stderr
            subprocess.run(
                ["git", "add", "-N", "."],
                cwd=os.path.expanduser(target_path),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(
                "[Warning] 'git add -N .' failed in _get_evaluation_diffs.\n"
                f"Exit Code: {e.returncode}\nError: {e.stderr}"
            )
        except Exception as e:
            print(f"[Error] Unexpected exception during git add: {e}")

        # The cumulative diff for the whole PR (against the target branch)
        raw_diff = get_git_diff(directory=target_path, target_branch=target_branch)
        # The incremental diff for the new uncommitted changes (against HEAD)
        incremental_diff = get_git_diff(directory=target_path)
        return raw_diff, incremental_diff

    except Exception as e:
        print(f"[Error] Failed to fetch git diffs in _get_evaluation_diffs: {e}")
        return None, None


def _handle_evaluation_pass(
    state: PRState,
    is_empty_diff: bool,
    is_empty_incremental: bool,
    explicit_escape: bool,
    retry_count: int,
) -> dict:
    """
    Helper to process a successful critic evaluation.
    Traps autonomous tool hallucinations by strictly validating diff changes
    against declared intents.
    """
    if is_empty_diff and is_empty_incremental:
        # Trap LLM Tool Hallucinations: If it was supposed to operate on the workspace
        # but didn't explicitly use the escape hatch, it likely hallucinated task completion.
        if not explicit_escape and state.get("intent_category") == "workspace_operation":
            if retry_count < settings.agent.max_sandbox_retries:
                msg = PromptManager.get("evaluation", "hallucinated_success_rejection").strip()
                return {
                    "messages": [HumanMessage(content=msg)],
                    "latest_traceback_error": "hallucinated_success",
                    "execution_retry_count": retry_count + 1,
                    "t1_base_calls": state.get("t1_base_calls", 0) + 1,
                }

            abort_msg = (
                "⚠️ **Execution Failed:** I was unable to invoke the necessary tools "
                f"to satisfy the requirements after {retry_count} attempts.\n\n"
                "*Workflow safely aborted.*"
            )
            return {
                "messages": [AIMessage(content=abort_msg)],
                "is_aborted": True,
                "latest_traceback_error": None,
                "execution_retry_count": retry_count,
            }

        # Valid read-only pass
        return {"latest_traceback_error": None, "intent_category": "workspace_read_only"}

    # Standard pass with valid changes
    return {"latest_traceback_error": None}


def _handle_evaluation_fail(eval_result: str, state: PRState, retry_count: int) -> dict:
    """
    Helper to process a failed critic evaluation.
    Triggers a retry loop by sending formatted semantic feedback to the primary
    agent, or aborts if max attempts are reached.
    """
    if retry_count >= settings.agent.max_sandbox_retries:
        abort_msg = (
            "⚠️ **Execution Failed:** I was unable to satisfy the requirements after "
            f"{retry_count} attempts.\n\n"
            f"*Critic Feedback:* {eval_result}\n\n*Workflow safely aborted.*"
        )
        return {
            "messages": [AIMessage(content=abort_msg)],
            "is_aborted": True,
            "latest_traceback_error": None,
            "execution_retry_count": retry_count,
        }

    feedback = eval_result.replace("FAIL:", "").strip()
    msg = PromptManager.get("evaluation", "semantic_rejection", feedback=feedback).strip()

    return {
        "messages": [HumanMessage(content=msg)],
        "latest_traceback_error": "semantic_review_rejection",
        "execution_retry_count": retry_count + 1,
        "t1_base_calls": state.get("t1_base_calls", 0) + 1,
    }


# ==========================================
# Agentic CI/CD Nodes
# ==========================================


def agentic_ci_node(state: PRState, config: RunnableConfig = None) -> dict:
    """
    Agentic CI/CD Execution Node.
    Iterates through configured CI suites, executes them via OS-level subprocess isolation,
    and natively reports status checks and log aggregations back to the GitHub Pull Request.

    State Transitions:
    - Always updates 'ci_results' array containing suite execution dicts and raw logs.
    """
    target_path = state.get("workspace_absolute_path")
    repo_full_name = state.get("repo_full_name")
    commit_sha = state.get("commit_sha")
    pr_number = state.get("pr_number")

    if not target_path or not pr_number:
        return {}

    workspace_config = next(
        (ws for ws in settings.workspaces.values() if ws.path == target_path), None
    )

    if not workspace_config or not workspace_config.ci_suites:
        return {"ci_results": []}

    run_id = str(uuid.uuid4())

    if not _acquire_preemptive_lock(target_path, run_id, pr_number, commit_sha, repo_full_name):
        return {"ci_results": []}

    # Lock acquired (or bypassed via fail-open), proceed with execution
    expanded_target_path = os.path.expanduser(target_path)
    tmp_dir = None

    try:
        # 0. Generate unique ephemeral directory
        tmp_dir = tempfile.mkdtemp(prefix=f"ci_job_{run_id}_")
        print(f"Agentic CI: Creating ephemeral workspace at {tmp_dir}")

        # 1. Isolate: Clone the host repository to the /tmp directory locally (extremely fast)
        subprocess.run(
            ["git", "clone", expanded_target_path, tmp_dir], check=True, capture_output=True
        )

        # 2. Checkout exact PR commit from origin
        # We inject the credential helper just in case the remote requires auth
        github_token = os.getenv("GITHUB_TOKEN", "")
        cred_helper = (
            f'!f() {{ echo "username=x-access-token"; echo "password={github_token}"; }}; f'
        )

        # Explicitly build the GitHub URL so we don't fetch from the local 'origin'
        github_url = f"https://github.com/{repo_full_name}.git"

        subprocess.run(
            [
                "git",
                "-c",
                f"credential.helper={cred_helper}",
                "fetch",
                github_url,
                f"pull/{pr_number}/head",
            ],
            cwd=tmp_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", "FETCH_HEAD"], cwd=tmp_dir, check=True, capture_output=True
        )

        active_commit_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_dir)
            .decode("utf-8")
            .strip()
        )

        # 3. Dependency Caching: Use uv to build an ephemeral .venv lightning fast
        print(f"Agentic CI: Building isolated .venv using uv in {tmp_dir}...")
        subprocess.run(["uv", "venv"], cwd=tmp_dir, check=True, capture_output=True)

        req_file = None
        if os.path.exists(os.path.join(tmp_dir, "requirements-dev.txt")):
            req_file = "requirements-dev.txt"
        elif os.path.exists(os.path.join(tmp_dir, "requirements.txt")):
            req_file = "requirements.txt"

        if req_file:
            env = os.environ.copy()
            env["VIRTUAL_ENV"] = os.path.join(tmp_dir, ".venv")
            env["PATH"] = f"{os.path.join(tmp_dir, '.venv', 'bin')}:{env.get('PATH', '')}"
            subprocess.run(
                ["uv", "pip", "install", "-r", req_file],
                cwd=tmp_dir,
                env=env,
                check=True,
                capture_output=True,
            )
            print(f"Agentic CI: Dependencies installed successfully from {req_file}.")

        # 4. Execute all suites in the isolated environment
        results = _execute_ci_suites(
            target_path=target_path,
            expanded_target_path=tmp_dir,
            active_commit_sha=active_commit_sha,
            ci_suites=workspace_config.ci_suites,
            repo_full_name=repo_full_name,
            run_id=run_id,
        )

        # 5. Post Consolidated PR Comment
        _post_ci_results_comment(target_path, pr_number, repo_full_name, results)

        return {"ci_results": results}

    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
        print(f"Agentic CI setup error: {err_msg}", flush=True)
        if commit_sha:
            set_commit_status(
                directory=target_path,
                commit_sha=commit_sha,
                state="error",
                context_str="Agentic CI / Setup",
                description="Failed to prepare isolated CI environment.",
                repo_full_name=repo_full_name,
            )
        return {"ci_results": []}

    except InterruptedError as e:
        # We were gracefully aborted by a newer commit, exit quietly.
        print(f"Agentic CI gracefully aborted: {e}", flush=True)
        return {"ci_results": []}

    finally:
        # 6. Release Concurrency Lock & GC the isolated directory
        _release_preemptive_lock_and_cleanup(target_path, run_id, tmp_dir=tmp_dir)


def _acquire_preemptive_lock(
    target_path: str,
    run_id: str,
    pr_number: int,
    commit_sha: str | None,
    repo_full_name: str | None,
) -> bool:
    """
    Attempts to acquire the CI lock using Redis to limit concurrent environment load.
    Preempts older stale runs or rejects execution outright based on maximum concurrency rules.
    """
    if not redis_client:
        return True

    workspace_lock_id = f"ci_job:{target_path}"
    pid_key = f"ci_pid:{target_path}"

    try:
        is_already_running = redis_client.sismember("active_ci_jobs", workspace_lock_id)
        active_jobs = redis_client.scard("active_ci_jobs")
        limit_reached = active_jobs >= settings.agent.max_concurrent_ci_jobs

        # Only reject if we hit the limit AND it's not our own repo we can preempt
        if limit_reached and not is_already_running:
            _handle_concurrency_rejection(
                target_path,
                pr_number,
                commit_sha,
                repo_full_name,
                False,
                active_jobs,
            )
            return False

        if is_already_running:
            _preempt_stale_ci_run(workspace_lock_id, pid_key)

        # Acquire Lock for this new run
        redis_client.sadd("active_ci_jobs", workspace_lock_id)
        redis_client.set(workspace_lock_id, run_id)
        return True
    except Exception as e:
        print(f"Warning: Redis concurrency check failed: {e}")
        # Fail-open if Redis crashes temporarily so workflow isn't fully blocked
        return True


def _handle_concurrency_rejection(
    target_path: str,
    pr_number: int,
    commit_sha: str | None,
    repo_full_name: str | None,
    is_already_running: bool,
    active_jobs: int,
) -> None:
    """
    Helper to post rejection messages to GitHub when CI limits are hit.
    Notifies users of skipped tests to preserve global system resources.
    """
    max_jobs = settings.agent.max_concurrent_ci_jobs
    reason = (
        "A CI/CD job is already running for this repository"
        if is_already_running
        else f"Global concurrency limit ({max_jobs}) reached"
    )
    print(f"Agentic CI/CD skipped: {reason}. ({active_jobs}/{max_jobs})")

    comment_body = (
        f"⚠️ **Agentic CI/CD Skipped**\n\n"
        f"{reason} to preserve system resources. "
        f"If necessary, please comment `/retest` to try again once the current tests finish."
    )

    comment_on_pull_request(
        directory=target_path,
        pr_number=pr_number,
        body=comment_body,
        repo_full_name=repo_full_name,
    )

    if commit_sha:
        set_commit_status(
            directory=target_path,
            commit_sha=commit_sha,
            state="error",
            context_str="Agentic CI/CD / Concurrency",
            description="Concurrency limit reached. If necessary, comment /retest later.",
            repo_full_name=repo_full_name,
        )


def _preempt_stale_ci_run(workspace_lock_id: str, pid_key: str) -> None:
    """
    Helper to terminate an older, superseded Agentic CI test process tree.
    Uses psutil to aggressively clear memory and processes for the newest PR commit.
    """
    if not redis_client:
        return

    old_run_id = redis_client.get(workspace_lock_id)
    if not old_run_id:
        return

    old_run_id = old_run_id.decode("utf-8") if isinstance(old_run_id, bytes) else old_run_id
    print(f"Agentic CI: Preempting older execution {old_run_id} for newest commit.")

    # Mark old run as superseded
    redis_client.setex(f"ci_superseded:{old_run_id}", 3600, "1")

    # Send SIGKILL to the old test suite process tree
    old_pid = redis_client.get(pid_key)
    if old_pid:
        try:
            parent = psutil.Process(int(old_pid))
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except (psutil.NoSuchProcess, TypeError, ValueError):
            pass

    # Brief pause to let the old LangGraph thread trap
    # the InterruptedError and exit cleanly
    time.sleep(1.0)


def _execute_ci_suites(
    target_path: str,
    expanded_target_path: str,
    active_commit_sha: str,
    ci_suites: list,
    repo_full_name: str | None,
    run_id: str,
) -> list[dict]:
    """
    Helper to isolate CI subprocess executions and coordinate GitHub commit statuses.
    Gracefully surfaces errors, OS timeouts, or intentional interrupt signals if a
    newer commit supersedes the process.
    """
    # 1. Post pending statuses for all suites upfront
    for suite in ci_suites:
        context_str = f"Agentic CI / {suite.name}"
        set_commit_status(
            directory=target_path,
            commit_sha=active_commit_sha,
            state="pending",
            context_str=context_str,
            description="Evaluation is running...",
            repo_full_name=repo_full_name,
        )

    results = []

    # Inject the uv .venv into the execution environment
    env = os.environ.copy()
    venv_path = os.path.join(expanded_target_path, ".venv")
    if os.path.exists(venv_path):
        env["VIRTUAL_ENV"] = venv_path
        env["PATH"] = f"{os.path.join(venv_path, 'bin')}:{env.get('PATH', '')}"

    # 2. Execute each suite strictly sequentially
    for suite in ci_suites:
        # Check if a newer commit preempted us before we even start this suite
        if redis_client and redis_client.get(f"ci_superseded:{run_id}"):
            raise InterruptedError("Superseded by a newer commit.")

        context_str = f"Agentic CI / {suite.name}"
        try:
            process = subprocess.Popen(
                suite.command,
                cwd=expanded_target_path,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )

            # Store the PID so a newer commit can preemptively kill this process
            if redis_client:
                redis_client.set(f"ci_pid:{target_path}", process.pid)

            try:
                stdout, stderr = process.communicate(timeout=suite.timeout_seconds)
                passed = process.returncode == 0
                logs = f"{stdout}\n{stderr}".strip()
            except subprocess.TimeoutExpired:
                # Force kill the process tree instantly using psutil
                try:
                    parent = psutil.Process(process.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass

                # Grab whatever output was generated before the timeout kill
                stdout, stderr = process.communicate()
                logs = (
                    f"{stdout}\n{stderr}\n\n"
                    f"[ERROR: TimeoutExpired] Command forcefully terminated after "
                    f"{suite.timeout_seconds} seconds."
                ).strip()
                passed = False
            finally:
                # If we were killed by a newer commit, psutil causes communicate()
                # to return instantly. We trap that here and abort cleanly before
                # posting false test results to GitHub.
                if redis_client and redis_client.get(f"ci_superseded:{run_id}"):
                    raise InterruptedError("Superseded by a newer commit.")

            set_commit_status(
                directory=target_path,
                commit_sha=active_commit_sha,
                state="success" if passed else "failure",
                context_str=context_str,
                description="Evaluation passed"
                if passed
                else "Execution timed out"
                if not passed and "[ERROR: TimeoutExpired]" in logs
                else "Evaluation failed",
                repo_full_name=repo_full_name,
            )

            results.append({"name": suite.name, "passed": passed, "logs": logs})

        except InterruptedError:
            # Bubble up the intentional abort signal
            raise
        except Exception as e:
            set_commit_status(
                directory=target_path,
                commit_sha=active_commit_sha,
                state="error",
                context_str=context_str,
                description="OS error during execution",
                repo_full_name=repo_full_name,
            )
            results.append({"name": suite.name, "passed": False, "logs": f"OS Error: {str(e)}"})

    return results


def _post_ci_results_comment(
    target_path: str, pr_number: int, repo_full_name: str | None, results: list[dict]
) -> None:
    """
    Helper to format and post the consolidated CI results to GitHub.
    Automatically truncates massive test logs to prevent hitting GitHub API limits.
    """
    if not results:
        return

    summary_table = "| Suite | Status |\n|---|---|\n"
    for r in results:
        status_icon = "✅ Pass" if r["passed"] else "❌ Fail"
        summary_table += f"| {r['name']} | {status_icon} |\n"

    details_sections = ""
    for r in results:
        safe_logs = (
            r["logs"][:MAX_GITHUB_COMMENT_LENGTH] + "\n...[TRUNCATED]"
            if len(r["logs"]) > MAX_GITHUB_COMMENT_LENGTH
            else r["logs"]
        )

        details_sections += (
            f"\n<details><summary>Logs: {r['name']}</summary>\n\n"
            f"```text\n{safe_logs}\n```\n</details>\n"
        )

    raw_prefix = settings.agent.agent_prefix or ""
    prefix_str = f"{raw_prefix} " if raw_prefix and not raw_prefix.endswith(" ") else raw_prefix

    comment_body = f"## {prefix_str}Agentic CI/CD Results\n\n{summary_table}\n{details_sections}"

    comment_on_pull_request(
        directory=target_path,
        pr_number=pr_number,
        body=comment_body,
        repo_full_name=repo_full_name,
    )


def _release_preemptive_lock_and_cleanup(
    target_path: str,
    run_id: str,
    tmp_dir: str | None = None,
) -> None:
    """
    Releases the Redis concurrency lock if we still own it, and safely garbage collects the tmp dir.
    Guarantees cleanup even if the process was preempted to prevent local filesystem exhaustion.
    """
    workspace_lock_id = f"ci_job:{target_path}"
    pid_key = f"ci_pid:{target_path}"

    if redis_client:
        try:
            current_owner = redis_client.get(workspace_lock_id)
            if current_owner:
                current_owner = (
                    current_owner.decode("utf-8")
                    if isinstance(current_owner, bytes)
                    else current_owner
                )

                if current_owner == run_id:
                    # We still own the lock, clean it up
                    redis_client.srem("active_ci_jobs", workspace_lock_id)
                    redis_client.delete(workspace_lock_id)
                    redis_client.delete(pid_key)
                # If a newer commit preempted us, we just leave the lock alone.
        except Exception as e:
            print(f"Warning: Failed to verify Redis lock ownership: {e}")

    # 5. Cleanup: Delete the ephemeral directory.
    # We do this regardless of lock ownership because every run gets a unique tmp_dir!
    if tmp_dir and os.path.exists(tmp_dir):
        try:
            shutil.rmtree(tmp_dir)
            print(f"Agentic CI: Ephemeral workspace {tmp_dir} cleaned up successfully.")
        except Exception as e:
            print(f"Warning: Failed to GC temporary CI directory {tmp_dir}: {e}")


# ==========================================
# Pull Request Lifecycle Nodes
# ==========================================


def review_pr_node(state: PRState, config: RunnableConfig = None) -> dict:
    """
    Branch-and-Link execution.
    Branches the code, commits, pushes, and suspends for human review.

    State Transitions:
    - On success: Persists 'active_agent_branch', populates 'pending_pr_url',
      and updates human-facing 'messages'.
    - On failure: Flags 'is_aborted' to true or loops by updating
      'latest_traceback_error' and 'execution_retry_count'.
    """
    config = config or {}
    target_path = state.get("workspace_absolute_path")
    active_branch = state.get("active_agent_branch")
    pending_pr_url = state.get("pending_pr_url")

    # Determine if we are updating an existing PR or making a new one
    is_refining = bool(active_branch and pending_pr_url)
    repo_name = target_path.split("/")[-1]
    target_branch = state.get("target_branch", "main")

    # Extract the absolute latest user feedback, explicitly ignoring system traps
    latest_human_msg = next(
        (
            m.content
            for m in reversed(state.get("messages", []))
            if m.type == "human"
            and not str(m.content).startswith("SYSTEM REJECTION")
            and not str(m.content).startswith("SYSTEM ERROR")
        ),
        "",
    )
    original_instruction = state.get("original_instruction", "")

    # Build a holistic instruction context so the LLM doesn't lose the forest for the trees
    if is_refining and latest_human_msg and latest_human_msg != original_instruction:
        instruction_context = (
            f"Original PR Goal: {original_instruction}\n"
            f"Latest Revision Requested: {latest_human_msg}"
        )
    else:
        instruction_context = latest_human_msg if latest_human_msg else original_instruction

    # 1. Grab the raw diffs and blueprints
    try:
        # The cumulative diff for the whole PR (against the target branch)
        raw_diff = get_git_diff(directory=target_path, target_branch=target_branch)
        blueprint = get_git_diff_blueprint(directory=target_path, target_branch=target_branch)

        # The incremental diff for the new commit (against HEAD)
        incremental_diff = get_git_diff(directory=target_path)
        incremental_blueprint = get_git_diff_blueprint(directory=target_path)
    except Exception:
        raw_diff = ""
        blueprint = "Unknown Blueprint"
        incremental_diff = ""
        incremental_blueprint = "Unknown Blueprint"

    # 2. Generate the dynamic commit message based strictly on the uncommitted incremental changes
    commit_message = _generate_commit_message(
        latest_human_msg, incremental_diff, incremental_blueprint, config
    )

    # 3. Generate branch and PR metadata using the Map-Reduce pipeline
    if is_refining:
        branch_name = active_branch
        _, generated_body, generated_summary = _generate_pr_metadata(
            instruction_context, raw_diff, blueprint, config
        )
    else:
        branch_name, generated_body, generated_summary = _generate_pr_metadata(
            instruction_context, raw_diff, blueprint, config
        )

    # 4. Branch and Commit
    commit_response = json.loads(
        create_branch_and_commit(
            directory=target_path, new_branch=branch_name, commit_message=commit_message
        )
    )

    if commit_response.get("status") == "error":
        return _handle_commit_failure(
            commit_response, state.get("execution_retry_count", 0), branch_name
        )

    raw_prefix = settings.agent.agent_prefix or ""
    prefix_str = f"{raw_prefix} " if raw_prefix and not raw_prefix.endswith(" ") else raw_prefix

    # 5. Handle existing PR push success and dynamically update Title/Body
    if is_refining:
        pr_title = f"{prefix_str}{generated_summary}"
        update_status_msg = _attempt_pr_update(
            target_path, pending_pr_url, pr_title, generated_body, generated_summary
        )

        msg = (
            "✅ **Revisions Applied**\n"
            f"📦 `[{repo_name}]`\n\n"
            "I have pushed the requested updates to the existing Pull Request:\n"
            f"🔗 {pending_pr_url}\n{update_status_msg}\n\n"
            "*Merge the PR on GitHub to finalize, or provide feedback here for further edits.*"
        )
        return {"messages": [AIMessage(content=msg)]}

    # 6. Open a new PR
    pr_title = f"{prefix_str}{generated_summary}"
    pr_response = open_pull_request(
        directory=target_path,
        title=pr_title,
        head_branch=branch_name,
        base_branch=state.get("target_branch", "main"),
        body=generated_body,
    )

    try:
        pr_data = json.loads(pr_response)
        if pr_data.get("status") == "error":
            return {
                "active_agent_branch": branch_name,
                "messages": [
                    AIMessage(
                        content=(
                            "⚠️ **PR Creation Failed:** The code was pushed to GitHub, but "
                            "the Pull Request API rejected the request.\n"
                            f"**Reason:** {pr_data.get('reason')}\n"
                            f"**Details:** `{pr_data.get('details', '')}`\n\n"
                            "*Aborting and cleaning up temporary branches.*"
                        )
                    )
                ],
                "is_aborted": True,
            }

        pr_url = pr_data.get("pr_url", "[Error Fetching URL]")
    except json.JSONDecodeError:
        pr_url = "[Error Fetching URL]"

    msg = (
        "✅ **Execution Complete**\n"
        f"📦 `[{repo_name}]`\n\n"
        "I have pushed the changes to a new branch and opened a Pull Request for your review:\n"
        f"🔗 {pr_url}\n\n"
        "*Merge the PR on GitHub to finalize, or provide feedback here for further edits.*"
    )

    return {
        "active_agent_branch": branch_name,
        "pending_pr_url": pr_url,
        "messages": [AIMessage(content=msg)],
    }


def _generate_commit_message(
    instruction_context: str, incremental_diff: str, blueprint: str, config: RunnableConfig = None
) -> str:
    """
    Helper to generate a semantic commit message using the LLM based on the actual diff.
    Ensures commits have contextually accurate summaries regardless of original intention.
    """
    raw_prefix = settings.agent.agent_prefix or ""
    prefix_str = f"{raw_prefix} " if raw_prefix and not raw_prefix.endswith(" ") else raw_prefix

    if not instruction_context and not incremental_diff:
        return f"{prefix_str}Apply human feedback revisions"

    config = config or {}
    try:
        llms = get_execution_llm_sequence(requested_tier=TIER_BASE, temperature=0.2)
        eval_chain = _build_fallback_chain(llms)

        # Fast track for huge diffs: just use the blueprint and the first chunk
        chunks = _chunk_git_diff(incremental_diff, MAX_DIFF_LENGTH)
        diff_snippet = chunks[0] if chunks else ""

        diff_str = f"\nACTUAL CODE DIFF SNIPPET:\n{diff_snippet}\n" if diff_snippet else ""

        prompt = PromptManager.get(
            "pr_generation",
            "commit_message",
            instruction_context=instruction_context,
            blueprint=blueprint,
            diff_snippet=diff_str,
        ).strip()

        res = eval_chain.invoke(prompt, config=config).content.strip().strip("\"'")
        if res:
            return f"{prefix_str}{res}"
    except Exception as e:
        print(f"Error generating commit message: {e}")

    return f"{prefix_str}Apply human feedback revisions"


def _generate_pr_metadata(
    instruction_context: str, raw_diff: str, blueprint: str, config: RunnableConfig = None
) -> tuple[str, str, str]:
    """
    Helper to generate semantic branch names and PR descriptions using Map-Reduce.
    Combines deep file-by-file context mapping to bypass LLM token limits on large modifications.
    """
    config = config or {}
    try:
        llms = get_execution_llm_sequence(requested_tier=TIER_BASE, temperature=0.2)
        eval_chain = _build_fallback_chain(llms)
        chunks = _chunk_git_diff(raw_diff, MAX_DIFF_LENGTH)

        # Map phase: Anchor each chunk with the Blueprint and parse file-by-file
        if len(chunks) > 1:
            chunk_summaries = []
            for i, chunk_text in enumerate(chunks):
                map_prompt = PromptManager.get(
                    "pr_generation",
                    "map_phase",
                    instruction_context=instruction_context,
                    blueprint=blueprint,
                    part_num=i + 1,
                    total_parts=len(chunks),
                    chunk_text=chunk_text,
                ).strip()

                res = eval_chain.invoke(map_prompt, config=config).content.strip()
                chunk_summaries.append(res)

            synthesized_diff = "\n\n".join(
                [f"--- Part {i + 1} Summary ---\n{s}" for i, s in enumerate(chunk_summaries)]
            )
        else:
            synthesized_diff = chunks[0] if chunks else ""

        syn_diff_str = (
            f"\nACTUAL SYNTHESIZED DIFF SUMMARY:\n{synthesized_diff}\n" if synthesized_diff else ""
        )

        summary_prompt = PromptManager.get(
            "pr_generation",
            "summary_phase",
            instruction_context=instruction_context,
            blueprint=blueprint,
            synthesized_diff=syn_diff_str,
        ).strip()

        generated_summary = (
            eval_chain.invoke(summary_prompt, config=config).content.strip().strip("\"'")
        )

        body_prompt = PromptManager.get(
            "pr_generation",
            "body_phase",
            instruction_context=instruction_context,
            blueprint=blueprint,
            synthesized_diff=syn_diff_str,
        ).strip()

        generated_body = eval_chain.invoke(body_prompt, config=config).content.strip()
    except Exception as e:
        print(f"Error in PR generation map-reduce: {e}")
        generated_summary = "Automated agent modifications"
        generated_body = "Automated PR generated by the LangGraph Agent."

    clean_slug = re.sub(r"[^a-zA-Z0-9\s]", "", generated_summary).strip()
    clean_slug = re.sub(r"\s+", "-", clean_slug).lower()
    short_hash = str(uuid.uuid4())[:4]

    branch_name = f"agent/{clean_slug}-{short_hash}"

    return branch_name, generated_body, generated_summary


def _chunk_git_diff(raw_diff: str, max_chunk_length: int = MAX_DIFF_LENGTH) -> list[str]:
    """
    Splits a raw git diff into manageable chunks safely along file boundaries.
    Truncates extremely large single-file diffs to prevent downstream prompt injection failures.
    """
    if is_diff_empty(raw_diff):
        return []

    parts = raw_diff.split("diff --git ")
    chunks = []
    current_chunk = ""

    for part in parts:
        if not part.strip():
            continue
        file_diff = "diff --git " + part

        # If a single file diff is too massive, truncate just that specific file
        if len(file_diff) > max_chunk_length:
            half = max_chunk_length // 2
            file_diff = (
                file_diff[:half] + "\n\n...[single file diff truncated]...\n\n" + file_diff[-half:]
            )

        # If adding this file pushes the current chunk over the limit, flush the chunk
        if current_chunk and len(current_chunk) + len(file_diff) > max_chunk_length:
            chunks.append(current_chunk.strip())
            current_chunk = file_diff
        else:
            current_chunk += "\n" + file_diff if current_chunk else file_diff

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def _handle_commit_failure(commit_response: dict, retry_count: int, branch_name: str) -> dict:
    """
    Helper to handle self-healing loops or aborts on commit failures.
    Differentiates 'no changes' failures from terminal OS-level git errors for
    accurate LLM feedback.
    """
    if commit_response.get("reason") == "no_changes_to_commit":
        if retry_count >= settings.agent.max_sandbox_retries:
            return {
                "messages": [
                    AIMessage(
                        content="⚠️ **Execution Failed:** I was unable to successfully "
                        "modify any files after multiple attempts. *Workflow safely aborted.*"
                    )
                ],
                "is_aborted": True,
            }
        return {
            "messages": [
                HumanMessage(
                    content=PromptManager.get("pr_generation", "no_changes_rejection").strip()
                )
            ],
            "latest_traceback_error": "no_changes_to_commit",
            "execution_retry_count": retry_count + 1,
        }

    return {
        "messages": [
            AIMessage(
                content=(
                    f"⚠️ **Commit & Push Failed:** Could not push updates to `{branch_name}`.\n"
                    f"**Reason:** {commit_response.get('reason')}\n"
                    f"**Details:** `{commit_response.get('details', 'No details provided')}`"
                )
            )
        ],
        "is_aborted": True,
    }


def _attempt_pr_update(
    directory: str,
    pending_pr_url: str,
    pr_title: str,
    generated_body: str,
    generated_summary: str,
) -> str:
    """
    Attempts to patch an existing GitHub PR.
    Silently fails and returns an empty string to gracefully degrade if URL parsing or APIs crash.
    """
    if not directory or not pending_pr_url:
        return ""

    try:
        pr_number = int(pending_pr_url.rstrip("/").split("/")[-1])
        update_res = json.loads(
            update_pull_request(
                directory=directory,
                pr_number=pr_number,
                title=pr_title,
                body=generated_body,
            )
        )
        if update_res.get("status") == "success":
            return (
                "\n*I also updated the PR title to reflect the "
                f"latest direction: '{generated_summary}'*"
            )
    except Exception:
        # Gracefully degrade if URL parsing or API call fails
        pass

    return ""


def pr_merged_node(state: AgentState) -> dict:
    """
    Finalization step after human review.
    Cleans up the active branch locally and resets the global Git state.

    State Transitions:
    - Overrides the graph state by clearing PR contexts ('human_approved',
      'pending_pr_url', 'active_agent_branch') to null.
    """
    target_path = state.get("workspace_absolute_path")
    target_branch = state.get("target_branch", "main")
    agent_branch = state.get("active_agent_branch")

    repo_name = target_path.split("/")[-1] if target_path else "Unknown Repo"

    if target_path and target_branch and agent_branch:
        cleanup_local_branch(target_path, target_branch, agent_branch)

    msg = (
        "🎉 **Pull Request Merged!**\n"
        f"📦 `[{repo_name}]`\n\n"
        f"🧹 *Cleanup Complete:* I have switched your local repository back to `{target_branch}` "
        f"and deleted the temporary `{agent_branch}` branch."
    )

    return {
        "messages": [AIMessage(content=msg)],
        "human_approved": False,
        "pending_pr_url": None,
        "active_agent_branch": None,
        "modified_tex_files": [],
    }

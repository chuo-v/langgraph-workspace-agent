import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import datetime

import git
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEPLOYMENT_ENV_PATH = os.path.join(
    os.path.dirname(REPO_ROOT), "langgraph-workspace-agent-deployment", ".env"
)

# Load the environment variables from the specific deployment path
if os.path.exists(DEPLOYMENT_ENV_PATH):
    load_dotenv(dotenv_path=DEPLOYMENT_ENV_PATH)
else:
    load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402
from langfuse import Langfuse, get_client, propagate_attributes  # noqa: E402
from langfuse.langchain import CallbackHandler  # noqa: E402

from src.workspace_agent.integrations.github_webhook import parse_github_pr_action  # noqa: E402
from src.workspace_agent.orchestrator.graph import agent_app  # noqa: E402
from tests.evals.mock_workspace import MockWorkspaceTracker  # noqa: E402
from tests.utilities.assertions import (  # noqa: E402
    assert_interrupts,
    validate_state,
    validate_tool_calls,
)

# If executing locally on a host machine terminal (not inside a Docker container
# and not in an automated CI pipeline), remap internal Docker service endpoints
# to localhost loopback ports.
IS_INSIDE_CONTAINER = os.path.exists("/.dockerenv")
if not os.getenv("CI") and not IS_INSIDE_CONTAINER:
    for key, value in os.environ.items():
        if isinstance(value, str):
            if "host.docker.internal" in value:
                os.environ[key] = value.replace("host.docker.internal", "127.0.0.1")
            if "redis" in value and "://" in value:
                os.environ[key] = value.replace("redis:6379", "127.0.0.1:6379").replace(
                    "redis://redis", "redis://127.0.0.1"
                )
            if "ollama:11434" in value:
                os.environ[key] = value.replace("ollama:11434", "127.0.0.1:11434")

DATASET_NAME = "workspace_agent_eval_e2e_pipeline"
DATASET_PATH = os.path.join(CURRENT_DIR, "datasets", "02_e2e_pipeline.json")

# Target threshold for E2E tests should remain high for core mechanics
TARGET_THRESHOLD = 90.0


def init_langfuse():
    """Initializes the Langfuse client pointing to the local instance."""
    # Force the local host into the active environment space BEFORE getting the client.
    # This guarantees all singleton clients and handlers route to localhost correctly.
    env_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    if "langfuse-" in env_host or "web" in env_host or "server" in env_host:
        os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

    # Use get_client() to tap into the global OTEL singleton
    return get_client()


def sync_dataset(lf: Langfuse):
    """Upserts the local JSON E2E test cases into the Langfuse database."""
    print(f"Synchronizing E2E dataset '{DATASET_NAME}' to Langfuse...")

    with open(DATASET_PATH) as f:
        cases = json.load(f)

    lf.create_dataset(
        name=DATASET_NAME, description="End-to-End Pull Request Refinement and Webhook Loops."
    )

    for case in cases:
        # Store the multi-turn script in metadata so the runner can pull it down
        lf.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=case["case_id"],
            expected_output={"turns_count": len(case["turns"])},
            metadata=case,
        )
    print(f"✅ Successfully synced {len(cases)} E2E cases.")


def _execute_turn(turn: dict, config: dict, run_config: dict) -> None:
    """Helper to execute a single turn of the graph based on input type."""
    input_type = turn["input_type"]
    current_state = agent_app.get_state(config)

    if input_type == "human_message":
        if turn.get("step") == 1:
            # Start Fresh: Pass the full payload to ensure default flags are clean for assertions
            payload = {
                "messages": [HumanMessage(content=turn["input"])],
                "original_instruction": turn["input"],
                "t1_base_calls": 0,
                "t2_standard_calls": 0,
                "t3_frontier_calls": 0,
                "execution_retry_count": 0,
                "latest_traceback_error": None,
                "clarification_question": None,
                "disambiguation_options": None,
                "is_aborted": False,
                "human_approved": False,
                "active_agent_branch": None,
                "pending_pr_url": None,
                "modified_tex_files": [],
            }
            agent_app.invoke(payload, config=run_config)
        else:
            # Resumption: Safely apply the state to the exact node that paused the graph
            current_state = agent_app.get_state(config)
            payload = {"messages": [HumanMessage(content=turn["input"])]}
            pending_node = current_state.next[0] if current_state.next else "human_pr_node"
            agent_app.update_state(config, payload, as_node=pending_node)
            agent_app.invoke(None, config=run_config)

    elif input_type == "webhook_payload":
        # Simulate the webhook arrival by parsing the payload
        action = parse_github_pr_action(turn["input"])

        state_update = {}
        if action == "LGTM":
            state_update["human_approved"] = True
        elif action == "abort":
            state_update["is_aborted"] = True

        current_state = agent_app.get_state(config)

        # Apply webhook mutations strictly to the paused node so edges route to cleanup/merged nodes
        pending_node = current_state.next[0] if current_state.next else "human_pr_node"
        agent_app.update_state(config, state_update, as_node=pending_node)

        # Resume graph execution
        agent_app.invoke(None, config=run_config)


def _load_fixtures(fixtures: list, tracker, current_dir: str) -> None:
    """Helper to dynamically load and commit file fixtures into the mock workspace."""
    for fixture in fixtures:
        source_rel = fixture["source"]
        target_ws = fixture["target_workspace"]
        target_rel = fixture["target_path"]

        source_abs = os.path.join(current_dir, "fixtures", source_rel)
        target_base = tracker.sandbox_paths.get(target_ws)

        if target_base and os.path.exists(source_abs):
            target_abs = os.path.join(target_base, target_rel)
            os.makedirs(os.path.dirname(target_abs), exist_ok=True)
            shutil.copy2(source_abs, target_abs)

            # Stage and commit the fixture so git diffs work properly during testing
            try:
                repo = git.Repo(target_base)
                repo.git.add(A=True)
                repo.git.commit("-m", f"Agent Eval: Loaded fixture {target_rel}")
            except Exception as e:
                print(f"Warning: Failed to commit fixture {target_rel}: {e}")


def _evaluate_e2e_case(item, lf: Langfuse, session_name: str) -> tuple[int, int]:
    """Executes a full multi-turn script under a single thread_id and Langfuse Trace."""
    case_data = item.metadata
    case_id = case_data["case_id"]
    turns = case_data["turns"]
    fixtures = case_data.get("fixtures", [])

    print(f"\nEvaluating Case: '{case_id}' ({len(turns)} turns)")

    # 1. Initialize a unique thread ID for this specific E2E test case
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    total_score = 0
    max_score = 0

    # 2. Start a Root Trace Span for the entire multi-turn thread
    try:
        with lf.start_as_current_observation(as_type="span", name=f"E2E_{case_id}") as span:
            span.update(input={"thread_id": thread_id, "description": case_data.get("description")})
            trace_id = lf.get_current_trace_id()

            with propagate_attributes(session_id=session_name):
                # Wrap the entire execution in the mock sandbox
                with MockWorkspaceTracker() as tracker:
                    if fixtures:
                        _load_fixtures(fixtures, tracker, CURRENT_DIR)

                    for turn in turns:
                        step_num = turn["step"]
                        input_type = turn["input_type"]
                        print(f"  ▶ Turn {step_num} ({input_type})")

                        # The CallbackHandler natively inherits the OpenTelemetry trace context
                        handler = CallbackHandler()
                        run_config = {**config, "callbacks": [handler]}

                        # Execute Graph (Safe because tools are patched)
                        _execute_turn(turn, config, run_config)

                        current_state = agent_app.get_state(config)

                        interrupt_passed = assert_interrupts(
                            turn.get("expected_interrupt"), current_state.next
                        )

                        expected_state = turn.get("expected_state", {})
                        passed, total = validate_state(expected_state, current_state.values)

                        # Check if the dataset specifies tools that must have been called this turn
                        expected_tools = turn.get("expected_tool_calls", [])
                        tools_passed = validate_tool_calls(expected_tools, tracker)

                        # Update Scoring Logic
                        turn_score = passed + (1 if interrupt_passed else 0) + tools_passed
                        turn_max = total + 1 + len(expected_tools)

                        total_score += turn_score
                        max_score += turn_max

                        lf.create_score(
                            trace_id=trace_id,
                            name=f"turn_{step_num}_accuracy",
                            value=turn_score / turn_max if turn_max > 0 else 1.0,
                        )

            try:
                if hasattr(lf, "api") and hasattr(lf.api, "dataset_run_items"):
                    lf.api.dataset_run_items.create(
                        dataset_item_id=item.id, run_name=session_name, trace_id=trace_id
                    )
            except Exception:
                pass

    except Exception as e:
        # OpenTelemetry automatically catches exceptions and marks the span with an error status
        print(f"  ❌ FATAL ERROR during case execution: {e}")
        return total_score, max_score

    return total_score, max_score


def run_evaluations(lf: Langfuse):
    """Executes the full compiled Agent Application for E2E workflows."""
    print("Initializing LangGraph E2E Evaluations...\n")

    dataset = lf.get_dataset(DATASET_NAME)
    print(f"Starting evaluations for {len(dataset.items)} E2E cases...\n")

    total_score = 0
    max_score = 0
    session_name = f"E2E_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for item in dataset.items:
        score, mx = _evaluate_e2e_case(item, lf, session_name)
        total_score += score
        max_score += mx

    # Flush the global OTEL singleton. This safely pushes all traces, scores, and links at once!
    lf.flush()

    accuracy = (total_score / max_score) * 100 if max_score > 0 else 0

    print(f"\nE2E Evaluation Complete. Final Score: {total_score}/{max_score} ({accuracy:.1f}%)")
    print("Check Langfuse (http://localhost:3000) for complete multi-turn trace sessions.")

    if accuracy < TARGET_THRESHOLD:
        print(f"\n❌ System Health Warning: E2E accuracy dropped below {TARGET_THRESHOLD}%.")
        sys.exit(1)

    print(f"\n🚀 System Health Excellent: E2E workflows meet the {TARGET_THRESHOLD}% threshold!")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run End-to-End Agentic Evaluations")
    parser.add_argument("--sync", action="store_true", help="Sync local E2E dataset to Langfuse")
    parser.add_argument("--eval", action="store_true", help="Run E2E evaluations against dataset")
    args = parser.parse_args()

    lf_client = init_langfuse()

    if args.sync:
        sync_dataset(lf_client)

    if args.eval:
        run_evaluations(lf_client)

    if not args.sync and not args.eval:
        parser.print_help()

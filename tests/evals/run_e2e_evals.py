import argparse
import json
import os
import sys
import uuid
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langfuse import Langfuse, get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from src.workspace_agent.integrations.github_webhook import parse_github_pr_action

# --- E2E Integration Imports ---
from src.workspace_agent.orchestrator.graph import agent_app
from tests.evals.mock_workspace import MockWorkspaceTracker

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


def _validate_state(expected_state: dict, actual_state_values: dict) -> int:
    """Helper to assert the graph's internal state matches expectations."""
    passed_criteria = 0
    total_criteria = len(expected_state)

    for key, expected_val in expected_state.items():
        actual_val = actual_state_values.get(key)

        if expected_val == "__NOT_NULL__":
            if actual_val is not None:
                passed_criteria += 1
                print(f"      ✅ {key}: (Populated)")
            else:
                print(f"      ❌ {key}: Expected populated value, got None")
        elif actual_val == expected_val:
            passed_criteria += 1
            print(f"      ✅ {key}: {actual_val}")
        else:
            print(f"      ❌ {key}: Expected {expected_val}, got {actual_val}")

    return passed_criteria, total_criteria


def _validate_tool_calls(expected_tools: list, tracker, config: dict) -> int:
    """Helper to validate both mock and native tool calls across the entire graph history."""
    if not expected_tools:
        return 0

    tools_passed = 0
    # 1. Get remote/mocked tool calls from the sandbox boundary
    called_tools = [call["tool"] for call in tracker.invocation_history]

    # 2. Extract native LangChain tool calls from the ENTIRE state history.
    # This is critical because memory management nodes often prune intermediate
    # ToolMessages from the active state to save context window space.
    for snapshot in agent_app.get_state_history(config):
        for msg in snapshot.values.get("messages", []):
            # Robust extraction to handle both instantiated BaseMessages and raw serialized dicts
            is_dict = isinstance(msg, dict)
            msg_type = msg.get("type") if is_dict else getattr(msg, "type", None)
            msg_name = msg.get("name") if is_dict else getattr(msg, "name", None)
            tool_calls = msg.get("tool_calls", []) if is_dict else getattr(msg, "tool_calls", [])

            # Check AIMessage tool_calls
            if tool_calls:
                for tc in tool_calls:
                    tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if tc_name and tc_name not in called_tools:
                        called_tools.append(tc_name)

            # Check ToolMessages directly
            if msg_type == "tool" and msg_name:
                if msg_name not in called_tools:
                    called_tools.append(msg_name)

    # Validate against the aggregated tool list
    for tool in expected_tools:
        if tool in called_tools:
            tools_passed += 1
            print(f"      ✅ Tool execution verified: {tool}")
        else:
            print(f"      ❌ Missing expected tool call: {tool}")

    # Clear mock history after asserting to prepare for the next turn
    tracker.invocation_history.clear()

    return tools_passed


def _assert_interrupts(expected_interrupt: str | None, actual_next_nodes: tuple) -> bool:
    """Helper to assert if the graph paused at the expected node."""
    if expected_interrupt:
        if expected_interrupt in actual_next_nodes:
            print(f"      ✅ Graph Paused at: {expected_interrupt}")
            return True

        # Split across lines to fix E501
        print(
            f"      ❌ Graph failed to pause at {expected_interrupt}. "
            f"Currently at: {actual_next_nodes}"
        )
        return False

    if not actual_next_nodes:
        print("      ✅ Graph completed/exited successfully.")
        return True

    print(f"      ❌ Graph unexpectedly paused at: {actual_next_nodes}")
    return False


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


def _evaluate_e2e_case(item, lf: Langfuse, session_name: str) -> tuple[int, int]:
    """Executes a full multi-turn script under a single thread_id and Langfuse Trace."""
    case_data = item.metadata
    case_id = case_data["case_id"]
    turns = case_data["turns"]

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

                        interrupt_passed = _assert_interrupts(
                            turn.get("expected_interrupt"), current_state.next
                        )

                        expected_state = turn.get("expected_state", {})
                        passed, total = _validate_state(expected_state, current_state.values)

                        # Check if the dataset specifies tools that must have been called this turn
                        expected_tools = turn.get("expected_tool_calls", [])
                        tools_passed = _validate_tool_calls(expected_tools, tracker, config)

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

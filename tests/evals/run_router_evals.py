import argparse
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DEPLOYMENT_ENV_PATH = os.path.join(
    os.path.dirname(REPO_ROOT), "langgraph-workspace-agent-deployment", ".env"
)

if os.path.exists(DEPLOYMENT_ENV_PATH):
    load_dotenv(dotenv_path=DEPLOYMENT_ENV_PATH)
else:
    load_dotenv()

from langchain_core.messages import HumanMessage  # noqa: E402
from langfuse import Langfuse, get_client, propagate_attributes  # noqa: E402

# Directly import just the parsing node, bypassing the full graph and tool execution
from src.workspace_agent.orchestrator.nodes import parse_intent_node  # noqa: E402

DATASET_NAME = "workspace_agent_eval_router_nlu"
DATASET_PATH = os.path.join(CURRENT_DIR, "datasets", "01_router_nlu.json")

# NLU classifications should be strict for core intents
TARGET_THRESHOLD = 100.0


def init_langfuse():
    """Initializes the Langfuse client."""
    env_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    if "langfuse-" in env_host or "web" in env_host or "server" in env_host:
        os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
    return get_client()


def sync_dataset(lf: Langfuse):
    """Upserts the routing cases to Langfuse."""
    print(f"Synchronizing Router dataset '{DATASET_NAME}' to Langfuse...")
    with open(DATASET_PATH) as f:
        cases = json.load(f)

    lf.create_dataset(
        name=DATASET_NAME,
        description="Component evaluation for NLU Router classification accuracy.",
    )

    for i, case in enumerate(cases):
        # Auto-generate a case_id if it's missing from the JSON
        case_id = case.get("case_id", f"router_case_{i + 1:02d}")
        case["case_id"] = case_id

        lf.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=case["input"],
            expected_output=case["expected_output"],
            metadata=case,
        )
    print(f"✅ Successfully synced {len(cases)} Router cases.")


def _evaluate_single_case(item, lf: Langfuse, session_name: str) -> bool:
    """Executes a single NLU router test case and logs to Langfuse."""
    case_data = item.metadata
    case_id = case_data["case_id"]
    user_input = case_data["input"]
    expected_output = case_data["expected_output"]

    print(f"Evaluating: {case_id}")
    passed = True

    try:
        # Start an isolated trace using the v4 OpenTelemetry context manager
        with lf.start_as_current_observation(as_type="span", name=f"router_{case_id}") as span:
            span.update(input=user_input)
            trace_id = lf.get_current_trace_id()

            with propagate_attributes(session_id=session_name):
                # 1. Construct the minimal state required by parse_intent_node
                mock_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "original_instruction": user_input,
                }

                # 2. Execute ONLY the parsing node (Zero side effects)
                result_state = parse_intent_node(mock_state)
                output_to_log = {}

                # 3. Grade all keys defined in the JSON expected_output dictionary
                for key, expected_val in expected_output.items():
                    actual_val = result_state.get(key)

                    # Sanitize LLM string bleeding for nulls
                    if isinstance(actual_val, str) and actual_val.strip().lower() in [
                        "null",
                        "none",
                    ]:
                        actual_val = None

                    output_to_log[key] = actual_val

                    if expected_val == "__NOT_NULL__":
                        if actual_val is not None:
                            print(f"  ✅ {key}: (Populated)")
                        else:
                            print(f"  ❌ {key}: Expected populated value, got None")
                            passed = False
                    elif actual_val == expected_val:
                        print(f"  ✅ {key}: {actual_val}")
                    else:
                        print(f"  ❌ {key}: Expected {expected_val}, got {actual_val}")
                        passed = False

                # 4. Log scoring directly to the Langfuse trace
                span.update(output=output_to_log)
                lf.create_score(
                    trace_id=trace_id,
                    name="routing_accuracy",
                    value=1.0 if passed else 0.0,
                )

                try:
                    if hasattr(lf, "api") and hasattr(lf.api, "dataset_run_items"):
                        lf.api.dataset_run_items.create(
                            dataset_item_id=item.id, run_name=session_name, trace_id=trace_id
                        )
                except Exception:
                    pass

    except Exception as e:
        print(f"  ❌ FATAL ERROR: {e}")
        passed = False

    return passed


def run_evaluations(lf: Langfuse):
    """Executes the NLU routing node in total isolation."""
    print("Initializing Router NLU Evaluations...\n")
    dataset = lf.get_dataset(DATASET_NAME)
    print(f"Starting evaluations for {len(dataset.items)} cases...\n")

    total_score = 0
    max_score = len(dataset.items)
    session_name = f"Router_Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for item in dataset.items:
        # Delegate execution and grading to the helper function
        if _evaluate_single_case(item, lf, session_name):
            total_score += 1

    lf.flush()
    accuracy = (total_score / max_score) * 100 if max_score > 0 else 0

    print(f"\nRouter Evaluation Complete. Final Score: {total_score}/{max_score} ({accuracy:.1f}%)")

    if accuracy < TARGET_THRESHOLD:
        print(f"\n❌ System Health Warning: Router accuracy dropped below {TARGET_THRESHOLD}%.")
        sys.exit(1)

    print(f"\n🚀 System Health Excellent: Router accuracy meets the {TARGET_THRESHOLD}% threshold!")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Router NLU Evaluations")
    parser.add_argument("--sync", action="store_true", help="Sync local Router dataset to Langfuse")
    parser.add_argument(
        "--eval", action="store_true", help="Run Router evaluations against dataset"
    )
    args = parser.parse_args()

    lf_client = init_langfuse()

    if args.sync:
        sync_dataset(lf_client)

    if args.eval:
        run_evaluations(lf_client)

    if not args.sync and not args.eval:
        parser.print_help()

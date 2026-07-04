"""Shared assertion utilities for LangGraph state and mock tool execution."""


def validate_state(expected_state: dict, actual_state_values: dict) -> tuple[int, int]:
    """Helper to assert the graph's internal state matches expectations."""
    passed_criteria = 0
    total_criteria = len(expected_state)

    for key, expected_val in expected_state.items():
        actual_val = actual_state_values.get(key)

        if expected_val == "__NOT_NULL__":
            if actual_val is not None:
                passed_criteria += 1
                print(f" ✅ {key}: (Populated)")
            else:
                print(f" ❌ {key}: Expected populated value, got None")

        elif expected_val == "__NOT_EMPTY__":
            if actual_val:  # Evaluates truthiness (catches empty lists/strings/dicts and None)
                passed_criteria += 1
                print(f" ✅ {key}: (Not Empty)")
            else:
                print(f" ❌ {key}: Expected non-empty value, got {repr(actual_val)}")

        elif actual_val == expected_val:
            passed_criteria += 1
            print(f" ✅ {key}: {actual_val}")

        else:
            print(f" ❌ {key}: Expected {expected_val}, got {actual_val}")

    return passed_criteria, total_criteria


def _extract_called_tools(tracker) -> list[dict]:
    """Helper to extract chronologically executed tool calls directly from the mock tracker."""
    return [
        {"name": call.get("tool"), "args": call.get("kwargs", {})}
        for call in tracker.invocation_history
    ]


def _args_match(expected_args: dict, actual_args: dict) -> bool:
    """Helper to evaluate if actual arguments contain the expected subsets."""
    for k, expected_val in expected_args.items():
        actual_val = actual_args.get(k)

        if actual_val is None:
            return False

        # If string, use substring matching (handles absolute paths dynamically)
        if isinstance(expected_val, str) and isinstance(actual_val, str):
            if expected_val not in actual_val:
                return False
        # Exact match for booleans, ints, etc.
        elif actual_val != expected_val:
            return False

    return True


def _find_tool_match(expected, called_tools: list[dict], start_idx: int) -> int:
    """Finds the index of the first matching tool starting from start_idx."""
    if isinstance(expected, str):
        for i in range(start_idx, len(called_tools)):
            if called_tools[i]["name"] == expected:
                return i
        return -1

    expected_name = expected.get("name")
    expected_args = expected.get("args_contain", {})

    for i in range(start_idx, len(called_tools)):
        ct = called_tools[i]
        if ct["name"] == expected_name and _args_match(expected_args, ct["args"]):
            return i

    return -1


def validate_tool_calls(expected_tools: list, tracker) -> int:
    """Helper to sequentially validate actual executed tool calls for the current turn."""
    if not expected_tools:
        return 0

    tools_passed = 0
    called_tools = _extract_called_tools(tracker)

    # Acts as a pointer. Matches must occur AFTER the previously matched tool.
    search_idx = 0

    for expected in expected_tools:
        match_idx = _find_tool_match(expected, called_tools, search_idx)

        if match_idx != -1:
            tools_passed += 1
            search_idx = match_idx + 1  # Advance the pointer to strictly enforce sequence

            if isinstance(expected, str):
                print(f" ✅ Sequence verified: {expected}")
            else:
                name = expected.get("name")
                args = ", ".join(f"{k}='{v}'" for k, v in expected.get("args_contain", {}).items())
                print(f" ✅ Sequence verified: {name} ({args})")
        elif isinstance(expected, str):
            print(f" ❌ Missing/Out-of-order tool: {expected}")
        else:
            print(
                f" ❌ Missing/Out-of-order tool: {expected.get('name')} "
                f"with args {expected.get('args_contain')}"
            )

    # Clear mock history after asserting to prepare for the next turn
    tracker.invocation_history.clear()

    return tools_passed


def assert_interrupts(expected_interrupt: str | None, actual_next_nodes: tuple) -> bool:
    """Helper to assert if the graph paused at the expected node."""
    if expected_interrupt:
        if expected_interrupt in actual_next_nodes:
            print(f" ✅ Graph Paused at: {expected_interrupt}")
            return True

        # Split across lines to fix E501
        print(
            f" ❌ Graph failed to pause at {expected_interrupt}. Currently at: {actual_next_nodes}"
        )
        return False

    if not actual_next_nodes:
        print(" ✅ Graph completed/exited successfully.")
        return True

    print(f" ❌ Graph unexpectedly paused at: {actual_next_nodes}")
    return False

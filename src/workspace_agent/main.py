import io
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import docker
import httpx
import redis
from chromadb import PersistentClient
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from langchain_core.messages import HumanMessage
from langfuse import Langfuse
from pypdf import PdfReader
from redis.exceptions import LockError

from src.workspace_agent.core.config import settings
from src.workspace_agent.integrations.github_webhook import (
    parse_agentic_ci_trigger,
    parse_github_pr_action,
    verify_github_signature,
)
from src.workspace_agent.integrations.telegram import send_telegram_message
from src.workspace_agent.llm.callbacks import get_langfuse_callback
from src.workspace_agent.orchestrator.graph import agent_app, pr_app

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory fallback for thread tracking (used during unit tests or Redis downtime)
RUNTIME_STATE = {}

# ==========================================
# Lifespan & IO Management
# ==========================================


@dataclass
class IODependencies:
    """Encapsulates all external IO clients to prevent function signature bloat."""

    redis_client: redis.Redis | None = None
    chroma_collection: object | None = None
    docker_client: object | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup initialization and teardown of IO clients.
    Attaches persistent clients to `app.state` to prevent global scope race conditions.
    """
    logger.info("Initializing I/O clients...")

    # 1. ChromaDB
    try:
        chroma_path = os.path.join(os.getcwd(), ".chroma")
        chroma_client = PersistentClient(path=chroma_path)
        app.state.chroma_collection = chroma_client.get_or_create_collection(name="agent_memory")
        logger.info("ChromaDB connected.")
    except Exception as e:
        logger.warning(f"Could not connect to ChromaDB: {e}")
        app.state.chroma_collection = None

    # 2. Docker
    try:
        app.state.docker_client = docker.from_env()
        logger.info("Docker daemon connected.")
    except Exception as e:
        logger.warning(f"Could not connect to Docker daemon: {e}")
        app.state.docker_client = None

    # 3. Redis
    try:
        redis_client = redis.Redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
            socket_timeout=2.0,
        )
        redis_client.ping()  # Validate connection on startup
        app.state.redis_client = redis_client
        logger.info("Redis connected.")
    except Exception as e:
        logger.warning(
            f"Persistent Redis connection failed: {e}. Falling back to in-memory routing."
        )
        app.state.redis_client = None

    yield  # Application processes requests here

    logger.info("Tearing down I/O clients...")
    if getattr(app.state, "redis_client", None):
        app.state.redis_client.close()

    if getattr(app.state, "docker_client", None):
        app.state.docker_client.close()


app = FastAPI(title="LangGraph Workspace Agent", lifespan=lifespan)

# ==========================================
# Session Management Router
# ==========================================


def get_active_thread(chat_id: str, redis_client: redis.Redis | None = None) -> str:
    """Fetches the active thread ID, preferring Redis for cross-restart persistence."""
    if redis_client:
        try:
            val = redis_client.get(f"active_thread_{chat_id}")
            if val:
                return val
        except Exception as e:
            logger.error(f"Redis get error: {e}")

    return RUNTIME_STATE.get(chat_id, chat_id)


def set_active_thread(chat_id: str, thread_id: str, redis_client: redis.Redis | None = None):
    """Saves the active thread ID to both memory and Redis."""
    RUNTIME_STATE[chat_id] = thread_id
    if redis_client:
        try:
            redis_client.set(f"active_thread_{chat_id}", thread_id)
        except Exception as e:
            logger.error(f"Redis set error: {e}")


# ==========================================
# Private Helper Functions
# ==========================================


def _build_initial_inputs(text: str) -> dict:
    """Helper to construct the initial state payload for a new thread."""
    return {
        "messages": [HumanMessage(content=text)],
        "original_instruction": text,
        "t1_base_calls": 0,
        "t2_standard_calls": 0,
        "t3_frontier_calls": 0,
        "execution_retry_count": 0,
        "latest_traceback_error": None,
        "clarification_question": None,
        "disambiguation_options": None,
        "is_aborted": False,
        "modified_tex_files": [],
    }


def _build_state_update(text: str, current_state) -> dict:
    """Helper to determine state updates when resuming from a LangGraph breakpoint."""
    pending_node = current_state.next[0]
    state_update = {"messages": [HumanMessage(content=text)]}

    # handle explicit human abort commands
    if text.strip().lower() in ["abort", "cancel", "stop", "nevermind", "exit"]:
        state_update["is_aborted"] = True

    # check context using the values stored in the paused state
    elif pending_node == "human_pr_node":
        if current_state.values.get("pending_pr_url"):
            state_update["human_approved"] = text.strip().upper() == "LGTM"

    return state_update


def _invoke_agent_graph(
    text: str,
    config: dict,
    current_state,
    io_deps: IODependencies,
    lock: object | None,
) -> tuple[dict, object | None]:
    """Handles the branching logic of invoking the graph (new, resume, or recover)."""
    if current_state.next:
        pending_node = current_state.next[0]

        # if the graph is stuck on an execution node due to a previous crash, transparently
        # reset the thread so the new command isn't trapped in a dirty state.
        if pending_node not in ["human_clarify_node", "human_pr_node"]:
            thread_id = config["configurable"]["thread_id"]
            logger.warning(f"Thread {thread_id} stuck at {pending_node}. Auto-recovering.")

            # release lock on the broken thread
            try:
                clean_old_config = {"configurable": {"thread_id": thread_id}}
                agent_app.update_state(clean_old_config, {"is_busy": False})
            except Exception as e:
                logger.warning(f"Error clearing broken thread state: {e}")

            if lock:
                try:
                    lock.release()
                except Exception as e:
                    logger.warning(f"Error releasing old thread lock: {e}")

            # transparently migrate the user to a clean thread ID
            chat_id = config["configurable"]["user_id"]
            new_thread = f"{chat_id}_{int(time.time())}"
            set_active_thread(chat_id, new_thread, io_deps.redis_client)
            config["configurable"]["thread_id"] = new_thread

            # engage states and grab the Redis lock for the newly created thread
            try:
                clean_new_config = {"configurable": {"thread_id": new_thread}}
                agent_app.update_state(clean_new_config, {"is_busy": True})
            except Exception as e:
                logger.warning(f"Error setting is_busy state: {e}")

            if io_deps.redis_client:
                # Set thread_local=False so background tasks can release the lock
                lock = io_deps.redis_client.lock(
                    f"agent_lock_{new_thread}", timeout=300, thread_local=False
                )
                lock.acquire(blocking=False)
            else:
                lock = None

            inputs = _build_initial_inputs(text)
            return agent_app.invoke(inputs, config), lock

        # normal, healthy breakpoint resumption
        state_update = _build_state_update(text, current_state)
        state_update["is_busy"] = True

        try:
            # Use clean config to avoid checkpoint_id or serialization issues
            clean_resume_config = {
                "configurable": {"thread_id": config["configurable"]["thread_id"]}
            }
            agent_app.update_state(clean_resume_config, state_update, as_node=pending_node)
        except Exception as e:
            logger.error(f"Error updating state during resumption: {e}")

        return agent_app.invoke(None, config), lock

    # initialize a brand new instruction
    try:
        clean_config = {"configurable": {"thread_id": config["configurable"]["thread_id"]}}
        agent_app.update_state(clean_config, {"is_busy": True})
    except Exception as e:
        logger.error(f"Error setting is_busy state for new instruction: {e}")

    inputs = _build_initial_inputs(text)
    return agent_app.invoke(inputs, config), lock


def _format_final_response(final_state: dict) -> str:
    """Helper to extract the final AI message and dynamically append telemetry."""
    if not final_state or "messages" not in final_state:
        return "⚠️ An error occurred: No response was generated."

    last_message_content = final_state["messages"][-1].content

    # LangChain LLMs (like Gemini) sometimes return content as a list of dictionaries
    if isinstance(last_message_content, list):
        text_blocks = []
        for block in last_message_content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_blocks.append(block.get("text", ""))
            elif isinstance(block, str):
                text_blocks.append(block)
        last_message_text = "\n".join(text_blocks)
    else:
        last_message_text = str(last_message_content)

    if settings.agent.show_telemetry:
        t1 = final_state.get("t1_base_calls", 0)
        t2 = final_state.get("t2_standard_calls", 0)
        t3 = final_state.get("t3_frontier_calls", 0)

        # only append if at least one call was made during the session
        if t1 > 0 or t2 > 0 or t3 > 0:
            telemetry_footer = f"\n\n---\n📊 *Telemetry: T1: {t1} | T2: {t2} | T3: {t3}*"
            last_message_text += telemetry_footer

    return last_message_text


# ==========================================
# Core Execution & Interaction
# ==========================================


def is_agent_busy(thread_id: str) -> bool:
    """
    Queries the LangGraph checkpointer to see if the agent is currently
    computing a task for this thread.
    Note: This is now strictly a fallback mechanism if Redis is offline.
    """
    try:
        state = agent_app.get_state({"configurable": {"thread_id": thread_id}})
        return state.values.get("is_busy", False)
    except Exception:
        return False


def process_ci_trigger(ci_trigger: dict, io_deps: IODependencies):
    """
    Background Task: Stateless, isolated execution of the Agentic CI pipeline.
    Invokes the Pull Request Sub-Graph directly without polluting conversational memory.
    """
    logger.info(f"Processing CI Trigger: {ci_trigger}")

    repo_name = ci_trigger.get("repo_name")
    pr_number = ci_trigger.get("pr_number")
    repo_full_name = ci_trigger.get("repo_full_name")
    commit_sha = ci_trigger.get("commit_sha")

    # If it's a ChatOps trigger, we need to fetch the latest commit SHA from the PR
    if ci_trigger.get("is_chatops") and not commit_sha:
        token = os.getenv("GITHUB_TOKEN")
        if token and repo_full_name and pr_number:
            try:
                url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}"
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github.v3+json",
                }
                with httpx.Client() as client:
                    resp = client.get(url, headers=headers, timeout=10.0)
                    if resp.status_code == httpx.codes.OK:
                        pr_data = resp.json()
                        commit_sha = pr_data.get("head", {}).get("sha")
                        ci_trigger["commit_sha"] = commit_sha
            except Exception as e:
                logger.error(f"Failed to fetch PR details for ChatOps CI trigger: {e}")

    if not commit_sha:
        logger.warning("Agentic CI aborted: Missing commit SHA.")
        return

    # Resolve the absolute workspace path from the repo name mapping
    target_path = None
    for ws_name, ws_config in settings.workspaces.items():
        # safely expand user root path just in case
        expanded_path = os.path.expanduser(ws_config.path).rstrip("/")
        folder_name = os.path.basename(expanded_path)

        # Match exact workspace name, exact folder name, or dev clone suffixes (e.g., -dev)
        if repo_name in (ws_name, folder_name) or folder_name.startswith(f"{repo_name}-"):
            target_path = ws_config.path
            break

    if not target_path:
        logger.warning(
            f"Agentic CI aborted: Could not resolve workspace path for repo '{repo_name}'"
        )
        return

    ci_trigger["workspace_absolute_path"] = target_path

    # Construct a clean, stateless config for the subgraph
    # We do NOT use the main checkpointer here; CI runs synchronously
    config = {
        "configurable": {
            "thread_id": f"ci_{repo_name}_{pr_number}_{commit_sha}",
            "docker_client": io_deps.docker_client,
        }
    }

    try:
        logger.info(f"Invoking CI Subgraph for PR #{pr_number} at {commit_sha}")
        # Send it directly to the PR App; it routes to `agentic_ci` automatically
        pr_app.invoke(ci_trigger, config)
    except Exception as e:
        logger.error(f"Agentic CI execution failed: {e}")


class LockHeartbeat(threading.Thread):
    """
    Watchdog Thread: Periodically extends a Redis lock's TTL to prevent
    timeouts during long-running LangGraph executions.
    """

    def __init__(self, lock, extend_interval=60):
        super().__init__(daemon=True)
        self.lock = lock
        self.extend_interval = extend_interval
        self.stop_event = threading.Event()

    def run(self):
        # Loop until the stop_event is set or the interval passes
        while not self.stop_event.wait(self.extend_interval):
            try:
                # reacquire() safely resets the lock's TTL back to its initial timeout (300s)
                self.lock.reacquire()
                logger.debug("Redis lock TTL successfully extended via Watchdog heartbeat.")
            except LockError as e:
                # The lock was lost (e.g., Redis restart) or hijacked. Stop trying.
                logger.warning(f"Watchdog lost ownership of the Redis lock: {e}")
                break
            except Exception as e:
                # General network blip. Log and try again next loop.
                logger.warning(f"Watchdog encountered network error extending lock: {e}")

    def stop(self):
        self.stop_event.set()


def process_agent_message(
    chat_id: str,
    text: str,
    thread_id: str,
    io_deps: IODependencies | None = None,
    lock: object | None = None,
):
    """
    The main execution wrapper. Feeds Telegram messages into the LangGraph state machine,
    handles breakpoint resumptions, and streams the AI's response back to Telegram.
    """
    logger.info(f"Executing graph for thread {thread_id}: {text}")

    # Start the Watchdog heartbeat
    heartbeat = None
    if lock:
        heartbeat = LockHeartbeat(lock)
        heartbeat.start()

    io_deps = io_deps or IODependencies()

    # Safely initialize the handler. Will be None if API keys are missing in .env
    langfuse_handler = get_langfuse_callback(
        session_id=thread_id, user_id=str(chat_id), tags=["telegram_bot"]
    )
    callbacks = [langfuse_handler] if langfuse_handler else []

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": str(chat_id),
            "chroma_collection": io_deps.chroma_collection,
            "docker_client": io_deps.docker_client,
        },
        "callbacks": callbacks,
    }

    try:
        current_state = agent_app.get_state(config)
        final_state, lock = _invoke_agent_graph(text, config, current_state, io_deps, lock)

        # format and send the response
        response_text = _format_final_response(final_state)
        send_telegram_message(chat_id, response_text)

    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        send_telegram_message(
            chat_id, "⚠️ A critical error occurred during execution. Please check the server logs."
        )
    finally:
        # Stop the Watchdog safely
        if heartbeat:
            heartbeat.stop()
            # Wait up to 2 seconds to ensure the thread finishes extending before we release
            heartbeat.join(timeout=2.0)

        # Guarantee the lock is released when execution pauses (human input) or finishes
        try:
            # Pull the potentially updated thread ID from the config in case
            # it was switched during auto-recovery.
            active_thread_id = config["configurable"]["thread_id"]
            clean_config = {"configurable": {"thread_id": active_thread_id}}
            agent_app.update_state(clean_config, {"is_busy": False})
        except Exception as e:
            logger.error(f"Failed to clear is_busy state: {e}")

        # Release the distributed lock
        if lock:
            try:
                lock.release()
                logger.debug(f"Redis lock released for thread {active_thread_id}")
            except LockError:
                # Lock was already released or lost; safe to ignore during teardown
                pass
            except Exception as e:
                # Ignore if the lock expired naturally during a long execution cycle
                logger.debug(f"Could not release Redis lock: {e}")

        # 1. Flush local Langchain handler buffers
        if langfuse_handler:
            try:
                if hasattr(langfuse_handler, "flush"):
                    langfuse_handler.flush()
            except Exception:
                pass

        # 2. Force flush global OTEL queue before the background task dies
        try:
            Langfuse().flush()
        except Exception as e:
            logger.error(f"Failed to flush global Langfuse queue: {e}")


# ==========================================
# Webhook Ingress Routes
# ==========================================


async def _process_telegram_attachment(document: dict, current_text: str) -> str:
    """Helper to asynchronously download and parse Telegram file attachments."""
    file_id = document.get("file_id")
    file_name = document.get("file_name", "attached_file.txt").lower()
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not bot_token:
        return current_text

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # 1. Ask Telegram for the file path
            file_info_url = f"https://api.telegram.org/bot{bot_token}/getFile?file_id={file_id}"
            file_info_resp = await client.get(file_info_url)
            file_info = file_info_resp.json()

            if not file_info.get("ok"):
                error_msg = (
                    "\n\n[System Note: User attempted to attach a file, but it failed "
                    "to download (likely exceeding Telegram's 20MB bot download limit).]"
                )
                return current_text + error_msg

            file_path = file_info["result"]["file_path"]
            download_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"

            # 2. Download the RAW binary content using the same pooled client
            response = await client.get(download_url)
            response.raise_for_status()
            file_bytes = response.content

        # 3. Parse based on file type
        extracted_text = ""

        if file_name.endswith(".pdf"):
            # Process as PDF in-memory
            pdf_file = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_file)
            extracted_text = "\n".join(
                [page.extract_text() for page in reader.pages if page.extract_text()]
            )
        else:
            # For all other files (including unknown extensions or extensionless files
            # like 'Dockerfile'), attempt UTF-8 decoding.
            try:
                extracted_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Gracefully reject unsupported true binaries (images, videos, zips, etc.)
                extracted_text = (
                    f"[System Note: The user attached a file '{file_name}', but its "
                    "format is not currently supported for text extraction.]"
                )

        # 4. Append the contents to the instruction
        if extracted_text.strip():
            attachment_context = f"\n\n--- Contents of {file_name} ---\n{extracted_text}\n---"
            return current_text + attachment_context

    except httpx.ReadTimeout:
        logger.error(f"Timeout while downloading Telegram attachment {file_name}")
        return (
            current_text
            + f"\n\n[System Note: Timed out while attempting to download the file '{file_name}'.]"
        )
    except Exception as e:
        logger.error(f"Failed to download or parse Telegram attachment: {e}")

    return current_text


@app.post("/webhook")
async def telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: str = Header(None),
):
    """
    The primary ingress route for Telegram updates.
    """
    # fast-fail security check
    expected_secret = os.getenv("TELEGRAM_SECRET_TOKEN")
    if (
        not expected_secret
        or not x_telegram_bot_api_secret_token
        or not secrets.compare_digest(x_telegram_bot_api_secret_token, expected_secret)
    ):
        logger.warning("Unauthorized access attempt: Invalid or missing Secret Token.")
        raise HTTPException(status_code=401, detail="Unauthorized")

    # parse payload
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON") from e

    # extract message details
    message = payload.get("message", {})
    if not message:
        # telegram sometimes sends edits or other updates; ignore if not a standard message
        return {"status": "ignored"}

    chat_id = str(message.get("chat", {}).get("id", ""))
    expected_chat_id = os.getenv("AUTHORIZED_OWNER_CHAT_ID")

    text = message.get("text", "").strip()
    caption = message.get("caption", "").strip()

    # If it's an attachment, Telegram uses 'caption' instead of 'text' for the prompt
    if not text and caption:
        text = caption

    document = message.get("document")
    photo = message.get("photo")

    if document:
        text = await _process_telegram_attachment(document, text)
    elif photo:
        warning_msg = (
            "[System Note: Images are not currently supported, please send as a file/document.]"
        )
        # Append cleanly whether they included a caption or not
        text = f"{text}\n\n{warning_msg}" if text else warning_msg

    # strict whitelisting
    if not chat_id or chat_id != expected_chat_id:
        logger.warning(f"Unauthorized chat ID detected: {chat_id}. Dropping payload.")
        # return 200 OK so Telegram doesn't retry the delivery, but silently drop it
        return {"status": "ok"}

    # Pack dynamic IO clients
    io_deps = IODependencies(
        redis_client=getattr(request.app.state, "redis_client", None),
        chroma_collection=getattr(request.app.state, "chroma_collection", None),
        docker_client=getattr(request.app.state, "docker_client", None),
    )

    # memory management (/reset)
    if text == "/reset":
        new_thread = f"{chat_id}_{int(time.time())}"
        set_active_thread(chat_id, new_thread, io_deps.redis_client)
        logger.info(f"Memory reset. New thread ID: {new_thread}")
        # Delegate to background task to prevent blocking the async event loop
        background_tasks.add_task(
            send_telegram_message, chat_id, "Memory cleared. Started a fresh session."
        )
        return {"status": "ok"}

    current_thread = get_active_thread(chat_id, io_deps.redis_client)

    # concurrency protection (Redis distributed lock)
    is_locked = False
    lock = None
    if io_deps.redis_client:
        # Set thread_local=False so background tasks can release the lock
        lock = io_deps.redis_client.lock(
            f"agent_lock_{current_thread}", timeout=300, thread_local=False
        )
        # blocking=False guarantees an immediate return if the agent
        # is busy processing another concurrent message
        if not lock.acquire(blocking=False):
            is_locked = True
    else:
        # Fallback to checking the LangGraph state if Redis is offline
        is_locked = is_agent_busy(current_thread)

    if is_locked:
        logger.info("Agent is busy. Rejecting concurrent command.")
        # Delegate to background task to prevent blocking the async event loop
        background_tasks.add_task(
            send_telegram_message,
            chat_id,
            "⚠️ I am currently executing a task. Please wait for it to finish.",
        )
        return {"status": "ok"}

    # asynchronous handoff
    logger.info("Authorized payload accepted. Queuing background task.")
    background_tasks.add_task(
        process_agent_message,
        chat_id,
        text,
        current_thread,
        io_deps,
        lock,
    )

    # return immediately to satisfy Telegram's 5-second timeout requirement
    return {"status": "ok"}


@app.post("/github/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str = Header(None),
):
    """Listens for Pull Request events to auto-advance the agent or trigger CI."""
    payload_body = await request.body()
    if not verify_github_signature(payload_body, x_hub_signature_256):
        logger.warning("Unauthorized access attempt: Invalid GitHub Signature.")
        raise HTTPException(status_code=401, detail="Invalid GitHub signature")

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON") from e

    # Pack dynamic IO clients
    io_deps = IODependencies(
        redis_client=getattr(request.app.state, "redis_client", None),
        chroma_collection=getattr(request.app.state, "chroma_collection", None),
        docker_client=getattr(request.app.state, "docker_client", None),
    )

    # 1. Agentic CI Trigger Parsing
    ci_trigger = parse_agentic_ci_trigger(payload)
    if ci_trigger:
        logger.info("GitHub Webhook: Agentic CI trigger detected.")
        background_tasks.add_task(process_ci_trigger, ci_trigger, io_deps)
        return {"status": "ci_triggered"}

    # 2. PR Lifecycle Parsing (Merge / Abort)
    action_signal = parse_github_pr_action(payload)
    if action_signal:
        chat_id = os.getenv("AUTHORIZED_OWNER_CHAT_ID")
        thread_id = get_active_thread(chat_id, io_deps.redis_client)

        # concurrency protection (Redis distributed lock)
        is_locked = False
        lock = None
        if io_deps.redis_client:
            # Set thread_local=False so background tasks can release the lock
            lock = io_deps.redis_client.lock(
                f"agent_lock_{thread_id}", timeout=300, thread_local=False
            )
            if not lock.acquire(blocking=False):
                is_locked = True
        else:
            is_locked = is_agent_busy(thread_id)

        if is_locked:
            logger.info(
                "Agent is busy. Ignoring github webhook action to prevent concurrent execution."
            )
            return {"status": "ignored_busy"}

        logger.info(
            f"GitHub Webhook: Triggering background queue advancing with signal -> {action_signal}"
        )
        background_tasks.add_task(
            process_agent_message, chat_id, action_signal, thread_id, io_deps, lock
        )
        return {"status": "acknowledged_merge"}

    return {"status": "ignored"}

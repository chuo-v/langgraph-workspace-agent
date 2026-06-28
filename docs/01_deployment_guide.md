# 01. Deployment Guide

This guide covers the end-to-end process of setting up the LangGraph Workspace Agent. Because the agent is designed to run asynchronously and receive commands remotely via Telegram, the deployment process requires setting up a persistent host environment, securely configuring API tokens, and establishing a webhook tunnel.

## Step 1: Host Machine Configuration (Headless Best Practices)

The agent operates as a persistent background daemon. If you are deploying this on a primary workstation or a dedicated headless server (like a Mac Studio), you must configure the operating system to ensure container lifecycles are not interrupted during idle periods.

For macOS environments, apply the following system settings:
* **Energy Saver & Boot:** Navigate to `System Settings > Energy Saver` and enable **"Start up automatically after a power failure."**
* **Display & Sleep Lockouts:** Navigate to `System Settings > Displays > Advanced...` and toggle on **"Prevent automatic sleeping on power adapter when the display is off."** Additionally, configure your lock screen settings to **"Never"** turn off the display to ensure persistent container operations aren't halted by aggressive OS-level resource throttling.
* **FileVault Warning:** If running completely headless and relying on auto-boot after power failures, consider the implications of FileVault (`System Settings > Privacy & Security > FileVault`). If enabled, macOS halts the boot sequence to wait for a password *before* loading background services like Docker, which will lock you out remotely and prevent the agent from starting.

## Step 2: System Prerequisites

Ensure the following tools are installed on your host machine:
* **Git:** For cloning repositories and allowing the agent's MCP servers to manage your code.
* **Docker & Docker Compose:** The entire orchestration layer, memory store, and sandbox environments run in containers. If using Apple Silicon, ensure you install the ARM64-compatible version of Docker Desktop.
* **LLM Inference Engine (Cloud or Local):** The agent relies on the Base Tier (Tier 1) as its foundational engine to perform background tasks, including intent routing and memory extraction. You can use fast, low-cost cloud models (like Gemini 2.5 Flash) for this tier. If you prefer to keep these tasks entirely local, you can optionally install [Ollama](https://ollama.com/) natively on the host machine and pull the default models (`ollama pull qwen2.5:32b` and `ollama pull bge-small`).

> **Hardware Constraints & Cloud Alternatives:** If your host machine lacks the memory to comfortably run larger local models (like the default 32B parameter model), it is highly recommended to configure a low-cost cloud provider (e.g., Gemini Flash or DeepSeek) for your Base Tier. While you *can* step down to smaller local alternatives (like 8B or 14B class models), please note that their accuracy in complex routing and evaluation scenarios may be lower and has not been rigorously tested.

## Step 3: Cloning and Securing the Environment

It is recommended to organize your repositories in a standard workspace directory.

```bash
mkdir -p ~/git
cd ~/git
git clone git@github.com:chuo-v/langgraph-workspace-agent.git
cd langgraph-workspace-agent
```

Next, duplicate the environment template:

```bash
cp .env.example .env
```

⚠️ **WARNING: Never commit your `.env` file** Your `.env` file contains highly sensitive API keys and GitHub tokens that grant access to your repositories. While this project includes a `.gitignore` to prevent accidental commits, you must be careful not to force-add this file or copy it to other publicly tracked directories.

Immediately restrict the local file permissions to prevent unauthorized read access on your host machine:

```bash
chmod 600 .env
```

### Populating the `.env` File

Open the `.env` file and populate the necessary credentials:

1. **Infrastructure Secrets (Docker Compose):**
* You must generate secure passwords for the persistent databases and Langfuse UI sessions to prevent unauthorized internal access.
* Run `openssl rand -hex 32` in your terminal to generate safe, alphanumeric strings for `POSTGRES_PASSWORD`, `CLICKHOUSE_PASSWORD`, and `MINIO_ROOT_PASSWORD`.
* Run `openssl rand -base64 32` in your terminal to generate cryptographic session keys for `LANGFUSE_NEXTAUTH_SECRET` and `LANGFUSE_SALT`.


2. **Telegram Credentials:**
* Message `@BotFather` on Telegram, create a new bot, and retrieve the **HTTP API Token**.
* Send a message to your new bot, then visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` to find your personal `chat.id`. Set this as your `AUTHORIZED_OWNER_CHAT_ID` to strictly whitelist your account.
* Create a custom `TELEGRAM_SECRET_TOKEN` (e.g., using `openssl rand -hex 16`) to cryptographically verify incoming webhook payloads.


3. **GitHub Integrations:**
* Provide your account handle in `GITHUB_USERNAME`.
* Navigate to GitHub Developer Settings and generate a Fine-Grained Personal Access Token (`GITHUB_TOKEN`).
* Grant it `Read and Write` permissions for **Contents** (to push branches) and **Pull Requests** (to open PRs). Limit its scope *only* to the repositories the agent is allowed to touch.
* Create a custom `GITHUB_WEBHOOK_SECRET` for validating repository event webhooks.
* **Webhook Noise Mitigation:** When setting up the actual webhook on your GitHub repository settings, select **"Let me select individual events"** and check *only* **Pull requests** and **Issue comments**. Leaving it on the default "Send me everything" will cause GitHub to blast payloads for every minor repository event (stars, branch pushes, etc.), which will needlessly flood your agent's server logs.

4. **LLM Tiering:**
* Define your `BASE_PROVIDER`, `STANDARD_PROVIDER`, and `FRONTIER_PROVIDER` (e.g., `anthropic`, `deepseek`, `gemini`, `openai`).
* Add your corresponding API keys for the chosen providers (e.g., `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`).
* *Note:* If you are utilizing a local Ollama instance for your Base Tier, it will rely on your `OLLAMA_API_BASE` and requires no keys.


5. **Filesystem & State Configuration:**
* Set `HOST_WORKSPACE_DIR` to the absolute path on your host machine where your repositories are stored (e.g., `/Users/username/git`). This directory is mounted symmetrically into the agent container.
* Optionally define `WORKSPACE_AGENT_CONFIG_PATH` and `WORKSPACE_AGENT_STATE_DIR` if you want to store your `config.yaml` and state profiles outside the default repository directory.


6. **Sandbox Security & Airgap:**
* The ephemeral sandboxes allow outbound DNS to resolve package managers. Set `TRUSTED_DNS_PRIMARY` and `TRUSTED_DNS_SECONDARY` if you operate behind a strict corporate firewall or need specific resolvers instead of the defaults (1.1.1.1 / 9.9.9.9).


7. **Docker Socket Permissions (Linux Only):**
* If running natively on Linux, the orchestrator needs permission to communicate with the host's Docker daemon. Un-comment and set `GATEWAY_USER` to your user's UID and the host's 'docker' group GID (e.g., `"1000:999"`). Leave this blank if using macOS or Windows.


8. **Observability (Langfuse Local Stack):**
* Leave `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY` blank for now. You will generate these from your local dashboard during Step 5.



## Step 4: Network Ingress (Webhooks & Tunneling)

To receive commands from your mobile device while out and about, Telegram needs a way to send POST requests to your local machine. **Telegram webhooks strictly require a publicly accessible HTTPS endpoint.**

Your chosen networking solution must securely route external HTTPS traffic to your local Docker gateway service on port `8000`. You can achieve this using one of the following methods:

**Option A: Cloudflare Zero Trust Tunnel (Recommended for Persistent Daemons)**

1. Install `cloudflared` and create a Zero Trust tunnel via the Cloudflare Dashboard.
2. Route a public subdomain (e.g., `agent.yourdomain.com`) internally to your Docker service on `http://localhost:8000`.
3. Save the tunnel token to your `.env` as `CLOUDFLARE_TUNNEL_TOKEN`.

**Option B: Ngrok (Best for Quick Testing)**

1. Install [Ngrok](https://ngrok.com/) and authenticate your account.
2. Run `ngrok http 8000` to generate a temporary HTTPS URL that forwards to your local container. *(Note: Free Ngrok URLs change upon restart, requiring you to update your Telegram webhook each time).*

**Option C: Reverse Proxy (For Static IPs & Dedicated Servers)**

1. If your host has a static public IP, you can configure a reverse proxy like Nginx or Traefik.
2. Secure the endpoint with a Let's Encrypt SSL certificate and map the incoming traffic to the internal Docker port `8000`.

**Setting the Webhook:**
Once your HTTPS URL is active, register it with Telegram by making a GET request in your browser (replacing the placeholders):
`https://api.telegram.org/bot<YOUR_TOKEN>/setWebhook?url=https://<YOUR_HTTPS_URL>/webhook&secret_token=<YOUR_TELEGRAM_SECRET_TOKEN>`

## Step 5: Booting the Docker Stack

### Deployment Scope: Host Integration & Sandboxing

This stack is optimized for single-user execution on personal workstations or dedicated local hardware. To enable the agent to seamlessly test code and spin up ephemeral sibling environments, the `docker-compose.yml` utilizes two specific host integrations:

* **Docker Socket (`/var/run/docker.sock`):** Allows the gateway to communicate with the host's Docker daemon to dynamically spawn and destroy isolated sandbox containers.
* **Workspace Mounts (`${HOME}/git`):** Maps your local Git directory directly into the container. This ensures that when the agent writes or modifies code, the changes are natively synced and recognized by your host system.

**A Note on Cloud Deployments:** Because this architecture intentionally grants the agent deep integration with the host machine to perform its workspace duties, the default `docker-compose.yml` is strictly intended for private, trusted environments. If you intend to adapt this project for a shared cloud server or a public-facing host, it is highly recommended to implement stricter isolation boundaries (such as rootless Docker-in-Docker networks or dedicated virtual machines) prior to deployment.

### Initializing the Services

With your `.env` configured and your paths set up (refer to [`docs/02_configuration.md`](../docs/02_configuration.md) for `config.yaml` details), you are ready to boot the background services and build the sandbox.

```bash
# 1. Rebuild and gracefully restart the background daemon containers
docker compose up -d --build

# 2. Build the isolated runtime sandbox image so it is cached for the agent
docker build -t agent-sandbox -f Dockerfile.sandbox .
```

This sequence will cache the `agent-sandbox` image on your host and spin up the following detached containers:

* `redis` (LangGraph's state checkpointer).
* `postgres` & `clickhouse` (The telemetry databases).
* `minio` (S3-compatible blob storage).
* `langfuse` & `langfuse-worker` (The local observability backend).
* `gateway` (The FastAPI ingress and LangGraph orchestrator).

**Verification & Final Keys:**
Once the containers are running, navigate to `http://localhost:3000` on your host machine to access the local Langfuse UI. Create your initial admin account and set up a new project.

Navigate to **Settings > API Keys**, generate your host keys, and add them to your `.env` file as `LANGFUSE_SECRET_KEY` and `LANGFUSE_PUBLIC_KEY`. Finally, restart the gateway so it picks up the telemetry keys:

```bash
docker compose restart gateway
```

Your agent is now fully instrumented and actively listening for commands via Telegram!
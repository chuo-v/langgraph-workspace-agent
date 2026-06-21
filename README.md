# LangGraph Workspace Agent

[![CI](https://github.com/chuo-v/langgraph-workspace-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/chuo-v/langgraph-workspace-agent/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11+-blue.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

An autonomous, FastAPI-based orchestrator that securely manages local repositories and executes complex iterative coding tasks. Designed for a detached, mobile-first workflow, it handles the heavy lifting inside ephemeral Docker sandboxes and pings you on Telegram only when your final GitHub Pull Request is ready for review.

---

## 🎯 Design Goals & Core Capabilities

This agent is engineered to solve the security, cost-efficiency, and mobility constraints inherent to local-first repository engineering—challenges that traditional developer interfaces or transient chat wrappers often leave unaddressed.

* 📱 **Asynchronous Mobile Ingress:** Standard terminal agents require your primary machine to remain active and your session open. By utilizing background task queues over Telegram webhooks, long-running workflows can be safely triggered and monitored from your mobile device while on the go. The local daemon processes the execution graph independently and only alerts you upon completion.
* 💸 **Cost-Optimized Tiered Escalation:** Routine tasks shouldn't incur premium API costs. A probabilistic triage router at the ingress layer offloads context gathering and commit synthesis to low-cost or local base models. High-tier frontier execution engines are selectively engaged only for complex tasks or when explicitly requested via a `/frontier` flag.
* 🔒 **Infrastructure-Enforced Sandboxing:** To mitigate the risks of unconstrained code execution and prompt-injection, the agent operates within an immutable, infrastructure-level boundary. Using a Docker-out-of-Docker (DooD) engine, the ephemeral sandboxes only mount explicitly whitelisted paths, keeping your host system secure.
* 🧠 **Episodic Vector Memory:** Deep conversational context windows inherently suffer from degradation over long engineering sessions. This platform mitigates this via a dual-layer memory system: active thread states are checkpointed securely, while localized codebase constraints and historical "operational insights" are indexed via vector search for ahead-of-time semantic self-reminding.

---

## 🏗️ System Architecture

At a high level, the system operates using the following stack:
- **Orchestration:** LangGraph (Finite State Machine and Multi-Tier Router)
- **Ingress:** FastAPI + Cloudflare Zero Trust Tunnels
- **Memory Persistence:** Redis (Checkpointing) & ChromaDB (Vector Search)
- **Telemetry:** Langfuse (Local Postgres, Clickhouse, and MinIO Worker stack)
- **Execution Sandbox:** Ephemeral Docker-out-of-Docker containers

---

## 💻 System Requirements

To successfully run this agent, your environment must meet the following baseline requirements:

* **Containerization:** Docker and Docker Compose are strictly required to manage the orchestrator daemon and instantiate the Docker-out-of-Docker execution sandboxes.
* **Persistent Host Machine:** Because the agent relies on asynchronous webhooks for mobile ingress, it should be deployed on a machine capable of persistent, background operation (e.g., a dedicated local server or a headless Mac Studio). The host OS must be configured to prevent automatic sleeping or display lockouts that could interrupt active container lifecycles.
* **External Accounts:**
  * A Telegram Bot Token (for the UI/Ingress).
  * A GitHub Personal Access Token (for PR creation and repository management).
  * API keys for your preferred LLM providers.
  * A Cloudflare account (if utilizing Zero Trust Tunnels for secure webhook exposure).

---

## 📚 Documentation Index

Ready to get the daemon running? Head over to the documentation for step-by-step setup instructions, configuration details, and architecture deep-dives:

* **[01. Deployment Guide](docs/01_deployment_guide.md):** Step-by-step instructions for installing dependencies, cloning the repository, configuring API keys, and spinning up the Docker Compose stack.
* **[02. Configuration](docs/02_configuration.md):** How to tailor the agent to your repositories, whitelist paths, and adjust the triage router thresholds.
* **[03. Architecture Deep Dive](docs/03_architecture_deep_dive.md):** A technical breakdown of the LangGraph state machine, the dual-layer memory system, and the Docker-out-of-Docker sandbox.
* **[04. Usage and Workflows](docs/04_usage_and_workflows.md):** A user manual for interacting with the agent via Telegram, using the `/frontier` flag, and understanding the PR lifecycle.
* **[05. Development and Evaluation](docs/05_development_and_evaluation.md):** Instructions for running the local test suite, setting up the dev environment, and running the router evaluation pipelines.
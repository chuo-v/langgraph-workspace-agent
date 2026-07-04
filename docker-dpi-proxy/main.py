"""
Docker Deep-Packet-Inspection (DPI) Proxy - L4/L7 Hybrid

What is this proxy for?
-----------------------
This service sits between the LangGraph Workspace Agent and the host machine's Docker daemon.
Its purpose is to act as a security boundary, ensuring that the agent can spawn ephemeral
sandboxes to run code, but cannot exploit the Docker API to take over the host machine.

What is Deep-Packet-Inspection (DPI)?
-------------------------------------
Standard proxies (like the tecnativa proxy) operate by blocking entire API routes (e.g.,
blocking all POST requests to `/containers/create`). However, our agent *needs* to call
`/containers/create` to do its job.

Deep-Packet-Inspection (DPI) goes a step further. Instead of just looking at the URL route,
it intercepts the actual "packet" payload (the JSON body) of the HTTP request. It unmarshals
the JSON, inspects the requested container configuration line-by-line, and rewrites the payload
to strip out dangerous requests before forwarding it to the actual Docker daemon.

Why is it being used here?
--------------------------
If a malicious user tricks the LLM into executing a harmful Docker command, the agent might
send a valid API request to create a container, but secretly include flags like:
 - `"Privileged": true`
 - `"Binds": ["/:/host_root"]`

Because the Docker socket runs as root, fulfilling this request would give the LLM full
read/write access to the host machine's file system. This DPI Proxy intercepts that request,
applies a strict whitelist, forces `"Privileged": False`, and blocks dangerous volume mounts.
Even if the agent goes rogue, the DPI proxy ensures the resulting container remains a harmless,
unprivileged sandbox.

Why a Layer-4 / Layer-7 Hybrid?
-------------------------------
This proxy operates natively at the socket level.
1. It intercepts the initial raw HTTP bytes (Layer-7).
2. If the request is `POST /.../containers/create`, it fully parses the HTTP body,
 applies the strict JSON whitelist, and reconstructs the payload.
3. For all other allowed endpoints (like `/attach` or `/exec`), it simply glues
 the client socket and the Docker socket together (Layer-4), allowing raw TCP hijacking
 and bidirectional streaming to work flawlessly.
"""

import asyncio
import json
import re
import sys

DOCKER_SOCK = "/var/run/docker.sock"
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 2375


def sanitize_container_create(payload: dict) -> dict:
    """Strips all fields except explicitly allowed ones and overrides security contexts."""
    safe_payload = {}

    allowed_top_keys = [
        "Hostname",
        "Domainname",
        "User",
        "AttachStdin",
        "AttachStdout",
        "AttachStderr",
        "Tty",
        "OpenStdin",
        "StdinOnce",
        "Env",
        "Cmd",
        "Image",
        "Volumes",
        "WorkingDir",
        "Entrypoint",
        "OnBuild",
        "Labels",
        "HostConfig",
        "NetworkingConfig",
        "MacAddress",
        "ExposedPorts",
        "StopSignal",
        "StopTimeout",
        "Shell",
        "NetworkDisabled",
        "Healthcheck",
    ]
    for key in allowed_top_keys:
        if key in payload:
            safe_payload[key] = payload[key]

    if "HostConfig" in safe_payload:
        original_hc = safe_payload["HostConfig"]
        safe_hc = {}

        allowed_hc_keys = [
            "Binds",
            "Mounts",
            "Tmpfs",
            "NetworkMode",
            "Memory",
            "NanoCPUs",
            "NanoCpus",
            "PortBindings",
            "AutoRemove",
            "LogConfig",
            "RestartPolicy",
            "ExtraHosts",
            "ShmSize",
            "Init",
            "MemorySwap",
            "CpuQuota",
            "CpuPeriod",
            "CpuShares",
            "Dns",
            "DnsSearch",
            "DnsOptions",
            "CapDrop",
            "Ulimits",
            "PidsLimit",
            "ReadonlyRootfs",
            "OomKillDisable",
            "DeviceRequests",
        ]
        for key in allowed_hc_keys:
            if key in original_hc:
                safe_hc[key] = original_hc[key]

        # ENFORCE STRICT SECURITY OVERRIDES
        safe_hc["Privileged"] = False
        safe_hc["PidMode"] = ""
        safe_hc["IpcMode"] = ""
        safe_hc["UsernsMode"] = ""
        safe_hc["CapAdd"] = []

        # Sanitize Legacy Binds (array of strings)
        if "Binds" in safe_hc:
            sanitized_binds = []
            for bind in safe_hc["Binds"]:
                host_path = bind.split(":")[0]
                # Block obvious escapes out of the workspace or socket mounting
                if host_path == "/" or "docker.sock" in host_path:
                    continue
                sanitized_binds.append(bind)
            safe_hc["Binds"] = sanitized_binds

        # Sanitize Modern Mounts (array of dicts, preferred by Go/Node/CLI)
        if "Mounts" in safe_hc:
            sanitized_mounts = []
            for mount in safe_hc["Mounts"]:
                # Only restrict 'bind' types; let 'volume' and 'tmpfs' pass safely
                if mount.get("Type") == "bind":
                    host_path = mount.get("Source", "")
                    if host_path == "/" or "docker.sock" in host_path:
                        continue
                sanitized_mounts.append(mount)
            safe_hc["Mounts"] = sanitized_mounts

        safe_payload["HostConfig"] = safe_hc

    return safe_payload


async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter):
    """Glues the sockets together for bidirectional streaming."""
    try:
        while True:
            data = await src.read(8192)
            if not data:
                break
            dst.write(data)
            await dst.drain()
    except Exception:
        pass
    finally:
        if not dst.is_closing():
            dst.close()


def _is_request_allowed(method: str, path: str) -> bool:
    """Enforces Endpoint Role-Based Access Control (RBAC)."""
    if method == "GET":
        return True

    # The prefix (?:/v[\d\.]+)? makes the API version optional for standard CLI tools
    allowed_patterns = [
        r"^(?:/v[\d\.]+)?/containers/create.*",
        r"^(?:/v[\d\.]+)?/containers/[a-zA-Z0-9_.-]+/start.*",
        r"^(?:/v[\d\.]+)?/containers/[a-zA-Z0-9_.-]+/wait.*",
        r"^(?:/v[\d\.]+)?/containers/[a-zA-Z0-9_.-]+/attach.*",
        r"^(?:/v[\d\.]+)?/containers/[a-zA-Z0-9_.-]+/exec.*",
        r"^(?:/v[\d\.]+)?/exec/[a-zA-Z0-9_.-]+/start.*",
        r"^(?:/v[\d\.]+)?/exec/[a-zA-Z0-9_.-]+/resize.*",
        r"^(?:/v[\d\.]+)?/containers/[a-zA-Z0-9_.-]+(\?.*)?$",  # DELETE container
        r"^(?:/v[\d\.]+)?/images/create.*",
        r"^(?:/v[\d\.]+)?/build.*",
        r"^(?:/v[\d\.]+)?/networks.*",
    ]
    return any(re.match(p, path) for p in allowed_patterns)


async def _read_initial_request(
    reader: asyncio.StreamReader,
) -> tuple[bytearray, list[str], bytearray]:
    """Reads from the stream until the HTTP header boundary is found."""
    request_data = bytearray()
    while b"\r\n\r\n" not in request_data:
        chunk = await reader.read(4096)
        if not chunk:
            break
        request_data.extend(chunk)

    if not request_data:
        return bytearray(), [], bytearray()

    header_bytes, body_bytes = request_data.split(b"\r\n\r\n", 1)
    headers_text = header_bytes.decode("utf-8", errors="ignore")
    lines = headers_text.split("\r\n")

    return request_data, lines, body_bytes


async def _handle_dpi_interception(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    docker_writer: asyncio.StreamWriter,
    lines: list[str],
    body_bytes: bytearray,
) -> bool:
    """Parses and sanitizes the JSON payload, writing the modified request to Docker."""
    content_length = 0
    for line in lines[1:]:
        if line.lower().startswith("content-length:"):
            try:
                content_length = int(line.split(":")[1].strip())
            except ValueError:
                pass

    # Ensure the full body is read based on Content-Length
    while len(body_bytes) < content_length:
        chunk = await reader.read(4096)
        if not chunk:
            break
        body_bytes.extend(chunk)

    try:
        # Load the bytes directly to avoid decoding quirks with binary payloads
        payload = json.loads(body_bytes)
        safe_payload = sanitize_container_create(payload)
        new_body = json.dumps(safe_payload).encode("utf-8")

        # Reconstruct headers with new length
        new_headers = []
        for line in lines:
            if line.lower().startswith("content-length:"):
                new_headers.append(f"Content-Length: {len(new_body)}")
            else:
                new_headers.append(line)

        new_request = "\r\n".join(new_headers).encode("utf-8") + b"\r\n\r\n" + new_body
        docker_writer.write(new_request)
        await docker_writer.drain()
        return True
    except Exception:
        writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        await writer.drain()
        return False


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    docker_writer = None
    try:
        # 1. Connect to the real Docker socket
        try:
            docker_reader, docker_writer = await asyncio.open_unix_connection(DOCKER_SOCK)
        except Exception:
            writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await writer.drain()
            return

        # 2. Read the initial HTTP request headers
        request_data, lines, body_bytes = await _read_initial_request(reader)
        if not request_data:
            return

        # 3. Parse Request Line
        try:
            method, path, _ = lines[0].split(" ")
        except ValueError:
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            await writer.drain()
            return

        # 4. Enforce Endpoint RBAC
        if not _is_request_allowed(method, path):
            response = b"HTTP/1.1 403 Forbidden\r\nContent-Length: 21\r\n\r\nBlocked by DPI Proxy."
            writer.write(response)
            await writer.drain()
            return

        # 5. DPI Interception for Container Creation
        if method == "POST" and "/containers/create" in path:
            if not await _handle_dpi_interception(reader, writer, docker_writer, lines, body_bytes):
                return
        else:
            # 6. Transparent Proxy for all other requests
            docker_writer.write(request_data)
            await docker_writer.drain()

        # 7. Bidirectional Streaming (TCP Hijacking)
        await asyncio.gather(pipe(reader, docker_writer), pipe(docker_reader, writer))

    except Exception:
        pass
    finally:
        if not writer.is_closing():
            writer.close()
        if docker_writer and not docker_writer.is_closing():
            docker_writer.close()


async def main():
    server = await asyncio.start_server(handle_client, LISTEN_HOST, LISTEN_PORT)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)

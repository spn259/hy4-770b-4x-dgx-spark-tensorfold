#!/usr/bin/env python3
"""Lightweight OpenAI-compatible proxy for Tencent Hy4 on TensorFold.

Translates Hy4's native `<tool_calls:opensource><tool_call:opensource>NAME<arg_key:opensource>K</arg_key:opensource><arg_value:opensource>V</arg_value:opensource></tool_call:opensource></tool_calls:opensource>`
blocks into standard OpenAI `tool_calls` (both streaming SSE and non-streaming JSON).
"""
import http.server
import json
import os
import re
import socketserver
import threading
import urllib.request
import uuid

UPSTREAM = os.environ.get("HY4_UPSTREAM", "http://192.168.68.119:8895").rstrip("/")
LISTEN_HOST = os.environ.get("HY4_PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("HY4_PROXY_PORT", "8896"))
DEFAULT_TEMP = os.environ.get("HY4_DEFAULT_TEMP", "")  # optional override, e.g. "0.3"
THINKING_BUDGET = os.environ.get("HY4_THINKING_BUDGET", "")  # optional cap, e.g. "2048"
UPSTREAM_LOCK = threading.Lock()

OPEN_WRAP = "<tool_calls:opensource>"
CLOSE_WRAP = "</tool_calls:opensource>"
CALL_RE = re.compile(r"<tool_call:opensource>(.*?)</tool_call:opensource>", re.DOTALL)
KV_RE = re.compile(
    r"<arg_key:opensource>(.*?)</arg_key:opensource>\s*<arg_value:opensource>(.*?)</arg_value:opensource>",
    re.DOTALL,
)


def parse_hy4_tool_calls(block: str):
    """Parse one or more <tool_call:opensource>...</tool_call:opensource> elements into OpenAI tool_calls."""
    calls = []
    for m in CALL_RE.finditer(block):
        inner = m.group(1)
        first_kv = inner.find("<arg_key:opensource>")
        if first_kv == -1:
            name = inner.strip()
            args = {}
        else:
            name = inner[:first_kv].strip()
            args = {}
            for km in KV_RE.finditer(inner[first_kv:]):
                k = km.group(1).strip()
                raw_v = km.group(2)
                # Try JSON decode for numbers/bools/arrays/objects; fallback to raw string
                v_strip = raw_v.strip()
                if v_strip and v_strip[0] in '{["0123456789-tfn':
                    try:
                        val = json.loads(v_strip)
                    except Exception:
                        val = raw_v
                else:
                    val = raw_v
                args[k] = val
        if name:
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })
    return calls


def longest_prefix_suffix(buf: str, marker: str) -> int:
    """Return length of longest suffix of buf that is a prefix of marker (0..len(marker)-1)."""
    max_k = min(len(buf), len(marker) - 1)
    for k in range(max_k, 0, -1):
        if buf.endswith(marker[:k]):
            return k
    return 0


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _forward_simple(self, method: str):
        url = f"{UPSTREAM}{self.path}"
        req = urllib.request.Request(url, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except Exception as e:
            err = json.dumps({"error": {"message": str(e)}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)

    def do_GET(self):
        self._forward_simple("GET")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not self.path.startswith("/v1/chat/completions"):
            url = f"{UPSTREAM}{self.path}"
            req = urllib.request.Request(url, data=raw, method="POST", headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return

        payload = json.loads(raw)
        if DEFAULT_TEMP and "temperature" not in payload:
            payload["temperature"] = float(DEFAULT_TEMP)
        if THINKING_BUDGET and "thinking_budget" not in payload:
            payload["thinking_budget"] = int(THINKING_BUDGET)
        is_stream = bool(payload.get("stream", False))
        # Serialize /v1/chat/completions so single-stream Hy4 never rejects overlapping OMP requests
        with UPSTREAM_LOCK:
            self._handle_chat(payload, is_stream)

    def _handle_chat(self, payload, is_stream):
        out_raw = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{UPSTREAM}{self.path}",
            data=out_raw,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            err = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return

        if not is_stream:
            data = json.loads(resp.read())
            for ch in data.get("choices") or []:
                msg = ch.get("message") or {}
                txt = msg.get("content") or ""
                if OPEN_WRAP in txt or "<tool_call:opensource>" in txt:
                    pos = txt.find(OPEN_WRAP) if OPEN_WRAP in txt else txt.find("<tool_call:opensource>")
                    calls = parse_hy4_tool_calls(txt[pos:])
                    if calls:
                        msg["content"] = txt[:pos].strip() or None
                        msg["tool_calls"] = calls
                        ch["finish_reason"] = "tool_calls"
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Streaming SSE with incremental Hy4 <tool_calls:opensource> parser
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit_sse(obj):
            line = f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()
            self.wfile.write(line)
            self.wfile.flush()

        content_buf = ""
        in_tools = False
        tool_buf = ""
        emitted_tools = False
        last_meta = {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion.chunk", "created": 0, "model": "hy4"}

        def flush_tool_buf():
            nonlocal tool_buf, emitted_tools
            calls = parse_hy4_tool_calls(tool_buf)
            if not calls:
                return
            emitted_tools = True
            for idx, tc in enumerate(calls):
                emit_sse({
                    **last_meta,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": idx,
                                "id": tc["id"],
                                "type": "function",
                                "function": tc["function"],
                            }]
                        },
                        "finish_reason": None,
                    }],
                })
            tool_buf = ""

        try:
            for raw_line in resp:
                s = raw_line.decode("utf-8", errors="replace").strip()
                if not s:
                    continue
                if s == "data: [DONE]":
                    if in_tools and tool_buf:
                        flush_tool_buf()
                    elif content_buf:
                        emit_sse({**last_meta, "choices": [{"index": 0, "delta": {"content": content_buf}, "finish_reason": None}]})
                        content_buf = ""
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    break
                if not s.startswith("data: "):
                    continue
                chunk = json.loads(s[6:])
                for k in ("id", "object", "created", "model"):
                    if k in chunk:
                        last_meta[k] = chunk[k]
                choices = chunk.get("choices") or []
                if not choices:
                    emit_sse(chunk)
                    continue
                ch0 = choices[0]
                delta = ch0.get("delta") or {}
                piece = delta.get("content")
                finish = ch0.get("finish_reason")

                if piece is not None and piece != "":
                    if not in_tools:
                        content_buf += piece
                        # Check for either <tool_calls:opensource> or bare <tool_call:opensource>
                        idx_wrap = content_buf.find(OPEN_WRAP)
                        idx_bare = content_buf.find("<tool_call:opensource>")
                        cut = -1
                        if idx_wrap != -1 and (idx_bare == -1 or idx_wrap <= idx_bare):
                            cut = idx_wrap
                        elif idx_bare != -1:
                            cut = idx_bare
                        if cut != -1:
                            before = content_buf[:cut]
                            if before:
                                d2 = dict(delta)
                                d2["content"] = before
                                emit_sse({**chunk, "choices": [{**ch0, "delta": d2, "finish_reason": None}]})
                            tool_buf = content_buf[cut:]
                            content_buf = ""
                            in_tools = True
                        else:
                            hold = max(
                                longest_prefix_suffix(content_buf, OPEN_WRAP),
                                longest_prefix_suffix(content_buf, "<tool_call:opensource>"),
                            )
                            safe = content_buf[: len(content_buf) - hold] if hold else content_buf
                            content_buf = content_buf[len(safe):]
                            if safe:
                                d2 = dict(delta)
                                d2["content"] = safe
                                emit_sse({**chunk, "choices": [{**ch0, "delta": d2, "finish_reason": None}]})
                    else:
                        tool_buf += piece

                    if in_tools and CLOSE_WRAP in tool_buf:
                        flush_tool_buf()
                        in_tools = False
                else:
                    # Non-content delta (e.g. reasoning_content, role, or finish_reason)
                    if finish is not None:
                        if in_tools and tool_buf:
                            flush_tool_buf()
                            in_tools = False
                        elif content_buf:
                            emit_sse({**last_meta, "choices": [{"index": 0, "delta": {"content": content_buf}, "finish_reason": None}]})
                            content_buf = ""
                        if emitted_tools and finish == "stop":
                            ch0 = {**ch0, "finish_reason": "tool_calls"}
                            chunk = {**chunk, "choices": [ch0]}
                    emit_sse(chunk)
        finally:
            resp.close()


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    print(f"Hy4 OMP tool-call proxy listening on http://{LISTEN_HOST}:{LISTEN_PORT} -> {UPSTREAM}", flush=True)
    ThreadedServer((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()

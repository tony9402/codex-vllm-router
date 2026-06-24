from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI()

# =============================================================================
# Configuration
# =============================================================================

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "http://ip:port/v1").rstrip("/")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "600"))

# Backward-compatible env names.
LLM_KEY_HEADER = os.environ.get("LLM_KEY_HEADER", "Authorization")
RAW_LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()

# By default, do not forward Codex -> router credentials to upstream.
# Usually the router and upstream have separate auth domains.
FORWARD_INCOMING_LLM_KEY = os.environ.get("FORWARD_INCOMING_LLM_KEY", "0") not in {
    "0",
    "false",
    "False",
}

ENABLE_CODEX_TEXT_REPAIR = os.environ.get("ENABLE_CODEX_TEXT_REPAIR", "1") not in {
    "0",
    "false",
    "False",
}

INJECT_CODEX_TOOL_GUARD = os.environ.get("INJECT_CODEX_TOOL_GUARD", "1") not in {
    "0",
    "false",
    "False",
}

# pseudo-Codex <update_plan> is strip-only by default.
# Turning this on can cause unnecessary plan/tool loops with weaker models.
ENABLE_UPDATE_PLAN_TOOL_REPAIR = os.environ.get(
    "ENABLE_UPDATE_PLAN_TOOL_REPAIR",
    "0",
) not in {"0", "false", "False"}

COMBINE_REPAIRED_SHELL_COMMANDS = os.environ.get(
    "COMBINE_REPAIRED_SHELL_COMMANDS",
    "1",
) not in {"0", "false", "False"}

# Gemma/vLLM parser combinations are usually more stable with single tool calls.
FORCE_SINGLE_TOOL_CALL = os.environ.get("FORCE_SINGLE_TOOL_CALL", "1") not in {
    "0",
    "false",
    "False",
}

RESPONSE_STATE_TTL_SECONDS = int(os.environ.get("RESPONSE_STATE_TTL_SECONDS", "7200"))
RESPONSE_STATE_MAX_ITEMS = int(os.environ.get("RESPONSE_STATE_MAX_ITEMS", "512"))

# Empty output handling. This prevents Codex from appearing to hang with a blank
# assistant message when upstream returned neither text nor tool_calls, or when
# pseudo-Codex-only markup was stripped to empty.
EMPTY_RESPONSE_RETRY_COUNT = int(os.environ.get("EMPTY_RESPONSE_RETRY_COUNT", "1"))
EMIT_FALLBACK_ON_EMPTY = os.environ.get("EMIT_FALLBACK_ON_EMPTY", "1") not in {
    "0",
    "false",
    "False",
}
EMPTY_RESPONSE_RETRY_MESSAGE = os.environ.get(
    "EMPTY_RESPONSE_RETRY_MESSAGE",
    (
        "Your previous response contained no valid assistant text or structured tool call. "
        "Continue the task now. If work remains, call the appropriate function tool. "
        "If the task is complete, provide a concise final answer. Do not output Codex UI markup."
    ),
)

# Text-only progress/preamble handling. Weak tool-calling models sometimes say
# "I will inspect ..." without actually calling a tool. Returning that text makes
# Codex treat the turn as complete, so retry until the model either calls a tool
# or produces a real final answer.
ENABLE_STALL_TEXT_RETRY = os.environ.get("ENABLE_STALL_TEXT_RETRY", "1") not in {
    "0",
    "false",
    "False",
}
STALL_TEXT_RETRY_COUNT = int(os.environ.get("STALL_TEXT_RETRY_COUNT", "3"))
EMIT_FALLBACK_ON_STALL = os.environ.get("EMIT_FALLBACK_ON_STALL", "1") not in {
    "0",
    "false",
    "False",
}
STALL_TEXT_MAX_CHARS = int(os.environ.get("STALL_TEXT_MAX_CHARS", "2500"))
STALL_TEXT_RETRY_MESSAGE = os.environ.get(
    "STALL_TEXT_RETRY_MESSAGE",
    (
        "Your previous response was only a progress note or future-tense plan and did not "
        "contain a structured function tool call. Do not answer with another preamble. "
        "If any work remains, call exactly one appropriate function tool now. If the task "
        "is truly complete, provide a concrete final answer with findings. Do not output "
        "Codex UI markup."
    ),
)

# Conversation / adapter logging.
ENABLE_CONVERSATION_LOGGING = os.environ.get("ENABLE_CONVERSATION_LOGGING", "1") not in {
    "0",
    "false",
    "False",
}
LOG_DIR = Path(os.environ.get("LOG_DIR", "./codex-router-logs"))
LOG_FILE = Path(os.environ.get("LOG_FILE", str(LOG_DIR / "conversation.jsonl")))
LOG_MAX_BYTES = int(os.environ.get("LOG_MAX_BYTES", str(50 * 1024 * 1024)))
LOG_BACKUP_COUNT = int(os.environ.get("LOG_BACKUP_COUNT", "10"))
LOG_FULL_BODIES = os.environ.get("LOG_FULL_BODIES", "1") not in {
    "0",
    "false",
    "False",
}
LOG_HEADERS = os.environ.get("LOG_HEADERS", "0") not in {"0", "false", "False"}
LOG_SSE_EVENTS = os.environ.get("LOG_SSE_EVENTS", "0") not in {"0", "false", "False"}
LOG_MAX_CHARS = int(os.environ.get("LOG_MAX_CHARS", "200000"))
RECENT_LOG_MAX_ITEMS = int(os.environ.get("RECENT_LOG_MAX_ITEMS", "1000"))
ENABLE_DEBUG_ENDPOINTS = os.environ.get("ENABLE_DEBUG_ENDPOINTS", "0") not in {
    "0",
    "false",
    "False",
}

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
}

CODEX_TOOL_GUARD = """You are being used by OpenAI Codex through a protocol adapter.

Critical protocol rules:
- Do not print or imitate Codex UI markup.
- Do not output <update_plan>, <|channel>thought, "Shell", "$ command", "실행 완료", "출력 없음", or "성공" as plain text.
- If a tool is needed, call the provided function tool.
- If you need to run a shell command, call the shell/command execution tool instead of writing the command in text.
- Do not output or imitate planning markup such as <update_plan>. Continue with the actual next required action.
- Never claim that a command was executed unless the tool result is present in the conversation.
"""

# =============================================================================
# Logging helpers
# =============================================================================

REDACT_KEY_FRAGMENTS = (
    "authorization",
    "api_key",
    "apikey",
    "key",
    "token",
    "secret",
    "password",
    "cookie",
    "set-cookie",
)

RECENT_LOGS: deque[dict[str, Any]] = deque(maxlen=RECENT_LOG_MAX_ITEMS)
CONVERSATION_LOGGER = logging.getLogger("codex_vllm_router.conversation")
CONVERSATION_LOGGER.setLevel(logging.INFO)
CONVERSATION_LOGGER.propagate = False

if ENABLE_CONVERSATION_LOGGING and not CONVERSATION_LOGGER.handlers:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    _handler.setFormatter(logging.Formatter("%(message)s"))
    CONVERSATION_LOGGER.addHandler(_handler)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _format_llm_api_key_for_header(key: str, header_name: str) -> str:
    key = key.strip()
    if not key:
        return ""

    if header_name.lower() == "authorization":
        if key.lower().startswith("bearer "):
            return key
        return f"Bearer {key}"

    return key


LLM_API_KEY = _format_llm_api_key_for_header(RAW_LLM_API_KEY, LLM_KEY_HEADER)


def _truncate_str(value: str) -> str:
    if len(value) <= LOG_MAX_CHARS:
        return value
    return value[:LOG_MAX_CHARS] + f"... [truncated {len(value) - LOG_MAX_CHARS} chars]"


def _is_sensitive_key(key: Any) -> bool:
    key_lower = str(key).lower()
    return any(fragment in key_lower for fragment in REDACT_KEY_FRAGMENTS)


def _sanitize_for_log(value: Any, *, depth: int = 0) -> Any:
    if depth > 20:
        return "...[max_depth]"

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                sanitized[str(key)] = "[REDACTED]"
            else:
                sanitized[str(key)] = _sanitize_for_log(item, depth=depth + 1)
        return sanitized

    if isinstance(value, list):
        return [_sanitize_for_log(item, depth=depth + 1) for item in value]

    if isinstance(value, tuple):
        return [_sanitize_for_log(item, depth=depth + 1) for item in value]

    if isinstance(value, str):
        return _truncate_str(value)

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    return _truncate_str(str(value))


def _json_dumps_for_log(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _log_event(event: str, request_id: str | None = None, **fields: Any) -> None:
    if not ENABLE_CONVERSATION_LOGGING:
        return

    entry = {
        "ts": _utc_now_iso(),
        "event": event,
        "request_id": request_id,
        **fields,
    }
    sanitized = _sanitize_for_log(entry)

    try:
        CONVERSATION_LOGGER.info(_json_dumps_for_log(sanitized))
        RECENT_LOGS.append(sanitized)
    except Exception:
        # Logging must never break the proxy path.
        pass


def _headers_for_log(headers: Any) -> dict[str, Any]:
    if not LOG_HEADERS:
        return {}
    try:
        return _sanitize_for_log(dict(headers))
    except Exception:
        return {}


def _body_for_log(body: Any) -> Any:
    if not LOG_FULL_BODIES:
        return "[disabled: set LOG_FULL_BODIES=1 to log payload bodies]"
    return _sanitize_for_log(body)


def _summarize_responses_input(input_items: Any) -> dict[str, Any]:
    if isinstance(input_items, str):
        return {"kind": "string", "text_chars": len(input_items)}

    if not isinstance(input_items, list):
        return {"kind": type(input_items).__name__}

    counts: dict[str, int] = {}
    roles: dict[str, int] = {}
    tool_outputs = 0
    function_calls = 0

    for item in input_items:
        if not isinstance(item, dict):
            counts[type(item).__name__] = counts.get(type(item).__name__, 0) + 1
            continue
        item_type = str(item.get("type", "<missing>"))
        counts[item_type] = counts.get(item_type, 0) + 1
        role = item.get("role")
        if isinstance(role, str):
            roles[role] = roles.get(role, 0) + 1
        if item_type in {"function_call_output", "tool_result"}:
            tool_outputs += 1
        if item_type == "function_call":
            function_calls += 1

    return {
        "kind": "list",
        "items": len(input_items),
        "types": counts,
        "roles": roles,
        "function_calls": function_calls,
        "tool_outputs": tool_outputs,
    }


def _summarize_chat_messages(messages: Any) -> dict[str, Any]:
    if not isinstance(messages, list):
        return {"kind": type(messages).__name__}

    roles: dict[str, int] = {}
    tool_call_count = 0
    tool_result_count = 0
    total_content_chars = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "<missing>"))
        roles[role] = roles.get(role, 0) + 1
        content = message.get("content")
        if isinstance(content, str):
            total_content_chars += len(content)
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            tool_call_count += len(tool_calls)
        if role == "tool":
            tool_result_count += 1

    return {
        "messages": len(messages),
        "roles": roles,
        "tool_calls": tool_call_count,
        "tool_results": tool_result_count,
        "content_chars": total_content_chars,
    }


def _summarize_responses_output(output_items: Any) -> dict[str, Any]:
    if not isinstance(output_items, list):
        return {"kind": type(output_items).__name__}

    counts: dict[str, int] = {}
    function_names: list[str] = []
    output_text_chars = 0

    for item in output_items:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "<missing>"))
        counts[item_type] = counts.get(item_type, 0) + 1
        if item_type == "function_call" and isinstance(item.get("name"), str):
            function_names.append(item["name"])
        if item_type == "message":
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        output_text_chars += len(part["text"])

    return {
        "items": len(output_items),
        "types": counts,
        "function_names": function_names,
        "output_text_chars": output_text_chars,
    }


def _summarize_chat_response(chat_response: Any) -> dict[str, Any]:
    if not isinstance(chat_response, dict):
        return {"kind": type(chat_response).__name__}

    choices = chat_response.get("choices")
    summary: dict[str, Any] = {
        "id": chat_response.get("id"),
        "model": chat_response.get("model"),
        "choices": len(choices) if isinstance(choices, list) else 0,
        "usage": chat_response.get("usage"),
    }

    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        summary.update(
            {
                "finish_reason": first.get("finish_reason"),
                "content_chars": len(content) if isinstance(content, str) else 0,
                "tool_calls": len(tool_calls) if isinstance(tool_calls, list) else 0,
                "tool_names": [
                    call.get("function", {}).get("name")
                    for call in tool_calls
                    if isinstance(call, dict)
                ]
                if isinstance(tool_calls, list)
                else [],
            }
        )

    return summary


# =============================================================================
# Response state store
# =============================================================================


@dataclass
class ResponseState:
    created_at: float
    messages: list[dict[str, Any]]
    output_items: list[dict[str, Any]]
    model: str | None
    instructions: str | None
    request_id: str | None = None


RESPONSE_STATE: OrderedDict[str, ResponseState] = OrderedDict()


class MissingPreviousResponseState(RuntimeError):
    pass


def _prune_response_state() -> None:
    now = time.time()

    expired_ids = [
        response_id
        for response_id, state in RESPONSE_STATE.items()
        if now - state.created_at > RESPONSE_STATE_TTL_SECONDS
    ]

    for response_id in expired_ids:
        RESPONSE_STATE.pop(response_id, None)

    while len(RESPONSE_STATE) > RESPONSE_STATE_MAX_ITEMS:
        RESPONSE_STATE.popitem(last=False)


def _get_response_state(response_id: Any) -> ResponseState | None:
    if not isinstance(response_id, str) or not response_id:
        return None

    _prune_response_state()

    state = RESPONSE_STATE.get(response_id)
    if state is None:
        return None

    RESPONSE_STATE.move_to_end(response_id)
    return copy.deepcopy(state)


def _put_response_state(response_id: str, state: ResponseState) -> None:
    _prune_response_state()
    RESPONSE_STATE[response_id] = copy.deepcopy(state)
    RESPONSE_STATE.move_to_end(response_id)


# =============================================================================
# Header / payload normalization helpers
# =============================================================================


def _copy_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        k: v
        for k, v in headers.items()
        if k.lower() not in HOP_BY_HOP_HEADERS
    }


def _upstream_headers(request: Request) -> dict[str, str]:
    headers = {
        "content-type": "application/json",
    }

    incoming_key = request.headers.get(LLM_KEY_HEADER)

    if LLM_API_KEY:
        headers[LLM_KEY_HEADER] = LLM_API_KEY
    elif FORWARD_INCOMING_LLM_KEY and incoming_key:
        headers[LLM_KEY_HEADER] = incoming_key

    return headers


def _ensure_output_text_annotations(content: Any) -> None:
    if not isinstance(content, list):
        return

    for part in content:
        if not isinstance(part, dict):
            continue

        if part.get("type") == "output_text":
            part.setdefault("annotations", [])


def normalize_responses_body(body: dict[str, Any]) -> dict[str, Any]:
    """
    Make Codex multi-turn Responses payload easier for strict validators.
    """
    input_items = body.get("input")

    if not isinstance(input_items, list):
        return body

    for item in input_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "message" and item.get("role") == "assistant":
            item.setdefault("id", f"msg_{uuid.uuid4().hex}")
            item.setdefault("status", "completed")
            _ensure_output_text_annotations(item.get("content"))

        elif item_type == "reasoning":
            item.setdefault("id", f"rs_{uuid.uuid4().hex}")
            item.setdefault("summary", item.get("summary", []))

        elif item_type == "message":
            _ensure_output_text_annotations(item.get("content"))

    return body


def _extract_text_from_responses_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: list[str] = []

        for part in content:
            if isinstance(part, str):
                texts.append(part)
                continue

            if not isinstance(part, dict):
                continue

            part_type = part.get("type")

            if part_type in {
                "input_text",
                "output_text",
                "text",
                "summary_text",
            }:
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)

        return "\n".join(t for t in texts if t)

    return str(content)


def _stringify_tool_output(output: Any) -> str:
    if output is None:
        return ""

    if isinstance(output, str):
        return output

    return json.dumps(output, ensure_ascii=False)


# =============================================================================
# Responses -> Chat Completions conversion
# =============================================================================


def _responses_input_to_chat_messages(
    body: dict[str, Any],
    *,
    include_system_messages: bool = True,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    if include_system_messages and INJECT_CODEX_TOOL_GUARD:
        messages.append(
            {
                "role": "system",
                "content": CODEX_TOOL_GUARD,
            }
        )

    instructions = body.get("instructions")
    if include_system_messages and isinstance(instructions, str) and instructions.strip():
        messages.append(
            {
                "role": "system",
                "content": instructions,
            }
        )

    input_items = body.get("input")

    if isinstance(input_items, str):
        messages.append(
            {
                "role": "user",
                "content": input_items,
            }
        )
        return messages

    if not isinstance(input_items, list):
        return messages

    for item in input_items:
        if isinstance(item, str):
            messages.append(
                {
                    "role": "user",
                    "content": item,
                }
            )
            continue

        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        role = item.get("role")

        if item_type in (None, "message") and role in {
            "system",
            "developer",
            "user",
            "assistant",
        }:
            chat_role = "system" if role == "developer" else role
            content = _extract_text_from_responses_content(item.get("content"))

            if chat_role == "assistant" and not content.strip():
                continue

            messages.append(
                {
                    "role": chat_role,
                    "content": content,
                }
            )
            continue

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
            name = item.get("name") or ""
            arguments = item.get("arguments") or "{}"

            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)

            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }

            if messages and messages[-1].get("role") == "assistant":
                messages[-1].setdefault("tool_calls", [])
                messages[-1]["tool_calls"].append(tool_call)
                if not messages[-1].get("content"):
                    messages[-1]["content"] = None
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call],
                    }
                )

            continue

        if item_type in {"function_call_output", "tool_result"}:
            call_id = item.get("call_id") or item.get("tool_call_id")
            if not call_id:
                continue

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _stringify_tool_output(item.get("output")),
                }
            )
            continue

        # Do not re-inject reasoning into the upstream model.
        if item_type == "reasoning":
            continue

    return messages


def _input_contains_own_history(input_items: Any) -> bool:
    if not isinstance(input_items, list):
        return False

    for item in input_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "function_call":
            return True

        if item_type == "message" and item.get("role") == "assistant":
            return True

    return False


def _input_is_only_tool_outputs(input_items: Any) -> bool:
    if not isinstance(input_items, list) or not input_items:
        return False

    for item in input_items:
        if not isinstance(item, dict):
            return False

        if item.get("type") not in {"function_call_output", "tool_result"}:
            return False

    return True


def _responses_output_items_to_chat_messages(
    output_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_assistant_texts: list[str] = []
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_pending_assistant() -> None:
        nonlocal pending_assistant_texts, pending_tool_calls

        content = "\n".join(text for text in pending_assistant_texts if text.strip()).strip()

        if pending_tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": pending_tool_calls,
                }
            )
        elif content:
            messages.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )

        pending_assistant_texts = []
        pending_tool_calls = []

    for item in output_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "message":
            if pending_tool_calls:
                flush_pending_assistant()

            text = _extract_text_from_responses_content(item.get("content"))
            if text.strip():
                pending_assistant_texts.append(text)

        elif item_type == "function_call":
            name = item.get("name")
            arguments = item.get("arguments") or "{}"
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"

            if not isinstance(name, str) or not name:
                continue

            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)

            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )

    flush_pending_assistant()
    return messages


def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tools, list):
        return None

    chat_tools: list[dict[str, Any]] = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = tool.get("type")

        # Already Chat Completions format.
        if tool_type == "function" and isinstance(tool.get("function"), dict):
            chat_tools.append(tool)
            continue

        if tool_type != "function":
            # Built-in Responses tools such as web_search/file_search/computer_use are
            # not directly representable as vLLM Chat Completions function tools.
            continue

        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue

        function: dict[str, Any] = {
            "name": name,
            "description": tool.get("description") or "",
            "parameters": tool.get("parameters") or {
                "type": "object",
                "properties": {},
            },
        }

        if "strict" in tool:
            function["strict"] = tool["strict"]

        chat_tools.append(
            {
                "type": "function",
                "function": function,
            }
        )

    return chat_tools or None


def _responses_tool_choice_to_chat_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        return tool_choice

    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            name = tool_choice.get("name")
            if isinstance(name, str) and name:
                return {
                    "type": "function",
                    "function": {
                        "name": name,
                    },
                }

    return tool_choice


def _responses_body_to_chat_body(
    body: dict[str, Any],
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    previous_response_id = body.get("previous_response_id")
    previous_state = _get_response_state(previous_response_id)
    input_has_own_history = _input_contains_own_history(body.get("input"))

    _log_event(
        "state_lookup",
        request_id,
        previous_response_id=previous_response_id,
        state_hit=previous_state is not None,
        input_has_own_history=input_has_own_history,
        state_items=len(RESPONSE_STATE),
    )

    if previous_response_id and previous_state and not input_has_own_history:
        delta_messages = _responses_input_to_chat_messages(
            body,
            include_system_messages=False,
        )
        messages = previous_state.messages + delta_messages
        state_mode = "previous_response_id"

    elif previous_response_id and not previous_state and not input_has_own_history:
        raise MissingPreviousResponseState(
            f"unknown previous_response_id: {previous_response_id}"
        )

    else:
        messages = _responses_input_to_chat_messages(
            body,
            include_system_messages=True,
        )
        state_mode = "full_history_or_first_turn"

    tools = _responses_tools_to_chat_tools(body.get("tools"))

    chat_body: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        # The adapter synthesizes Responses SSE itself.
        "stream": False,
    }

    if body.get("max_output_tokens") is not None:
        chat_body["max_tokens"] = body["max_output_tokens"]
    elif body.get("max_tokens") is not None:
        chat_body["max_tokens"] = body["max_tokens"]

    for key in (
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stop",
    ):
        if key in body and body[key] is not None:
            chat_body[key] = body[key]

    if tools:
        chat_body["tools"] = tools

        tool_choice = _responses_tool_choice_to_chat_tool_choice(body.get("tool_choice"))

        # After tool outputs, forcing required can lead to endless tool calls.
        if tool_choice == "required" and _input_is_only_tool_outputs(body.get("input")):
            chat_body["tool_choice"] = "auto"
        elif tool_choice is not None:
            chat_body["tool_choice"] = tool_choice
        else:
            chat_body["tool_choice"] = "auto"

        if FORCE_SINGLE_TOOL_CALL:
            chat_body["parallel_tool_calls"] = False
        else:
            chat_body["parallel_tool_calls"] = bool(body.get("parallel_tool_calls", False))

    _log_event(
        "chat_request_built",
        request_id,
        state_mode=state_mode,
        summary=_summarize_chat_messages(messages),
        tool_count=len(tools) if tools else 0,
        tool_choice=chat_body.get("tool_choice"),
        parallel_tool_calls=chat_body.get("parallel_tool_calls"),
        body=_body_for_log(chat_body),
    )

    return chat_body


# =============================================================================
# Pseudo-Codex text repair
# =============================================================================


def _iter_response_function_tools(original_body: dict[str, Any]) -> list[dict[str, Any]]:
    tools = original_body.get("tools")
    if not isinstance(tools, list):
        return []

    result: list[dict[str, Any]] = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") == "function" and isinstance(tool.get("name"), str):
            result.append(tool)
            continue

        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            if isinstance(fn.get("name"), str):
                result.append(
                    {
                        "type": "function",
                        "name": fn.get("name"),
                        "description": fn.get("description") or "",
                        "parameters": fn.get("parameters") or {},
                    }
                )

    return result


def _find_tool_by_keywords(
    original_body: dict[str, Any],
    name_keywords: tuple[str, ...],
    description_keywords: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    for tool in _iter_response_function_tools(original_body):
        name = str(tool.get("name") or "").lower()
        desc = str(tool.get("description") or "").lower()

        if any(keyword in name for keyword in name_keywords):
            return tool

        if description_keywords and any(keyword in desc for keyword in description_keywords):
            return tool

    return None


def _tool_parameters(tool: dict[str, Any]) -> dict[str, Any]:
    params = tool.get("parameters")

    if isinstance(params, dict):
        return params

    fn = tool.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("parameters"), dict):
        return fn["parameters"]

    return {}


def _json_schema_value_for_command(prop_schema: Any, command: str) -> Any:
    if not isinstance(prop_schema, dict):
        return command

    prop_type = prop_schema.get("type")

    if prop_type == "array":
        return ["bash", "-lc", command]

    if prop_type == "object":
        return {
            "cmd": command,
        }

    return command


def _arguments_for_single_text_tool(tool: dict[str, Any], value: str) -> str:
    params = _tool_parameters(tool)
    props = params.get("properties") if isinstance(params, dict) else None
    required = params.get("required") if isinstance(params, dict) else None

    if not isinstance(props, dict):
        return json.dumps({"cmd": value}, ensure_ascii=False)

    preferred_keys = (
        "cmd",
        "command",
        "script",
        "input",
        "code",
        "query",
        "args",
    )

    for key in preferred_keys:
        if key in props:
            return json.dumps(
                {
                    key: _json_schema_value_for_command(props.get(key), value),
                },
                ensure_ascii=False,
            )

    if isinstance(required, list) and required:
        first_required = required[0]
        if isinstance(first_required, str):
            return json.dumps(
                {
                    first_required: _json_schema_value_for_command(
                        props.get(first_required),
                        value,
                    )
                },
                ensure_ascii=False,
            )

    if props:
        first_key = next(iter(props.keys()))
        return json.dumps(
            {
                first_key: _json_schema_value_for_command(props.get(first_key), value),
            },
            ensure_ascii=False,
        )

    return json.dumps({"cmd": value}, ensure_ascii=False)


def _make_function_call_item(
    name: str,
    arguments: str | dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)

    return {
        "id": f"fc_{uuid.uuid4().hex}",
        "type": "function_call",
        "status": "completed",
        "call_id": f"call_{uuid.uuid4().hex}",
        "name": name,
        "arguments": arguments,
    }


def _strip_codex_pseudo_markup(text: str) -> str:
    cleaned = text

    cleaned = re.sub(
        r"<update_plan>\s*.*?\s*</update_plan>",
        "",
        cleaned,
        flags=re.DOTALL,
    )

    cleaned = re.sub(
        r"<\|channel\>.*?(?:<channel\|>|$)",
        "",
        cleaned,
        flags=re.DOTALL,
    )

    cleaned = re.sub(
        r"(?ms)^\s*Shell\s*\n\$.*?(?:\n\s*(?:출력 없음|성공)\s*)+",
        "",
        cleaned,
    )

    cleaned = re.sub(
        r"(?m)^\s*(?:출력 없음|성공|실행 완료)\s*$",
        "",
        cleaned,
    )

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_json_object_maybe(text: str) -> dict[str, Any] | None:
    text = text.strip()

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value
    except Exception:
        return None

    return None


def _extract_update_plan_calls(
    text: str,
    original_body: dict[str, Any],
) -> list[dict[str, Any]]:
    update_plan_tool = _find_tool_by_keywords(
        original_body,
        name_keywords=("update_plan", "plan"),
        description_keywords=("plan", "planning"),
    )

    if not update_plan_tool:
        return []

    tool_name = update_plan_tool.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return []

    calls: list[dict[str, Any]] = []

    for match in re.finditer(
        r"<update_plan>\s*(.*?)\s*</update_plan>",
        text,
        flags=re.DOTALL,
    ):
        inner = match.group(1).strip()
        parsed = _extract_json_object_maybe(inner)

        if parsed is not None:
            arguments = parsed
        else:
            arguments = {"plan": inner}

        calls.append(_make_function_call_item(tool_name, arguments))

    return calls


def _normalize_repaired_shell_command(command: str) -> str:
    command = command.strip()

    # Remove pseudo Korean execution markers that may have been attached to the
    # heredoc delimiter or to a one-line command.
    command = re.sub(
        r"(?m)^([A-Za-z0-9_./-]+)\s+실행 완료\s*$",
        r"\1",
        command,
    )

    if "\n" not in command:
        command = re.sub(r"\s+실행 완료\s*$", "", command)

    command = re.sub(r"(?m)^\s*(?:출력 없음|성공)\s*$", "", command)
    command = re.sub(r"\n{3,}", "\n\n", command)

    return command.strip()


def _extract_shell_commands_from_transcript(text: str) -> list[str]:
    commands: list[str] = []

    # ```bash ... ``` form.
    for match in re.finditer(
        r"(?ms)```(?:bash|sh|shell)\s*\n(.*?)\n```",
        text,
    ):
        command = _normalize_repaired_shell_command(match.group(1))
        if command:
            commands.append(command)

    # Shell\n$... form.
    for match in re.finditer(
        r"(?ms)^\s*Shell\s*\n\$([^\n]+(?:\n(?!\s*(?:출력 없음|성공|Shell|\$|<\|channel\>)).*)*)",
        text,
    ):
        command = _normalize_repaired_shell_command(match.group(1))
        if command:
            commands.append(command)

    # heredoc + 실행 완료.
    for match in re.finditer(
        r"(?ms)^((?:cat|tee)\s+.*?<<\s*['\"]?EOF['\"]?.*?^EOF)\s*실행 완료",
        text,
    ):
        command = _normalize_repaired_shell_command(match.group(1))
        if command:
            commands.append(command)

    # single command + 실행 완료.
    command_prefixes = (
        "mkdir",
        "cat",
        "tee",
        "python",
        "python3",
        "pytest",
        "touch",
        "ls",
        "grep",
        "sed",
        "awk",
        "find",
        "git",
        "pip",
        "uvicorn",
        "rm",
        "cp",
        "mv",
        "chmod",
        "apply_patch",
        "echo",
        "printf",
    )
    prefix_pattern = "|".join(re.escape(prefix) for prefix in command_prefixes)

    for match in re.finditer(
        rf"(?m)^\s*(({prefix_pattern})\b[^\n]*)\s+실행 완료\s*$",
        text,
    ):
        command = _normalize_repaired_shell_command(match.group(1))
        if command:
            commands.append(command)

    deduped: list[str] = []
    seen: set[str] = set()

    for command in commands:
        if command not in seen:
            deduped.append(command)
            seen.add(command)

    return deduped


def _extract_shell_calls(
    text: str,
    original_body: dict[str, Any],
) -> list[dict[str, Any]]:
    shell_tool = _find_tool_by_keywords(
        original_body,
        name_keywords=(
            "shell",
            "bash",
            "terminal",
            "command",
            "exec",
            "run",
        ),
        description_keywords=(
            "shell",
            "bash",
            "terminal",
            "command",
            "execute",
        ),
    )

    if not shell_tool:
        return []

    tool_name = shell_tool.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        return []

    commands = _extract_shell_commands_from_transcript(text)
    if not commands:
        return []

    if COMBINE_REPAIRED_SHELL_COMMANDS:
        command = commands[0] if len(commands) == 1 else "set -e\n\n" + "\n\n".join(commands)
        arguments = _arguments_for_single_text_tool(shell_tool, command)
        return [_make_function_call_item(tool_name, arguments)]

    calls: list[dict[str, Any]] = []
    for command in commands:
        arguments = _arguments_for_single_text_tool(shell_tool, command)
        calls.append(_make_function_call_item(tool_name, arguments))

    return calls


def _repair_codex_pseudo_tool_output(
    content: str,
    original_body: dict[str, Any],
    *,
    request_id: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if not ENABLE_CODEX_TEXT_REPAIR or not isinstance(content, str) or not content.strip():
        return [], content or ""

    calls: list[dict[str, Any]] = []

    if ENABLE_UPDATE_PLAN_TOOL_REPAIR:
        calls.extend(_extract_update_plan_calls(content, original_body))

    shell_calls = _extract_shell_calls(content, original_body)
    calls.extend(shell_calls)

    cleaned_text = _strip_codex_pseudo_markup(content)

    if calls or cleaned_text != content.strip():
        _log_event(
            "codex_text_repair",
            request_id,
            input_chars=len(content),
            output_chars=len(cleaned_text),
            repaired_call_count=len(calls),
            repaired_function_names=[call.get("name") for call in calls],
            update_plan_tool_repair=ENABLE_UPDATE_PLAN_TOOL_REPAIR,
            shell_repaired=bool(shell_calls),
            original_text=_body_for_log(content),
            cleaned_text=_body_for_log(cleaned_text),
        )

    if calls:
        return calls, ""

    return [], cleaned_text


# =============================================================================
# Chat Completions -> Responses conversion
# =============================================================================


def _chat_usage_to_responses_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None

    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": 0,
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": 0,
        },
        "total_tokens": total_tokens,
    }


def _make_message_output_item(text: str) -> dict[str, Any]:
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }


def _clean_visible_assistant_text(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    return _strip_codex_pseudo_markup(text).strip()


def _chat_response_to_responses_body(
    chat: dict[str, Any],
    original_body: dict[str, Any],
    *,
    request_id: str | None = None,
    actual_parallel_tool_calls: bool | None = None,
) -> dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex}"
    created_at = int(time.time())
    model = chat.get("model") or original_body.get("model")

    choices = chat.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}

    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []

    output_items: list[dict[str, Any]] = []

    if isinstance(tool_calls, list) and tool_calls:
        preamble = _clean_visible_assistant_text(content)
        if preamble:
            output_items.append(_make_message_output_item(preamble))

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            function = tool_call.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments") or "{}"

            if not isinstance(name, str) or not name:
                continue

            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)

            call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex}"

            output_items.append(
                {
                    "id": f"fc_{uuid.uuid4().hex}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                }
            )

        # Content, if any, was emitted as a preamble item before the tool call.
        content = ""

    elif isinstance(content, str) and content.strip():
        repaired_calls, repaired_text = _repair_codex_pseudo_tool_output(
            content,
            original_body,
            request_id=request_id,
        )

        if repaired_calls:
            output_items.extend(repaired_calls)
            content = ""
        else:
            content = repaired_text

    if isinstance(content, str) and content.strip():
        output_items.append(_make_message_output_item(content))

    if not output_items:
        output_items.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                    }
                ],
            }
        )

    output_text = "".join(
        part.get("text", "")
        for item in output_items
        if item.get("type") == "message"
        for part in item.get("content", [])
        if isinstance(part, dict) and part.get("type") == "output_text"
    )

    response_body: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": original_body.get("instructions"),
        "max_output_tokens": original_body.get("max_output_tokens"),
        "model": model,
        "output": output_items,
        "output_text": output_text,
        "parallel_tool_calls": (
            actual_parallel_tool_calls
            if actual_parallel_tool_calls is not None
            else bool(original_body.get("parallel_tool_calls", False))
        ),
        "previous_response_id": original_body.get("previous_response_id"),
        "reasoning": original_body.get("reasoning"),
        "store": original_body.get("store", False),
        "temperature": original_body.get("temperature"),
        "text": original_body.get("text"),
        "tool_choice": original_body.get("tool_choice", "auto"),
        "tools": original_body.get("tools", []),
        "top_p": original_body.get("top_p"),
        "truncation": original_body.get("truncation", "disabled"),
        "usage": _chat_usage_to_responses_usage(chat.get("usage")),
        "user": original_body.get("user"),
        "metadata": original_body.get("metadata"),
    }

    return response_body


def _responses_body_is_empty_assistant_message(response_body: dict[str, Any]) -> bool:
    output = response_body.get("output")
    if not isinstance(output, list) or len(output) != 1:
        return False

    item = output[0]
    if not isinstance(item, dict) or item.get("type") != "message":
        return False

    content = item.get("content")
    if not isinstance(content, list):
        return True

    text = "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict)
    )
    return not text.strip()


def _responses_body_has_function_call(response_body: dict[str, Any]) -> bool:
    output = response_body.get("output")
    if not isinstance(output, list):
        return False
    return any(isinstance(item, dict) and item.get("type") == "function_call" for item in output)


def _responses_output_text(response_body: dict[str, Any]) -> str:
    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    output = response_body.get("output")
    if not isinstance(output, list):
        return ""

    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        text = _extract_text_from_responses_content(item.get("content"))
        if text:
            texts.append(text)

    return "\n".join(texts).strip()


def _has_function_tools(body: dict[str, Any]) -> bool:
    return bool(_responses_tools_to_chat_tools(body.get("tools")))


def _looks_like_agentic_stall_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > STALL_TEXT_MAX_CHARS:
        return False

    lower = stripped.lower()

    english_markers = (
        "i will ",
        "i'll ",
        "i’m going to",
        "i'm going to",
        "i am going to",
        "let me ",
        "i need to ",
        "i should ",
        "i’ll start",
        "i'll start",
        "i will now",
        "now i will",
        "next, i",
        "first, i",
        "start by",
        "begin by",
        "checking ",
        "inspecting ",
        "reviewing ",
    )
    if any(marker in lower for marker in english_markers):
        return True

    korean_markers = (
        "먼저",
        "이제",
        "다음으로",
        "살펴보겠습니다",
        "확인하겠습니다",
        "검토하겠습니다",
        "분석하겠습니다",
        "진행하겠습니다",
        "시작하겠습니다",
        "수정하겠습니다",
        "작성하겠습니다",
        "생성하겠습니다",
        "실행하겠습니다",
        "읽어보겠습니다",
        "찾아보겠습니다",
        "체크하겠습니다",
        "보겠습니다.",
        "보겠습니다\n",
    )
    if any(marker in stripped for marker in korean_markers):
        return True

    # Pseudo-plan text without actual tool calls is also a stall for Codex.
    if re.search(r"(?im)^\s*(?:plan|계획|검토 대상|우선순위)\s*[:：]", stripped):
        return True

    return False


def _responses_body_is_agentic_stall_text(
    response_body: dict[str, Any],
    original_body: dict[str, Any],
) -> bool:
    if not ENABLE_STALL_TEXT_RETRY:
        return False

    if _responses_body_has_function_call(response_body):
        return False

    if not _has_function_tools(original_body):
        return False

    text = _responses_output_text(response_body)
    return _looks_like_agentic_stall_text(text)


def _make_stall_fallback_response(
    original_body: dict[str, Any],
    *,
    request_id: str,
    stalled_text: str,
    actual_parallel_tool_calls: bool | None = None,
) -> dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex}"
    preview = stalled_text.strip().replace("\n", " ")[:300]
    text = (
        "[router] Upstream model repeatedly returned progress/planning text without a "
        "structured tool call, so Codex would stop here. "
        f"request_id={request_id}. Check {LOG_FILE}. "
        f"Last stalled text preview: {preview}"
    )

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": original_body.get("instructions"),
        "max_output_tokens": original_body.get("max_output_tokens"),
        "model": original_body.get("model"),
        "output": [_make_message_output_item(text)],
        "output_text": text,
        "parallel_tool_calls": (
            actual_parallel_tool_calls
            if actual_parallel_tool_calls is not None
            else bool(original_body.get("parallel_tool_calls", False))
        ),
        "previous_response_id": original_body.get("previous_response_id"),
        "reasoning": original_body.get("reasoning"),
        "store": original_body.get("store", False),
        "temperature": original_body.get("temperature"),
        "text": original_body.get("text"),
        "tool_choice": original_body.get("tool_choice", "auto"),
        "tools": original_body.get("tools", []),
        "top_p": original_body.get("top_p"),
        "truncation": original_body.get("truncation", "disabled"),
        "usage": None,
        "user": original_body.get("user"),
        "metadata": original_body.get("metadata"),
    }


def _make_empty_fallback_response(
    original_body: dict[str, Any],
    *,
    request_id: str,
    actual_parallel_tool_calls: bool | None = None,
) -> dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex}"
    text = (
        f"[router] Upstream model returned no valid assistant text or structured tool call "
        f"after retry. Check {LOG_FILE} for request_id={request_id}."
    )

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": original_body.get("instructions"),
        "max_output_tokens": original_body.get("max_output_tokens"),
        "model": original_body.get("model"),
        "output": [
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": text,
        "parallel_tool_calls": (
            actual_parallel_tool_calls
            if actual_parallel_tool_calls is not None
            else bool(original_body.get("parallel_tool_calls", False))
        ),
        "previous_response_id": original_body.get("previous_response_id"),
        "reasoning": original_body.get("reasoning"),
        "store": original_body.get("store", False),
        "temperature": original_body.get("temperature"),
        "text": original_body.get("text"),
        "tool_choice": original_body.get("tool_choice", "auto"),
        "tools": original_body.get("tools", []),
        "top_p": original_body.get("top_p"),
        "truncation": original_body.get("truncation", "disabled"),
        "usage": None,
        "user": original_body.get("user"),
        "metadata": original_body.get("metadata"),
    }


# =============================================================================
# SSE synthesis
# =============================================================================


def _sse_event(event_name: str, payload: dict[str, Any]) -> bytes:
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    ).encode("utf-8")


async def _stream_responses_sse_from_response(
    response_body: dict[str, Any],
    *,
    request_id: str | None = None,
):
    seq = 0

    def event(event_name: str, payload: dict[str, Any]) -> bytes:
        nonlocal seq
        payload = {
            "type": event_name,
            "sequence_number": seq,
            **payload,
        }
        seq += 1

        if LOG_SSE_EVENTS:
            _log_event(
                "sse_event",
                request_id,
                event_name=event_name,
                payload=_body_for_log(payload),
            )

        return _sse_event(event_name, payload)

    _log_event(
        "sse_stream_start",
        request_id,
        response_id=response_body.get("id"),
        output_summary=_summarize_responses_output(response_body.get("output")),
    )

    created_response = {
        **response_body,
        "status": "in_progress",
        "output": [],
    }

    yield event(
        "response.created",
        {
            "response": created_response,
        },
    )

    for output_index, item in enumerate(response_body.get("output", [])):
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "message":
            yield event(
                "response.output_item.added",
                {
                    "response_id": response_body["id"],
                    "output_index": output_index,
                    "item": {
                        **item,
                        "content": [],
                    },
                },
            )

            content = item.get("content") or []

            for content_index, part in enumerate(content):
                if not isinstance(part, dict):
                    continue

                if part.get("type") != "output_text":
                    continue

                text = part.get("text") or ""

                empty_part = {
                    "type": "output_text",
                    "text": "",
                    "annotations": [],
                }

                yield event(
                    "response.content_part.added",
                    {
                        "response_id": response_body["id"],
                        "item_id": item["id"],
                        "output_index": output_index,
                        "content_index": content_index,
                        "part": empty_part,
                    },
                )

                if text:
                    yield event(
                        "response.output_text.delta",
                        {
                            "response_id": response_body["id"],
                            "item_id": item["id"],
                            "output_index": output_index,
                            "content_index": content_index,
                            "delta": text,
                        },
                    )

                yield event(
                    "response.output_text.done",
                    {
                        "response_id": response_body["id"],
                        "item_id": item["id"],
                        "output_index": output_index,
                        "content_index": content_index,
                        "text": text,
                    },
                )

                yield event(
                    "response.content_part.done",
                    {
                        "response_id": response_body["id"],
                        "item_id": item["id"],
                        "output_index": output_index,
                        "content_index": content_index,
                        "part": part,
                    },
                )

            yield event(
                "response.output_item.done",
                {
                    "response_id": response_body["id"],
                    "output_index": output_index,
                    "item": item,
                },
            )

            continue

        if item_type == "function_call":
            empty_item = {
                **item,
                "arguments": "",
                "status": "in_progress",
            }

            yield event(
                "response.output_item.added",
                {
                    "response_id": response_body["id"],
                    "output_index": output_index,
                    "item": empty_item,
                },
            )

            arguments = item.get("arguments") or ""

            if arguments:
                yield event(
                    "response.function_call_arguments.delta",
                    {
                        "response_id": response_body["id"],
                        "item_id": item["id"],
                        "output_index": output_index,
                        "delta": arguments,
                    },
                )

            yield event(
                "response.function_call_arguments.done",
                {
                    "response_id": response_body["id"],
                    "item_id": item["id"],
                    "output_index": output_index,
                    "arguments": arguments,
                },
            )

            yield event(
                "response.output_item.done",
                {
                    "response_id": response_body["id"],
                    "output_index": output_index,
                    "item": item,
                },
            )

    yield event(
        "response.completed",
        {
            "response": response_body,
        },
    )

    yield b"data: [DONE]\n\n"

    _log_event(
        "sse_stream_end",
        request_id,
        response_id=response_body.get("id"),
        sequence_count=seq,
    )


# =============================================================================
# Upstream call / Responses handler
# =============================================================================


async def _post_chat_completions(
    *,
    client: httpx.AsyncClient,
    request: Request,
    request_id: str,
    chat_body: dict[str, Any],
    attempt: int,
) -> httpx.Response:
    upstream_url = f"{UPSTREAM_BASE_URL}/chat/completions"
    start = time.perf_counter()

    _log_event(
        "upstream_chat_request_start",
        request_id,
        attempt=attempt,
        upstream_url=upstream_url,
        chat_summary=_summarize_chat_messages(chat_body.get("messages")),
        body=_body_for_log(chat_body),
    )

    upstream = await client.post(
        upstream_url,
        headers=_upstream_headers(request),
        params=dict(request.query_params),
        json=chat_body,
    )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    _log_event(
        "upstream_chat_response_received",
        request_id,
        attempt=attempt,
        status_code=upstream.status_code,
        elapsed_ms=elapsed_ms,
        headers=_headers_for_log(upstream.headers),
        body=_body_for_log(_safe_response_json_or_text(upstream)),
    )

    return upstream


def _safe_response_json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return response.text


async def handle_responses_via_chat_completions(
    body: dict[str, Any],
    request: Request,
    *,
    request_id: str,
):
    raw_body = copy.deepcopy(body)
    body = normalize_responses_body(body)

    client_wants_stream = body.get("stream") is True

    _log_event(
        "responses_request_received",
        request_id,
        model=body.get("model"),
        previous_response_id=body.get("previous_response_id"),
        stream=client_wants_stream,
        tool_choice=body.get("tool_choice"),
        input_summary=_summarize_responses_input(body.get("input")),
        raw_body=_body_for_log(raw_body),
        normalized_body=_body_for_log(body),
    )

    try:
        chat_body = _responses_body_to_chat_body(body, request_id=request_id)
    except MissingPreviousResponseState as exc:
        _log_event(
            "missing_previous_response_state",
            request_id,
            error=str(exc),
            previous_response_id=body.get("previous_response_id"),
            input_summary=_summarize_responses_input(body.get("input")),
            state_items=len(RESPONSE_STATE),
        )
        return JSONResponse(
            {
                "error": {
                    "message": str(exc),
                    "type": "missing_previous_response_state",
                    "hint": (
                        "The router received previous_response_id but no stored state. "
                        "Use one uvicorn worker, increase RESPONSE_STATE_TTL_SECONDS, "
                        "or make Codex send full input history."
                    ),
                }
            },
            status_code=409,
            headers={"x-router-request-id": request_id},
        )

    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        upstream = await _post_chat_completions(
            client=client,
            request=request,
            request_id=request_id,
            chat_body=chat_body,
            attempt=0,
        )

        if upstream.status_code < 200 or upstream.status_code >= 300:
            _log_event(
                "upstream_chat_error_returned",
                request_id,
                status_code=upstream.status_code,
                body=_body_for_log(_safe_response_json_or_text(upstream)),
            )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers={
                    **_copy_response_headers(upstream.headers),
                    "x-router-request-id": request_id,
                },
                media_type=upstream.headers.get("content-type"),
            )

        try:
            chat_response = upstream.json()
        except Exception as exc:
            _log_event(
                "upstream_non_json_error",
                request_id,
                error=str(exc),
                upstream_text=upstream.text,
            )
            return JSONResponse(
                {
                    "error": f"upstream returned non-json response: {exc}",
                    "upstream_text": upstream.text,
                },
                status_code=502,
                headers={"x-router-request-id": request_id},
            )

        responses_body = _chat_response_to_responses_body(
            chat_response,
            body,
            request_id=request_id,
            actual_parallel_tool_calls=chat_body.get("parallel_tool_calls"),
        )

        _log_event(
            "responses_body_built",
            request_id,
            attempt=0,
            chat_summary=_summarize_chat_response(chat_response),
            output_summary=_summarize_responses_output(responses_body.get("output")),
            body=_body_for_log(responses_body),
        )

        retry_attempt = 0
        empty_retry_count = 0
        stall_retry_count = 0
        last_retry_reason: str | None = None

        while True:
            retry_reason: str | None = None
            retry_message: str | None = None
            retry_event_prefix: str | None = None

            if (
                _responses_body_is_empty_assistant_message(responses_body)
                and empty_retry_count < EMPTY_RESPONSE_RETRY_COUNT
            ):
                empty_retry_count += 1
                retry_reason = "empty_response"
                retry_message = EMPTY_RESPONSE_RETRY_MESSAGE
                retry_event_prefix = "empty_response_retry"

            elif (
                _responses_body_is_agentic_stall_text(responses_body, body)
                and stall_retry_count < STALL_TEXT_RETRY_COUNT
            ):
                stall_retry_count += 1
                retry_reason = "agentic_stall_text"
                retry_message = STALL_TEXT_RETRY_MESSAGE
                retry_event_prefix = "agentic_stall_retry"

            if retry_reason is None or retry_message is None or retry_event_prefix is None:
                break

            retry_attempt += 1
            last_retry_reason = retry_reason
            stalled_text = _responses_output_text(responses_body)

            _log_event(
                f"{retry_event_prefix}_start",
                request_id,
                retry_attempt=retry_attempt,
                empty_retry_count=empty_retry_count,
                stall_retry_count=stall_retry_count,
                reason=retry_reason,
                stalled_text=stalled_text,
            )

            retry_chat_body = copy.deepcopy(chat_body)
            retry_chat_body.setdefault("messages", [])

            # Preserve the text-only assistant output as hidden model context, then
            # immediately correct the protocol error with a user continuation.
            if stalled_text.strip():
                retry_chat_body["messages"].append(
                    {
                        "role": "assistant",
                        "content": stalled_text,
                    }
                )

            retry_chat_body["messages"].append(
                {
                    "role": "user",
                    "content": retry_message,
                }
            )
            retry_chat_body["tool_choice"] = "auto"

            retry_upstream = await _post_chat_completions(
                client=client,
                request=request,
                request_id=request_id,
                chat_body=retry_chat_body,
                attempt=retry_attempt,
            )

            if retry_upstream.status_code < 200 or retry_upstream.status_code >= 300:
                _log_event(
                    f"{retry_event_prefix}_upstream_error",
                    request_id,
                    retry_attempt=retry_attempt,
                    status_code=retry_upstream.status_code,
                    body=_body_for_log(_safe_response_json_or_text(retry_upstream)),
                )
                break

            try:
                retry_chat_response = retry_upstream.json()
            except Exception as exc:
                _log_event(
                    f"{retry_event_prefix}_non_json_error",
                    request_id,
                    retry_attempt=retry_attempt,
                    error=str(exc),
                    upstream_text=retry_upstream.text,
                )
                break

            retry_responses_body = _chat_response_to_responses_body(
                retry_chat_response,
                body,
                request_id=request_id,
                actual_parallel_tool_calls=retry_chat_body.get("parallel_tool_calls"),
            )

            _log_event(
                f"{retry_event_prefix}_result",
                request_id,
                retry_attempt=retry_attempt,
                chat_summary=_summarize_chat_response(retry_chat_response),
                output_summary=_summarize_responses_output(retry_responses_body.get("output")),
                body=_body_for_log(retry_responses_body),
            )

            responses_body = retry_responses_body
            chat_body = retry_chat_body

        if _responses_body_is_empty_assistant_message(responses_body) and EMIT_FALLBACK_ON_EMPTY:
            _log_event(
                "empty_response_fallback_emitted",
                request_id,
                reason=(
                    "Upstream still returned no valid assistant text/tool call after retry. "
                    "Emitting visible router fallback instead of blank output."
                ),
                last_retry_reason=last_retry_reason,
            )
            responses_body = _make_empty_fallback_response(
                body,
                request_id=request_id,
                actual_parallel_tool_calls=chat_body.get("parallel_tool_calls"),
            )

        if _responses_body_is_agentic_stall_text(responses_body, body) and EMIT_FALLBACK_ON_STALL:
            stalled_text = _responses_output_text(responses_body)
            _log_event(
                "agentic_stall_fallback_emitted",
                request_id,
                reason=(
                    "Upstream repeatedly returned progress/planning text without a tool call. "
                    "Emitting visible router diagnostic instead of letting Codex silently stop."
                ),
                stalled_text=stalled_text,
                last_retry_reason=last_retry_reason,
            )
            responses_body = _make_stall_fallback_response(
                body,
                request_id=request_id,
                stalled_text=stalled_text,
                actual_parallel_tool_calls=chat_body.get("parallel_tool_calls"),
            )

    assistant_messages = _responses_output_items_to_chat_messages(
        responses_body.get("output", [])
    )

    _put_response_state(
        responses_body["id"],
        ResponseState(
            created_at=time.time(),
            messages=chat_body["messages"] + assistant_messages,
            output_items=responses_body.get("output", []),
            model=responses_body.get("model"),
            instructions=body.get("instructions"),
            request_id=request_id,
        ),
    )

    _log_event(
        "response_state_stored",
        request_id,
        response_id=responses_body.get("id"),
        state_items=len(RESPONSE_STATE),
        stored_message_summary=_summarize_chat_messages(chat_body["messages"] + assistant_messages),
        output_summary=_summarize_responses_output(responses_body.get("output")),
    )

    if client_wants_stream:
        return StreamingResponse(
            _stream_responses_sse_from_response(responses_body, request_id=request_id),
            status_code=200,
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
                "x-router-request-id": request_id,
            },
        )

    return JSONResponse(
        responses_body,
        headers={"x-router-request-id": request_id},
    )


# =============================================================================
# FastAPI endpoints
# =============================================================================


@app.get("/health")
async def health():
    return {
        "ok": True,
        "upstream": UPSTREAM_BASE_URL,
        "llm_header": LLM_KEY_HEADER,
        "has_llm_key": bool(RAW_LLM_API_KEY),
        "forward_incoming_llm_key": FORWARD_INCOMING_LLM_KEY,
        "codex_text_repair": ENABLE_CODEX_TEXT_REPAIR,
        "codex_tool_guard": INJECT_CODEX_TOOL_GUARD,
        "update_plan_tool_repair": ENABLE_UPDATE_PLAN_TOOL_REPAIR,
        "force_single_tool_call": FORCE_SINGLE_TOOL_CALL,
        "empty_response_retry_count": EMPTY_RESPONSE_RETRY_COUNT,
        "emit_fallback_on_empty": EMIT_FALLBACK_ON_EMPTY,
        "stall_text_retry": ENABLE_STALL_TEXT_RETRY,
        "stall_text_retry_count": STALL_TEXT_RETRY_COUNT,
        "emit_fallback_on_stall": EMIT_FALLBACK_ON_STALL,
        "conversation_logging": ENABLE_CONVERSATION_LOGGING,
        "log_file": str(LOG_FILE),
        "log_full_bodies": LOG_FULL_BODIES,
        "log_headers": LOG_HEADERS,
        "debug_endpoints": ENABLE_DEBUG_ENDPOINTS,
        "response_state_items": len(RESPONSE_STATE),
    }


@app.get("/debug/logs/recent")
async def debug_recent_logs(limit: int = Query(default=100, ge=1, le=1000)):
    if not ENABLE_DEBUG_ENDPOINTS:
        return JSONResponse(
            {"error": "debug endpoints are disabled; set ENABLE_DEBUG_ENDPOINTS=1"},
            status_code=404,
        )

    return {
        "log_file": str(LOG_FILE),
        "items": list(RECENT_LOGS)[-limit:],
    }


@app.get("/debug/state")
async def debug_state():
    if not ENABLE_DEBUG_ENDPOINTS:
        return JSONResponse(
            {"error": "debug endpoints are disabled; set ENABLE_DEBUG_ENDPOINTS=1"},
            status_code=404,
        )

    _prune_response_state()
    return {
        "items": len(RESPONSE_STATE),
        "ttl_seconds": RESPONSE_STATE_TTL_SECONDS,
        "max_items": RESPONSE_STATE_MAX_ITEMS,
        "states": [
            {
                "response_id": response_id,
                "age_seconds": round(time.time() - state.created_at, 2),
                "messages": len(state.messages),
                "output_summary": _summarize_responses_output(state.output_items),
                "model": state.model,
                "request_id": state.request_id,
            }
            for response_id, state in RESPONSE_STATE.items()
        ],
    }


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(path: str, request: Request):
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex}"
    upstream_url = f"{UPSTREAM_BASE_URL}/{path}"
    method = request.method.upper()
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=30.0)
    start = time.perf_counter()

    _log_event(
        "proxy_request_start",
        request_id,
        method=method,
        path=path,
        upstream_url=upstream_url,
        query_params=dict(request.query_params),
        headers=_headers_for_log(request.headers),
    )

    if method == "GET":
        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream = await client.request(
                method,
                upstream_url,
                headers=_upstream_headers(request),
                params=dict(request.query_params),
            )

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        _log_event(
            "proxy_get_response",
            request_id,
            status_code=upstream.status_code,
            elapsed_ms=elapsed_ms,
            headers=_headers_for_log(upstream.headers),
            body=_body_for_log(_safe_response_json_or_text(upstream)),
        )

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers={
                **_copy_response_headers(upstream.headers),
                "x-router-request-id": request_id,
            },
            media_type=upstream.headers.get("content-type"),
        )

    try:
        body = await request.json()
    except Exception as exc:
        _log_event(
            "invalid_json_body",
            request_id,
            error=str(exc),
        )
        return JSONResponse(
            {"error": f"invalid json body: {exc}"},
            status_code=400,
            headers={"x-router-request-id": request_id},
        )

    if path == "responses" and isinstance(body, dict):
        try:
            return await handle_responses_via_chat_completions(
                body,
                request,
                request_id=request_id,
            )
        except Exception as exc:
            _log_event(
                "unhandled_responses_exception",
                request_id,
                error=repr(exc),
            )
            raise

    # Other endpoints remain a thin proxy.
    is_stream = isinstance(body, dict) and body.get("stream") is True

    _log_event(
        "proxy_json_request_body",
        request_id,
        path=path,
        stream=is_stream,
        body=_body_for_log(body),
    )

    if is_stream:
        client = httpx.AsyncClient(timeout=None)
        upstream_req = client.build_request(
            method,
            upstream_url,
            headers=_upstream_headers(request),
            params=dict(request.query_params),
            json=body,
        )
        upstream = await client.send(upstream_req, stream=True)

        _log_event(
            "proxy_stream_response_start",
            request_id,
            status_code=upstream.status_code,
            headers=_headers_for_log(upstream.headers),
        )

        async def stream_body():
            total_bytes = 0
            try:
                async for chunk in upstream.aiter_raw():
                    total_bytes += len(chunk)
                    if LOG_SSE_EVENTS:
                        _log_event(
                            "proxy_stream_chunk",
                            request_id,
                            bytes=len(chunk),
                            chunk=_body_for_log(chunk.decode("utf-8", errors="replace")),
                        )
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()
                elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
                _log_event(
                    "proxy_stream_response_end",
                    request_id,
                    total_bytes=total_bytes,
                    elapsed_ms=elapsed_ms,
                )

        return StreamingResponse(
            stream_body(),
            status_code=upstream.status_code,
            headers={
                **_copy_response_headers(upstream.headers),
                "x-router-request-id": request_id,
            },
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    async with httpx.AsyncClient(timeout=timeout) as client:
        upstream = await client.request(
            method,
            upstream_url,
            headers=_upstream_headers(request),
            params=dict(request.query_params),
            json=body,
        )

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    _log_event(
        "proxy_response",
        request_id,
        status_code=upstream.status_code,
        elapsed_ms=elapsed_ms,
        headers=_headers_for_log(upstream.headers),
        body=_body_for_log(_safe_response_json_or_text(upstream)),
    )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers={
            **_copy_response_headers(upstream.headers),
            "x-router-request-id": request_id,
        },
        media_type=upstream.headers.get("content-type"),
    )

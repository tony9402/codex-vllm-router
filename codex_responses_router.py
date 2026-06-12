import json
import os
import re
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse

app = FastAPI()

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "http://ip:port/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_API_KEY = LLM_API_KEY if LLM_API_KEY.startswith("Bearer") else f"Bearer {LLM_API_KEY}"
LLM_KEY_HEADER = os.environ.get("LLM_KEY_HEADER", "Authorization")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "600"))

# 깨진 pseudo-Codex 출력을 실제 tool call로 복구할지 여부
ENABLE_CODEX_TEXT_REPAIR = os.environ.get("ENABLE_CODEX_TEXT_REPAIR", "1") not in {
    "0",
    "false",
    "False",
}

# upstream 모델에 Codex protocol 준수를 강하게 지시할지 여부
INJECT_CODEX_TOOL_GUARD = os.environ.get("INJECT_CODEX_TOOL_GUARD", "1") not in {
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
- If you need to update a plan and an update_plan tool exists, call that tool instead of printing XML or JSON plan text.
- Never claim that a command was executed unless the tool result is present in the conversation.
"""


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

    if incoming_key:
        headers[LLM_KEY_HEADER] = incoming_key
    elif LLM_API_KEY:
        headers[LLM_KEY_HEADER] = LLM_API_KEY

    # LiteLLM custom header 인증 구조에서는 Authorization을 일부러 넣지 않는다.
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
    Codex multi-turn Responses payload를 strict validator가 받아들이기 쉽게 보정한다.

    보정:
    - assistant message에 id/status 추가
    - assistant output_text part에 annotations 추가
    - reasoning item에 id/summary 추가
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


def _responses_input_to_chat_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Responses API input/history를 Chat Completions messages로 변환한다.
    """
    messages: list[dict[str, Any]] = []

    if INJECT_CODEX_TOOL_GUARD:
        messages.append(
            {
                "role": "system",
                "content": CODEX_TOOL_GUARD,
            }
        )

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
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

        # reasoning item은 upstream model에 다시 넣지 않는다.
        if item_type == "reasoning":
            continue

    return messages


def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]] | None:
    """
    Responses API tools를 Chat Completions tools로 변환한다.
    """
    if not isinstance(tools, list):
        return None

    chat_tools: list[dict[str, Any]] = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = tool.get("type")

        if tool_type == "function" and isinstance(tool.get("function"), dict):
            chat_tools.append(tool)
            continue

        if tool_type != "function":
            # web_search, file_search, computer_use 같은 built-in Responses tool은
            # vLLM Chat Completions function tool로 직접 변환할 수 없다.
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


def _responses_body_to_chat_body(body: dict[str, Any]) -> dict[str, Any]:
    messages = _responses_input_to_chat_messages(body)
    tools = _responses_tools_to_chat_tools(body.get("tools"))

    chat_body: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
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
        if tool_choice is not None:
            chat_body["tool_choice"] = tool_choice
        else:
            chat_body["tool_choice"] = "auto"

        if "parallel_tool_calls" in body:
            chat_body["parallel_tool_calls"] = body["parallel_tool_calls"]

    return chat_body


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


def _arguments_for_single_text_tool(tool: dict[str, Any], value: str) -> str:
    """
    shell/command류 tool의 parameter schema가 cmd/command/script/input 등 무엇이든
    가능한 맞춰서 JSON arguments를 만든다.
    """
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
            return json.dumps({key: value}, ensure_ascii=False)

    if isinstance(required, list) and required:
        first_required = required[0]
        if isinstance(first_required, str):
            return json.dumps({first_required: value}, ensure_ascii=False)

    if props:
        first_key = next(iter(props.keys()))
        return json.dumps({first_key: value}, ensure_ascii=False)

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
    """
    모델이 평문으로 뱉은 Codex 내부 태그/가짜 shell transcript를 제거한다.
    복구 가능한 tool call이 없을 때만 visible text fallback에 사용한다.
    """
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

    # 빈 줄 정리
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


def _extract_shell_command_from_transcript(text: str) -> str | None:
    """
    Gemma가 다음과 같이 가짜 Codex transcript를 출력하는 케이스를 복구한다.

    Shell
    $mkdir -p ...

    또는

    mkdir -p ... 실행 완료

    또는 heredoc:

    cat << 'EOF' > file.py
    ...
    EOF 실행 완료
    """
    # 1. Shell transcript의 $ command 라인
    shell_match = re.search(
        r"(?ms)^\s*Shell\s*\n\$([^\n]+(?:\n(?!\s*(?:출력 없음|성공|Shell|\$|<\|channel\>)).*)*)",
        text,
    )
    if shell_match:
        command = shell_match.group(1).strip()
        if command:
            return command

    # 2. heredoc command + EOF 실행 완료
    heredoc_match = re.search(
        r"(?ms)^((?:cat|tee)\s+.*?<<\s*['\"]?EOF['\"]?.*?^EOF)\s*실행 완료",
        text,
    )
    if heredoc_match:
        command = heredoc_match.group(1).strip()
        if command:
            return command

    # 3. 일반 shell command + 실행 완료
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

    line_match = re.search(
        rf"(?m)^\s*(({prefix_pattern})\b[^\n]*)\s+실행 완료\s*$",
        text,
    )
    if line_match:
        command = line_match.group(1).strip()
        if command:
            return command

    # 4. 코드블록 안에 shell/bash가 있는 경우
    fenced_match = re.search(
        r"(?ms)```(?:bash|sh|shell)\s*\n(.*?)\n```",
        text,
    )
    if fenced_match:
        command = fenced_match.group(1).strip()
        if command:
            return command

    return None


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

    command = _extract_shell_command_from_transcript(text)
    if not command:
        return []

    arguments = _arguments_for_single_text_tool(shell_tool, command)
    return [_make_function_call_item(tool_name, arguments)]


def _repair_codex_pseudo_tool_output(
    content: str,
    original_body: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """
    upstream이 OpenAI tool_calls를 만들지 못하고 Codex UI 흉내 텍스트를 출력한 경우
    가능한 부분을 Responses function_call로 복구한다.

    반환:
    - 복구된 function_call output items
    - fallback visible text
    """
    if not ENABLE_CODEX_TEXT_REPAIR or not isinstance(content, str) or not content.strip():
        return [], content or ""

    calls: list[dict[str, Any]] = []

    calls.extend(_extract_update_plan_calls(content, original_body))
    calls.extend(_extract_shell_calls(content, original_body))

    cleaned_text = _strip_codex_pseudo_markup(content)

    # tool call이 복구되면 깨진 일반 텍스트는 Codex에 넘기지 않는다.
    # Codex는 function_call_output을 받은 뒤 다음 턴에 이어서 진행해야 한다.
    if calls:
        return calls, ""

    return [], cleaned_text


def _chat_response_to_responses_body(
    chat: dict[str, Any],
    original_body: dict[str, Any],
) -> dict[str, Any]:
    """
    Chat Completions response를 Responses API response object로 변환한다.
    """
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

        # tool_calls가 있으면 content는 보통 비어 있어야 한다.
        # 일부 모델이 content도 같이 주는 경우 Codex 혼선을 막기 위해 생략한다.
        content = ""

    elif isinstance(content, str) and content.strip():
        repaired_calls, repaired_text = _repair_codex_pseudo_tool_output(
            content,
            original_body,
        )

        if repaired_calls:
            output_items.extend(repaired_calls)
            content = ""
        else:
            content = repaired_text

    if isinstance(content, str) and content.strip():
        output_items.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        )

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
        "parallel_tool_calls": original_body.get("parallel_tool_calls", True),
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


def _sse_event(event_name: str, payload: dict[str, Any]) -> bytes:
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    ).encode("utf-8")


async def _stream_responses_sse_from_response(response_body: dict[str, Any]):
    """
    이미 완성된 Responses response object를 Responses API 스타일 SSE로 방출한다.

    upstream은 non-stream으로 호출하지만, Codex 클라이언트가 stream=true를 보낼 수 있으므로
    router가 최소 호환 SSE event를 합성한다.
    """
    seq = 0

    def event(event_name: str, payload: dict[str, Any]) -> bytes:
        nonlocal seq
        payload = {
            "type": event_name,
            "sequence_number": seq,
            **payload,
        }
        seq += 1
        return _sse_event(event_name, payload)

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


async def handle_responses_via_chat_completions(
    body: dict[str, Any],
    request: Request,
):
    """
    /v1/responses 요청을 vLLM/LiteLLM의 /v1/chat/completions tool calling으로 브리지한다.
    """
    body = normalize_responses_body(body)

    client_wants_stream = body.get("stream") is True

    chat_body = _responses_body_to_chat_body(body)
    upstream_url = f"{UPSTREAM_BASE_URL}/chat/completions"

    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        upstream = await client.post(
            upstream_url,
            headers=_upstream_headers(request),
            params=dict(request.query_params),
            json=chat_body,
        )

    if upstream.status_code < 200 or upstream.status_code >= 300:
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_copy_response_headers(upstream.headers),
            media_type=upstream.headers.get("content-type"),
        )

    try:
        chat_response = upstream.json()
    except Exception as exc:
        return JSONResponse(
            {
                "error": f"upstream returned non-json response: {exc}",
                "upstream_text": upstream.text,
            },
            status_code=502,
        )

    responses_body = _chat_response_to_responses_body(chat_response, body)

    if client_wants_stream:
        return StreamingResponse(
            _stream_responses_sse_from_response(responses_body),
            status_code=200,
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "x-accel-buffering": "no",
            },
        )

    return JSONResponse(responses_body)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "upstream": UPSTREAM_BASE_URL,
        "litellm_header": LLM_KEY_HEADER,
        "has_llm_key": bool(LLM_API_KEY),
        "codex_text_repair": ENABLE_CODEX_TEXT_REPAIR,
        "codex_tool_guard": INJECT_CODEX_TOOL_GUARD,
    }


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(path: str, request: Request):
    upstream_url = f"{UPSTREAM_BASE_URL}/{path}"
    method = request.method.upper()

    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=30.0)

    if method == "GET":
        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream = await client.request(
                method,
                upstream_url,
                headers=_upstream_headers(request),
                params=dict(request.query_params),
            )

        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=_copy_response_headers(upstream.headers),
            media_type=upstream.headers.get("content-type"),
        )

    try:
        body = await request.json()
    except Exception as exc:
        return JSONResponse(
            {"error": f"invalid json body: {exc}"},
            status_code=400,
        )

    if path == "responses" and isinstance(body, dict):
        return await handle_responses_via_chat_completions(body, request)

    is_stream = isinstance(body, dict) and body.get("stream") is True

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

        async def stream_body():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream.status_code,
            headers=_copy_response_headers(upstream.headers),
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

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=_copy_response_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )

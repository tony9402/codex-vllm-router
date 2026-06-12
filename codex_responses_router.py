import json
import os
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse

app = FastAPI()

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
API_KEY = API_KEY if API_KEY.startswith("Bearer") else f"Bearer {API_KEY}"
KEY_HEADER = os.environ.get("LLM_KEY_HEADER", "Authorization")
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "600"))

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

    incoming_key = request.headers.get(KEY_HEADER)

    if incoming_key:
        headers[KEY_HEADER] = incoming_key
    elif API_KEY:
        headers[KEY_HEADER] = API_KEY

    # LiteLLM custom header 인증을 쓰는 구조에서는 Authorization을 넣지 않는다.
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
    """
    Responses API content part list를 Chat Completions용 text로 평탄화한다.
    Codex에는 보통 input_text/output_text가 들어온다.
    """
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

    Responses function_call:
      {"type":"function_call","call_id":"call_x","name":"...","arguments":"{...}"}

    Chat Completions assistant tool call:
      {"role":"assistant","tool_calls":[
        {"id":"call_x","type":"function","function":{"name":"...","arguments":"{...}"}}
      ]}

    Responses function_call_output:
      {"type":"function_call_output","call_id":"call_x","output":"..."}

    Chat Completions tool result:
      {"role":"tool","tool_call_id":"call_x","content":"..."}
    """
    messages: list[dict[str, Any]] = []

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

        # OpenAI Responses에서 type 없이 {"role":"user","content":"..."} 형태도 가능하다.
        if item_type in (None, "message") and role in {
            "system",
            "developer",
            "user",
            "assistant",
        }:
            chat_role = "system" if role == "developer" else role
            content = _extract_text_from_responses_content(item.get("content"))

            # 빈 assistant message는 chat template에 따라 문제를 만들 수 있으므로 생략.
            if chat_role == "assistant" and not content.strip():
                continue

            messages.append(
                {
                    "role": chat_role,
                    "content": content,
                }
            )
            continue

        # 이전 턴에서 모델이 낸 function_call을 Chat Completions history로 복원.
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

            # 직전 assistant message가 있으면 그 message에 tool_calls를 붙인다.
            # 없으면 별도의 assistant tool_calls message를 만든다.
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

        # tool 실행 결과를 Chat Completions tool message로 변환.
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
        # reasoning summary를 일반 text로 주입하면 모델 품질과 보안 양쪽에 악영향이 날 수 있다.
        if item_type == "reasoning":
            continue

    return messages


def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]] | None:
    """
    Responses API tools를 Chat Completions tools로 변환한다.

    Responses function tool:
      {"type":"function","name":"foo","description":"...","parameters":{...}}

    Chat tool:
      {"type":"function","function":{"name":"foo","description":"...","parameters":{...}}}
    """
    if not isinstance(tools, list):
        return None

    chat_tools: list[dict[str, Any]] = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = tool.get("type")

        # 이미 Chat Completions 형식이면 그대로 통과.
        if tool_type == "function" and isinstance(tool.get("function"), dict):
            chat_tools.append(tool)
            continue

        if tool_type != "function":
            # web_search, file_search, computer_use 같은 built-in Responses tool은
            # vLLM Chat Completions function tool로 직접 변환할 수 없다.
            # Codex의 local function tool만 여기서 처리한다.
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
    """
    Responses tool_choice를 Chat Completions tool_choice로 변환한다.
    """
    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        # auto / none / required
        return tool_choice

    if isinstance(tool_choice, dict):
        # Responses:
        #   {"type":"function","name":"foo"}
        #
        # Chat Completions:
        #   {"type":"function","function":{"name":"foo"}}
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
    """
    /v1/responses request를 /v1/chat/completions request로 변환한다.
    """
    messages = _responses_input_to_chat_messages(body)
    tools = _responses_tools_to_chat_tools(body.get("tools"))

    chat_body: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        # router가 Responses SSE를 직접 합성하기 위해 upstream은 non-stream으로 호출한다.
        "stream": False,
    }

    # Responses API: max_output_tokens
    # Chat Completions: max_tokens
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

    if isinstance(tool_calls, list):
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

    # tool call도 text도 없으면 빈 assistant message를 반환한다.
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

    # 일부 OpenAI-compatible client는 [DONE] sentinel을 기대한다.
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
        "litellm_header": KEY_HEADER,
        "has_litellm_key": bool(API_KEY),
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

    # 핵심: Codex의 /v1/responses는 그대로 proxy하지 말고
    # Chat Completions tools로 변환해서 upstream에 보낸 뒤 다시 Responses 형식으로 반환한다.
    if path == "responses" and isinstance(body, dict):
        return await handle_responses_via_chat_completions(body, request)

    # 그 외 endpoint는 기존처럼 단순 proxy.
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

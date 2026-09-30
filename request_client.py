"""带 prefix-cache 统计的 OpenAI chat completions 流式客户端（sglang 内置客户端的 fork）。

sglang 内置 async_request_openai_chat_completions 只解析 usage.completion_tokens，
无法统计 prefix cache 命中。本 fork 额外解析每个请求的：

  - usage.prompt_tokens                       -> output.prompt_tokens_actual（服务端实际输入 token 数）
  - usage.prompt_tokens_details.cached_tokens  -> output.cached_tokens（前缀缓存命中 token 数）

服务端要求：sglang 启动时加 --enable-cache-report 才会在 usage 中返回
prompt_tokens_details.cached_tokens；否则 cached_tokens 恒为 0（命中率不可测，
报告 cache_observable_rate 会显示 0）。

请求侧自动附带 stream_options.include_usage=true，确保流式模式下 usage 在最终
chunk 中返回（OpenAI 标准；vLLM 必需，sglang 亦支持）。
"""

import json
import os
import sys
import time
import traceback
from typing import Dict

import aiohttp

import sglang.benchmark.serving as serving_mod
from sglang.benchmark.serving import RequestFuncOutput

_ROUTING_KEY_HEADER = "X-SMG-Routing-Key"

BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60
BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024 ** 2


def _remove_prefix(text: str, prefix: str) -> str:
    return text[len(prefix):] if text.startswith(prefix) else text


def _create_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS),
        read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES,
    )


def _get_request_headers() -> Dict[str, str]:
    """优先复用 sglang 内置 get_request_headers（含 --header 自定义头），失败则本地实现。"""
    fn = getattr(serving_mod, "get_request_headers", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    api_key = os.environ.get("API_KEY")
    if api_key:
        return {"Authorization": api_key}
    return {}


def _apply_usage(output: RequestFuncOutput, usage: dict) -> None:
    """把服务端 usage 中的 token 记录到 output 对象上（供报告计算真实命中率）。"""
    prompt_tokens = usage.get("prompt_tokens") or 0
    if prompt_tokens:
        output.prompt_tokens_actual = prompt_tokens
    details = usage.get("prompt_tokens_details")
    if details is not None:
        output.cached_tokens_details = details
        output.cached_tokens = details.get("cached_tokens") or 0


async def async_request_openai_chat_completions_cached(
    request_func_input, pbar=None
) -> RequestFuncOutput:
    """Fork of sglang async_request_openai_chat_completions.

    与内置版本的差异：
      1. payload 附带 stream_options.include_usage=true（确保流式最终 chunk 带 usage）
      2. 解析 usage.prompt_tokens -> output.prompt_tokens_actual
      3. 解析 usage.prompt_tokens_details.cached_tokens -> output.cached_tokens
    其余行为（TTFT/ITL/latency/output_len/错误处理）与内置版本一致。
    """
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    if isinstance(request_func_input.prompt, list):
        messages = request_func_input.prompt
    elif request_func_input.image_data:
        content_items = [
            {"type": "image_url", "image_url": {"url": img_url}}
            for img_url in request_func_input.image_data
        ]
        content_items.append({"type": "text", "text": request_func_input.prompt})
        messages = [{"role": "user", "content": content_items}]
    else:
        messages = [{"role": "user", "content": request_func_input.prompt}]

    sargs = getattr(serving_mod, "args", None)
    disable_stream = bool(getattr(sargs, "disable_stream", False))
    disable_ignore_eos = bool(getattr(sargs, "disable_ignore_eos", False))

    extra_request_body = request_func_input.extra_request_body or {}

    async with _create_session() as session:
        payload = {
            "model": request_func_input.model,
            "messages": messages,
            "max_completion_tokens": request_func_input.output_len,
            "stream": not disable_stream,
        }
        if "temperature" not in extra_request_body:
            payload["temperature"] = 0.0
        if "ignore_eos" not in extra_request_body:
            payload["ignore_eos"] = not disable_ignore_eos
        payload.update(extra_request_body)
        if payload.get("stream"):
            payload["stream_options"] = {"include_usage": True}

        headers = _get_request_headers()
        routing_key = getattr(request_func_input, "routing_key", None)
        if routing_key:
            headers[_ROUTING_KEY_HEADER] = routing_key

        output = RequestFuncOutput.init_new(request_func_input)
        output.prompt_tokens_actual = 0
        output.cached_tokens = 0

        generated_text = ""
        output_len = request_func_input.output_len
        ttft = 0.0
        st = time.perf_counter()
        output.start_time = st
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    if disable_stream:
                        response_json = await response.json()
                        output.generated_text = response_json["choices"][0][
                            "message"
                        ]["content"]
                        output.success = True
                        output.latency = time.perf_counter() - st
                        output.ttft = output.latency
                        usage = response_json.get("usage") or {}
                        comp = usage.get("completion_tokens")
                        if comp is not None:
                            output.output_len = comp
                        _apply_usage(output, usage)
                    else:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            chunk = _remove_prefix(
                                chunk_bytes.decode("utf-8"), "data: "
                            )
                            latency = time.perf_counter() - st
                            if chunk == "[DONE]":
                                pass
                            else:
                                data = json.loads(chunk)

                                # include_usage 的最终 chunk choices 为空列表
                                choices = data.get("choices") or []
                                if choices:
                                    delta = (choices[0] or {}).get("delta") or {}
                                    content = delta.get("content", "")
                                    if content:
                                        timestamp = time.perf_counter()
                                        if ttft == 0.0:
                                            ttft = timestamp - st
                                            output.ttft = ttft
                                        else:
                                            output.itl.append(
                                                timestamp - most_recent_timestamp
                                            )
                                        most_recent_timestamp = timestamp
                                        generated_text += content

                                usage = data.get("usage") or {}
                                if usage:
                                    comp = usage.get("completion_tokens")
                                    if comp is not None:
                                        output_len = comp
                                    _apply_usage(output, usage)

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = latency
                        output.output_len = output_len
                else:
                    output.error = (
                        (response.reason or "") + ": " + (await response.text())
                    )
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

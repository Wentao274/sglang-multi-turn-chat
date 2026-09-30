"""压测前探测：服务端响应 usage 是否返回 cached_tokens（prefix cache 命中数据）。

用途：正式压测前快速确认服务端已开启 --enable-cache-report。若响应缺失
prompt_tokens_details.cached_tokens，报告中缓存命中率将不可测（显示 N/A）。

行为（模拟多轮场景）：
  请求 A：长 system prompt + user 问题（全新上下文，预期 cached_tokens = 0）
  请求 B：重发 A 的完整历史 + assistant 回复 + 追问（预期 cached_tokens > 0，
           即下一轮命中前一轮的前缀 —— 多轮缓存命中率测试的微观验证）

每一步都打印响应中的原始 usage 字段和单请求命中率 = cached_tokens / prompt_tokens。

用法：
  python probe_cache_report.py [--base-url http://...] [--model ...] [--api-key ...] \
                               [--env-file configs/.env]

base-url/model/api-key 缺省时自动读取 .env（BASE_URL/MODEL/API_KEY）。
"""

import argparse
import json
import os
import sys

import requests


def load_env(path):
    values = {}
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def chat(base_url, api_key, model, messages, max_tokens):
    """发送流式 chat 请求（与压测客户端一致：stream_options.include_usage=true）。

    返回 (assistant_text, usage_dict)。usage 取最终携带 usage 的 chunk。
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=300, stream=True)
    resp.raise_for_status()
    text = ""
    usage = {}
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data: "):
            continue
        data = raw[len("data: "):]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        for ch in chunk.get("choices") or []:
            text += (ch.get("delta") or {}).get("content") or ""
        if chunk.get("usage"):
            usage = chunk["usage"]
    return text, usage


def show(tag, usage):
    print(f"\n[{tag}] 响应 usage 字段（服务端原样返回）：")
    print(json.dumps(usage, ensure_ascii=False, indent=2))
    if not usage:
        print("  [FAIL] 流式响应未包含 usage：检查服务端是否支持 stream_options.include_usage")
        return False
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict) or "cached_tokens" not in details:
        print("  [FAIL] 响应未包含 prompt_tokens_details.cached_tokens")
        print("         -> 服务端大概率未开启 --enable-cache-report")
        return False
    cached = details.get("cached_tokens") or 0
    prompt = usage.get("prompt_tokens") or 0
    if prompt:
        print(f"  本请求命中率 = cached_tokens / prompt_tokens = {cached} / {prompt} = {cached / prompt:.2%}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="探测服务端响应 usage 中的 prefix cache 命中字段",
    )
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key")
    ap.add_argument("--env-file", default="configs/.env")
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    env = load_env(args.env_file)
    base_url = args.base_url or env.get("BASE_URL")
    model = args.model or env.get("MODEL")
    api_key = args.api_key or env.get("API_KEY")
    if not base_url:
        ap.error("需要 --base-url（或 .env 中 BASE_URL）")
    if not model:
        try:
            items = requests.get(base_url.rstrip("/") + "/v1/models", timeout=10).json().get("data") or []
            model = items[0].get("id") if items else None
        except Exception:
            model = None
        if not model:
            ap.error("需要 --model（或 .env 中 MODEL，或服务端 /v1/models 可探测）")

    print(f"[probe] base_url={base_url}  model={model}")

    filler = "prefix cache 通过 radix tree 复用历史 KV，降低重复前缀的重新计算开销。"
    system = "你是技术助手。参考资料：" + filler * 60
    msgs_a = [
        {"role": "system", "content": system},
        {"role": "user", "content": "用一句话说明什么是 prefix cache。"},
    ]

    print("\n=== 请求 A：全新上下文（预期 cached_tokens = 0） ===")
    try:
        reply_a, usage_a = chat(base_url, api_key, model, msgs_a, args.max_tokens)
    except Exception as e:
        print(f"[FAIL] 请求 A 失败：{e}")
        return 1
    ok_a = show("A", usage_a)

    msgs_b = msgs_a + [
        {"role": "assistant", "content": reply_a or "（无回复）"},
        {"role": "user", "content": "再补充一句它和 KV cache 的关系。"},
    ]
    print("\n=== 请求 B：重发完整历史 + 追问（多轮下一轮，预期 cached_tokens > 0） ===")
    try:
        _, usage_b = chat(base_url, api_key, model, msgs_b, args.max_tokens)
    except Exception as e:
        print(f"[FAIL] 请求 B 失败：{e}")
        return 1
    ok_b = show("B", usage_b)

    print("\n=== 结论 ===")
    if not (ok_a and ok_b):
        print("[FAIL] 服务端响应缺少 cached_tokens 字段。")
        print("       正式压测前请在 sglang 启动参数加 --enable-cache-report，")
        print("       否则报告中的缓存命中率不可测（显示 N/A）。")
        return 1
    cached_b = ((usage_b.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    if cached_b > 0:
        print("[OK] 服务端已返回 cached_tokens，且多轮下一轮成功命中前缀。")
        print("     压测框架 request_client.py 会自动采集该字段，")
        print("     报告'测试结论汇总/稳态缓存命中率'与'分轮表 cache hit 列'均为真实值。")
    else:
        print("[WARN] 字段存在但请求 B cached_tokens = 0（未命中）。可能原因：")
        print("       1) 服务端启动加了一 --disable-radix-cache（prefix cache 被禁用）；")
        print("       2) 请求间隔内缓存被淘汰/清理。可重跑一次确认。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Smoke tests for the cached-token client fork and report integration.

sglang cannot be imported locally (pybase64 fails), so we stub the
sglang.benchmark.serving / sglang.benchmark.utils modules in sys.modules
before importing our modules. Then:

1. Spin up a local aiohttp server that mimics an sglang server with
   --enable-cache-report (streams content chunks, then a final
   usage-only chunk with choices: [], then [DONE]).
2. Call the forked async_request_openai_chat_completions_cached and
   verify: success, cached_tokens, prompt_tokens_actual.
3. Run report._round_breakdown with mocked outputs to verify per-round
   cache hit uses server values (and falls back to estimate when absent).

Usage:  python test_cached_client.py
"""

import sys
import types
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

# ============================================================
# Stub sglang modules (mirrors the interface of sglang.benchmark.serving)
# ============================================================


@dataclass
class RequestFuncInput:
    prompt: Any
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    lora_name: str
    image_data: Optional[List[str]]
    extra_request_body: Optional[dict] = None
    routing_key: Optional[str] = None
    priority: int = 0
    seed: Optional[int] = None
    top_p: float = 1.0
    top_k: int = -1
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    start_time: Optional[float] = None


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: List[float] = field(default_factory=list)
    prompt_len: int = 0
    error: str = ""
    output_len: int = 0
    cached_tokens: int = 0
    cached_tokens_details: Optional[dict] = None
    prompt_tokens_actual: int = 0
    start_time: float = 0.0

    @classmethod
    def init_new(cls, input):
        return cls(
            prompt_len=input.prompt_len,
            output_len=input.output_len,
        )


@dataclass
class BenchmarkMetrics:
    completed: int = 0
    request_throughput: float = 0.0
    total_input: int = 0
    total_input_text: int = 0
    total_input_vision: int = 0
    total_output: int = 0
    total_output_retokenized: int = 0
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    std_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    std_tpot_ms: float = 0.0
    p90_tpot_ms: float = 0.0
    p95_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    std_itl_ms: float = 0.0
    p90_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    mean_e2e_ms: float = 0.0
    median_e2e_ms: float = 0.0
    std_e2e_ms: float = 0.0
    p90_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    concurrency: float = 0.0
    max_concurrent_requests: int = 0


_serving_ns = types.SimpleNamespace(
    disable_stream=False,
    disable_ignore_eos=False,
    max_concurrency=10,
    seed=42,
    pbar=False,
)


def calculate_metrics(other, input_requests, dur_ts, tokenizer, specific_part=1.0):
    if not input_requests:
        return BenchmarkMetrics(), []
    out_lens = []
    for i in input_requests:
        if not i.success:
            continue
        out_lens.append(i.output_len)
    metrics = BenchmarkMetrics(
        completed=len([r for r in input_requests if r.success]),
        total_output=sum(out_lens),
    )
    return metrics, out_lens


def wait_for_endpoint(endpoint_url, timeout=None, quiet=False):
    return True


def flush_server_cache(base_url, backend):
    return True


def wrap_multi_turn_request_func(request_func, backend):
    """镜像服务器安装版 sglang 的真实行为（1008_2 实测确认）：

    - 逐轮累积对话历史（round1+ 请求携带前轮内容 → prefix cache 命中前提）
    - 构造 inner RequestFuncInput 时传递安装版已知的必填字段
      （lora_name/image_data——服务器旧版 sglang 必填，缺失即 TypeError，
      1008_5 全部请求失败的根因）
    - **丢弃** extra_request_body / routing_key（安装版不认识的自定义字段，
      1008_2 中亲和与 reasoning_effort 未生效的根因）——由 request_client
      的内容哈希/全局 args 兜底补偿
    - 不访问 start_time（服务器版 RequestFuncInput 无该字段，
      1008_4 全部请求失败的根因）
    """
    async def _wrapped_multi_turn(input, pbar=None):
        prev_messages = []
        results = []
        for i, prompt in enumerate(input.prompt):
            prev_messages.append({"role": "user", "content": prompt})
            inner_input = RequestFuncInput(
                prompt=list(prev_messages),
                api_url=input.api_url,
                prompt_len=input.prompt_len,
                output_len=input.output_len,
                model=input.model,
                lora_name=getattr(input, "lora_name", ""),
                image_data=getattr(input, "image_data", None),
            )
            output = await request_func(
                request_func_input=inner_input,
                pbar=pbar if i == len(input.prompt) - 1 else None,
            )
            results.append(output)
            prev_messages.append(
                {"role": "assistant", "content": output.generated_text}
            )
        return results

    return _wrapped_multi_turn


async def async_request_openai_chat_completions(request_func_input, pbar=None):
    raise NotImplementedError("builtin client (not under test)")


def _build_stub_package():
    sglang_mod = types.ModuleType("sglang")
    benchmark_mod = types.ModuleType("sglang.benchmark")
    serving_mod = types.ModuleType("sglang.benchmark.serving")
    utils_mod = types.ModuleType("sglang.benchmark.utils")
    datasets_mod = types.ModuleType("sglang.benchmark.datasets")
    datasets_common_mod = types.ModuleType("sglang.benchmark.datasets.common")

    @dataclass
    class DatasetRow:
        prompt: str
        prompt_len: int
        output_len: int
        image_data: Optional[List[str]] = None

    datasets_common_mod.DatasetRow = DatasetRow
    datasets_common_mod.SHAREGPT_FILENAME = "sharegpt_multiturn.json"
    datasets_common_mod.SHAREGPT_REPO_ID = "Aeiftch/sharegpt_multiturn"
    datasets_common_mod.gen_prompt = lambda system_prompt, n: f"{system_prompt} {n}"
    utils_mod.download_and_cache_hf_file = lambda *a, **k: ""
    utils_mod.is_file_valid_json = lambda p: True

    serving_mod.RequestFuncInput = RequestFuncInput
    serving_mod.RequestFuncOutput = RequestFuncOutput
    serving_mod.BenchmarkMetrics = BenchmarkMetrics
    serving_mod.MULTI_TURN_BACKENDS = ["openai", "openai_azure", "sglang"]
    serving_mod.calculate_metrics = calculate_metrics
    serving_mod.wait_for_endpoint = wait_for_endpoint
    serving_mod.flush_server_cache = flush_server_cache
    serving_mod.wrap_multi_turn_request_func = wrap_multi_turn_request_func
    serving_mod.async_request_openai_chat_completions = async_request_openai_chat_completions
    serving_mod.get_request_headers = lambda: {}
    serving_mod.args = _serving_ns

    def get_tokenizer(tokenizer_id, **kwargs):
        class _DummyTokenizer:
            name_or_path = "dummy"

            def encode(self, *a, **k):
                return []

            def decode(self, *a, **k):
                return ""

            def apply_chat_template(self, *a, **k):
                return "chat-template-applied"

        return _DummyTokenizer()

    utils_mod.get_tokenizer = get_tokenizer

    sglang_mod.benchmark = benchmark_mod
    benchmark_mod.serving = serving_mod
    benchmark_mod.utils = utils_mod
    benchmark_mod.datasets = datasets_mod
    datasets_mod.common = datasets_common_mod
    serving_mod.utils = utils_mod

    sys.modules["sglang"] = sglang_mod
    sys.modules["sglang.benchmark"] = benchmark_mod
    sys.modules["sglang.benchmark.serving"] = serving_mod
    sys.modules["sglang.benchmark.utils"] = utils_mod
    sys.modules["sglang.benchmark.datasets"] = datasets_mod
    sys.modules["sglang.benchmark.datasets.common"] = datasets_common_mod
    return serving_mod


_build_stub_package()

# Stub transformers (not installed locally; only PreTrainedTokenizerBase type hint is used)
_tf_mod = types.ModuleType("transformers")
_tf_mod.PreTrainedTokenizerBase = object
sys.modules.setdefault("transformers", _tf_mod)

import request_client  # noqa: E402
import report  # noqa: E402
from request_client import async_request_openai_chat_completions_cached  # noqa: E402


# ============================================================
# Mock sglang server with --enable-cache-report
# ============================================================

SSE_CHUNKS = [
    b'data: {"choices": [{"delta": {"role": "assistant"}}]}',
    b'data: {"choices": [{"delta": {"content": "You"}}]}',
    b'data: {"choices": [{"delta": {"content": " are"}}]}',
    b'data: {"choices": [{"delta": {"content": " helpful"}}]}',
    b'data: {"choices": [{"delta": {"content": "!"}}]}',
    b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
    b'data: {"choices": [], "usage": {"prompt_tokens": 1213, "completion_tokens": 4, '
    b'"total_tokens": 1217, "prompt_tokens_details": {"cached_tokens": 1122, "audio_tokens": 0}}}',
    b"data: [DONE]",
]

# 推理模型（glm-5.3 等）：先流式输出 reasoning_content（思考段），再输出 content。
SSE_CHUNKS_REASONING = [
    b'data: {"choices": [{"delta": {"role": "assistant"}}]}',
    b'data: {"choices": [{"delta": {"reasoning_content": "Thinking hard"}}]}',
    b'data: {"choices": [{"delta": {"reasoning_content": " about it"}}]}',
    b'data: {"choices": [{"delta": {"content": "The answer"}}]}',
    b'data: {"choices": [{"delta": {"content": " is 42."}}]}',
    b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}',
    b'data: {"choices": [], "usage": {"prompt_tokens": 1213, "completion_tokens": 6, '
    b'"total_tokens": 1219, "prompt_tokens_details": {"cached_tokens": 1122, "audio_tokens": 0}}}',
    b"data: [DONE]",
]

_ACTIVE_CHUNKS = SSE_CHUNKS
_CAPTURED_PAYLOAD = {}
_CAPTURED_HEADERS = {}
_ROUTING_KEYS = []


async def _handle_chat_completions(request):
    from aiohttp import web

    try:
        _CAPTURED_PAYLOAD.clear()
        _CAPTURED_PAYLOAD.update(await request.json())
    except Exception:
        pass
    for k, v in request.headers.items():
        _CAPTURED_HEADERS[k] = v
    _ROUTING_KEYS.append(request.headers.get("X-SMG-Routing-Key"))
    resp = web.StreamResponse(status=200)
    resp.headers["Content-Type"] = "text/event-stream"
    await resp.prepare(request)
    for chunk in _ACTIVE_CHUNKS:
        await resp.write(chunk + b"\n\n")
        await asyncio.sleep(0.001)
    return resp


# ============================================================
# Tests
# ============================================================


def test_fork_captures_cached_tokens():
    """Fork must parse usage + prompt_tokens_details from the final SSE chunk."""

    async def _run():
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/v1/chat/completions", _handle_chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 18111)
        await site.start()
        try:
            inner_input = RequestFuncInput(
                prompt="hello world",
                api_url="http://127.0.0.1:18111/v1/chat/completions",
                prompt_len=11,
                output_len=4,
                model="test-model",
            lora_name="",
                image_data=None,
            )
            return await async_request_openai_chat_completions_cached(inner_input)
        finally:
            await runner.cleanup()

    out = asyncio.run(_run())
    assert out.success, f"request failed: {out.error}"
    assert out.prompt_tokens_actual == 1213, out.prompt_tokens_actual
    assert out.cached_tokens == 1122, out.cached_tokens
    assert out.cached_tokens_details is not None
    assert out.output_len == 4
    assert out.generated_text == "You are helpful!"
    print("PASS: fork captures cached_tokens=1122 / prompt_tokens_actual=1213")


def test_round_breakdown_uses_server_values():
    """_round_breakdown must use server-reported prompt tokens as denominator."""
    # Two sessions, 2 rounds each.
    # Session 0: round0 prompt=100 (server says 120), output 50; round1 prompt=100+50
    s0r0 = RequestFuncOutput(success=True, prompt_len=100, output_len=50, ttft=0.05,
                             latency=1.0, itl=[0.02, 0.03])
    s0r0.prompt_tokens_actual = 120
    s0r0.cached_tokens = 60
    s0r0.cached_tokens_details = {"cached_tokens": 60}
    s0r1 = RequestFuncOutput(success=True, prompt_len=100, output_len=40, ttft=0.06,
                             latency=0.9, itl=[0.02, 0.03])
    s0r1.prompt_tokens_actual = 200
    s0r1.cached_tokens = 150
    s0r1.cached_tokens_details = {"cached_tokens": 150}

    # Session 1: no server values -> estimate fallback
    s1r0 = RequestFuncOutput(success=True, prompt_len=90, output_len=45, ttft=0.07,
                            latency=1.1, itl=[0.02, 0.03])
    s1r1 = RequestFuncOutput(success=True, prompt_len=90, output_len=35, ttft=0.08,
                            latency=0.8, itl=[0.02, 0.03])

    sessions = [[s0r0, s0r1], [s1r0, s1r1]]
    rows, _ = report._round_breakdown(sessions, duration=10.0, max_tokens_per_turn=50)

    # Round 0: server 120 + estimate 90 = 210; cached = 60
    r0 = rows[0]
    assert r0[6] == 60 / 210, r0
    # Round 1: server 200 + estimate (90+45=135) = 335; cached = 150
    r1 = rows[1]
    assert r1[6] == 150 / 335, r1
    print(f"PASS: round0 hit={r0[6]:.4f} (60/210), round1 hit={r1[6]:.4f} (150/335)")


def test_generate_markdown_cache_metrics():
    """generate_markdown steady cache hit should use server-reported totals."""
    s0r0 = RequestFuncOutput(success=True, prompt_len=100, output_len=50, ttft=0.05,
                          latency=1.0, itl=[0.02, 0.03], start_time=100.0)
    s0r0.prompt_tokens_actual = 120
    s0r0.cached_tokens = 60
    s0r0.cached_tokens_details = {"cached_tokens": 60}
    s0r1 = RequestFuncOutput(success=True, prompt_len=100, output_len=40, ttft=0.06,
                          latency=0.9, itl=[0.02, 0.03], start_time=101.0)
    s0r1.prompt_tokens_actual = 200
    s0r1.cached_tokens = 150
    s0r1.cached_tokens_details = {"cached_tokens": 150}

    sessions = [[s0r0, s0r1]]
    steady_sessions = [[s0r0, s0r1]]

    args = _mk_args()

    metrics_full = BenchmarkMetrics(completed=2, request_throughput=0.2, total_output=90)
    metrics_steady = BenchmarkMetrics(completed=2, request_throughput=0.2, total_output=90)

    md = report.generate_markdown(
        args, metrics_full, metrics_steady,
        sessions, steady_sessions,
        wall_dur=10.0, steady_dur=10.0,
        steady_sessions=1, ramp_sessions=0,
        terminated=False, termination_reason=None,
        monitor_history=[], cum_429=0,
        backend="sglang", model="test-model",
        target_rps=0.2, start_rps=0.1, output_lens=[50, 40],
    )
    # steady input = 120 + 200 = 320 (server), cached = 210
    assert "steady_cache_hit_rate" in md
    # Find the cache hit value: 210/320 = 0.65625
    assert "0.6563" in md or "0.656" in md, "steady cache hit 210/320 not found"
    # Also check 测试结论汇总 exists
    assert "测试结论汇总" in md
    # full input uses server values too
    assert "cache_hit_rate | 0.66" in md or "0.6563" in md
    print("PASS: generate_markdown steady cache hit = 210/320 = 0.6563")


def _mk_args(*extra):
    """构建与 bench_multi_turn 一致的 args（供 generate_markdown 测试使用）。"""
    import argparse
    from config import build_parser, load_env, _preparse_env_file

    parser = argparse.ArgumentParser()
    env_file = _preparse_env_file()
    env = load_env(env_file)
    parser = build_parser(env)
    return parser.parse_args([
        "--base-url", "http://127.0.0.1:9999",
        "--dataset-path", "nonexistent.json",
        "--ramp-seconds", "10",
        "--sustain-seconds", "10",
    ] + list(extra))


def test_round_breakdown_no_cache_flag():
    """服务端未开 --enable-cache-report：无任何 cached_tokens_details。

    期望：cache hit 显示 None（N/A），而非误导性的 0.00%。
    """
    s0 = RequestFuncOutput(success=True, prompt_len=100, output_len=50, ttft=0.05,
                          latency=1.0, itl=[0.02, 0.03])
    s0.cached_tokens = 0
    s0.cached_tokens_details = None
    sessions = [[s0]]
    rows, _ = report._round_breakdown(sessions, duration=10.0, max_tokens_per_turn=50)
    assert rows[0][6] is None, "cache hit should be None (N/A) when server flag off"
    assert rows[0][7] == 0.0  # cache observable = 0
    print("PASS: flag-off run shows cache hit = None (N/A)")


def test_round_breakdown_cold_round_zero():
    """运行可测（有 details）时，第 0 轮冷前缀 cache hit = 真实 0.00%。

    场景：sglang cached=0 时省略 prompt_tokens_details
    （服务端 usage_processor._details_if_cached 只在 count>0 时返回）。
    """
    # round 0: 冷前缀，无 details
    r0 = RequestFuncOutput(success=True, prompt_len=100, output_len=50, ttft=0.05,
                         latency=1.0, itl=[0.02, 0.03])
    r0.cached_tokens = 0
    r0.cached_tokens_details = None
    # round 1: 命中 150/200
    r1 = RequestFuncOutput(success=True, prompt_len=100, output_len=50, ttft=0.06,
                         latency=0.9, itl=[0.02, 0.03])
    r1.cached_tokens = 150
    r1.cached_tokens_details = {"cached_tokens": 150}
    r1.prompt_tokens_actual = 200
    sessions = [[r0, r1]]
    rows, _ = report._round_breakdown(sessions, duration=10.0, max_tokens_per_turn=50)
    # round 0: hit = 0/(100+50) = 0.0（真实 0%，可测）
    assert rows[0][6] == 0.0
    # round 1: 服务端 prompt=200，cached=150
    assert rows[1][6] == 150 / 200
    print(f"PASS: cold round0 hit=0.00 (measurable), round1 hit={rows[1][6]:.4f}")


def test_tpm_concurrency_judgments():
    """新指标判断：TPM ≥ 1亿 判定通过；并发仅显示数值（信息项，不判定）。"""
    s = RequestFuncOutput(success=True, prompt_len=5000, output_len=170,
                         ttft=0.05, latency=5.0)
    s.prompt_tokens_actual = 5170  # server-reported
    s.cached_tokens = 1000
    s.cached_tokens_details = {"cached_tokens": 1000}
    args = _mk_args("--accept-steady-tpm", "100000000",
                   "--accept-peak-concurrency", "1000")
    md = report.generate_markdown(
        args, BenchmarkMetrics(completed=2, total_output=90),
        BenchmarkMetrics(completed=2, total_output=90),
        [[s]], [[s]],
        wall_dur=10.0, steady_dur=10.0, steady_sessions=1, ramp_sessions=0,
        terminated=False, termination_reason=None,
        monitor_history=[], cum_429=0,
        backend="sglang", model="test-model",
        target_rps=1.0, start_rps=0.1, output_lens=[170],
        peak_concurrency=1000,
    )
    # 稳态 TPM = (5170+170)/10*60 = 320,400 < 1亿 → 不通过
    tpm_row = [l for l in md.split("\n") if "稳态TPM" in l]
    assert any("不通过" in l for l in tpm_row), f"TPM row should fail: {tpm_row}"
    # 并发为信息项：显示数值，结论列 = "—"，无 通过/不通过
    conc_row = [l for l in md.split("\n") if "并发数" in l]
    assert conc_row and "| — |" in conc_row[0], f"concurrency info row: {conc_row}"
    assert not any("通过" in l or "不通过" in l for l in conc_row), \
        f"info row must not be judged: {conc_row}"
    # 全程表：peak_concurrency = 1000
    assert "| peak_concurrency | 1000 |" in md, "peak_concurrency not in full table"
    # 验收明细：信息项（要求 — / 结论 N/A）
    acc = [l for l in md.split("\n") if l.startswith("| peak_concurrency |") and "—" in l]
    assert acc and "| N/A | 信息项 |" in acc[0], f"info acceptance: {acc}"
    print("PASS: TPM judged, concurrency shown as informational value")


def test_generate_markdown_flag_off_shows_na():
    """服务端未开 --enable-cache-report。

    全程表 cache_hit_rate 应显示 N/A（而非误导性 0.00）；
    验收明细 steady_cache_hit_rate 实际值 None、结论 False（与客户模板一致）。
    """
    s = RequestFuncOutput(success=True, prompt_len=100, output_len=50, ttft=0.05,
                         latency=1.0, itl=[0.02, 0.03], start_time=100.0)
    s.cached_tokens = 0
    s.cached_tokens_details = None
    md = report.generate_markdown(
        _mk_args(), BenchmarkMetrics(completed=1, total_output=50),
        BenchmarkMetrics(completed=1, total_output=50),
        [[s]], [[s]],
        wall_dur=10.0, steady_dur=10.0, steady_sessions=1, ramp_sessions=0,
        terminated=False, termination_reason=None,
        monitor_history=[], cum_429=0,
        backend="sglang", model="test-model",
        target_rps=1.0, start_rps=0.1, output_lens=[50],
    )
    lines = md.split("\n")
    # 全程表：cache_hit_rate | N/A
    full_rows = [ln for ln in lines if ln.startswith("| cache_hit_rate")]
    assert full_rows, "cache_hit_rate row not found"
    assert any("N/A" in ln for ln in full_rows), f"expected N/A: {full_rows}"
    # 全程表：cache_observable_rate | 0.00（诚实显示）
    obs_rows = [ln for ln in lines if ln.startswith("| cache_observable_rate")]
    assert obs_rows and "0.00" in obs_rows[0], f"observable should be 0.00: {obs_rows}"
    # 验收明细：steady_cache_hit_rate 实际 None → 结论 False
    acc_rows = [ln for ln in lines if ln.startswith("| steady_cache_hit_rate")]
    assert acc_rows, "steady_cache_hit_rate acceptance row not found"
    assert "| None |" in acc_rows[0] and "| False |" in acc_rows[0], f"got: {acc_rows[0]}"
    print("PASS: flag-off generates N/A cache hit rate")


def test_request_payload_uses_max_tokens():
    """1007 事故回归：请求体必须发送 max_tokens 字段（而非 max_completion_tokens）。

    网关/旧版 sglang 不识别 OpenAI 新字段 max_completion_tokens，导致 256 上限
    被无视——实际每轮输出 ~1.7K token，decode 预算超 6.8x，验收结果失真。
    """

    async def _run():
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/v1/chat/completions", _handle_chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 18112)
        await site.start()
        try:
            inner_input = RequestFuncInput(
                prompt="hello world",
                api_url="http://127.0.0.1:18112/v1/chat/completions",
                prompt_len=11,
                output_len=4,
                model="test-model",
            lora_name="",
                image_data=None,
            )
            return await async_request_openai_chat_completions_cached(inner_input)
        finally:
            await runner.cleanup()

    out = asyncio.run(_run())
    assert out.success, f"request failed: {out.error}"
    assert _CAPTURED_PAYLOAD.get("max_tokens") == 4, \
        f"payload must use max_tokens: {_CAPTURED_PAYLOAD}"
    assert "max_completion_tokens" not in _CAPTURED_PAYLOAD, \
        f"payload must not use max_completion_tokens: {_CAPTURED_PAYLOAD}"
    print(f"PASS: payload uses max_tokens=4 (fields: {sorted(_CAPTURED_PAYLOAD)})")


def test_verify_output_cap():
    """warmup 上限校验（1007 事故回归）：超 cap 返回告警，正常返回 None。"""
    import bench_multi_turn

    ok = RequestFuncOutput(success=True, prompt_len=100, output_len=256, ttft=0.05,
                           latency=1.0, itl=[0.02])
    bad = RequestFuncOutput(success=True, prompt_len=100, output_len=1745, ttft=0.05,
                           latency=20.0, itl=[0.01])
    # 正常：全部 <= cap（含小容差）
    assert bench_multi_turn.verify_output_cap([ok], 256) is None
    assert bench_multi_turn.verify_output_cap([], 256) is None
    # 超标：告警含 cap 与倍数
    msg = bench_multi_turn.verify_output_cap([ok, bad], 256)
    assert msg and "cap=256" in msg and "6.8" in msg, msg
    # 单对象（非 list）同样处理
    assert bench_multi_turn.verify_output_cap(bad, 256) is not None
    # 失败轮不计入
    failed = RequestFuncOutput(success=False, prompt_len=100, output_len=999, ttft=0,
                               latency=0)
    assert bench_multi_turn.verify_output_cap([failed], 256) is None
    print("PASS: verify_output_cap flags oversize outputs, silent when normal")


def test_ttft_first_token_includes_reasoning():
    """1010 报告回归：TTFT 必须记在首个任意类型 token（含 reasoning_content）。

    推理模型先流式输出思考段再输出正文；若只认 content，纯思考请求
    ttft=0，TPOT=E2E/(n-1) 会把排队+prefill+思考全部摊入（52.8ms，
    真实 decode ≈9.5ms）。思考文本也不得混入 generated_text
    （下一轮上下文只回放正文）。
    """
    global _ACTIVE_CHUNKS
    _ACTIVE_CHUNKS = SSE_CHUNKS_REASONING

    async def _run():
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/v1/chat/completions", _handle_chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 18113)
        await site.start()
        try:
            inner_input = RequestFuncInput(
                prompt="hello world",
                api_url="http://127.0.0.1:18113/v1/chat/completions",
                prompt_len=11,
                output_len=6,
                model="test-model",
            lora_name="",
                image_data=None,
            )
            return await async_request_openai_chat_completions_cached(inner_input)
        finally:
            await runner.cleanup()

    try:
        out = asyncio.run(_run())
        assert out.success, f"request failed: {out.error}"
        # TTFT 已在首个 reasoning token 上记录（非 0，且早于整体延迟）
        assert 0 < out.ttft < out.latency, f"ttft={out.ttft}, latency={out.latency}"
        # 思考段不进入 generated_text（供下一轮回放的上下文）
        assert out.generated_text == "The answer is 42.", repr(out.generated_text)
        # 后续 token 记录 itl（4 个 delta token → 3 个间隔）
        assert len(out.itl) == 3, f"expected 3 itl entries, got {out.itl}"
        assert all(x > 0 for x in out.itl), out.itl
        assert out.output_len == 6
        print(f"PASS: TTFT={out.ttft * 1000:.1f}ms recorded at first reasoning token, "
              f"reasoning excluded from generated_text")
    finally:
        _ACTIVE_CHUNKS = SSE_CHUNKS


def test_reasoning_effort_in_payload():
    """--reasoning-effort low：extra_request_body 携带的 reasoning_effort 必须进入 payload。"""
    global _ACTIVE_CHUNKS
    _ACTIVE_CHUNKS = SSE_CHUNKS

    async def _run():
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/v1/chat/completions", _handle_chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 18114)
        await site.start()
        try:
            inner_input = RequestFuncInput(
                prompt="hello world",
                api_url="http://127.0.0.1:18114/v1/chat/completions",
                prompt_len=11,
                output_len=4,
                model="test-model",
                lora_name="",
                image_data=None,
                extra_request_body={"reasoning_effort": "low"},
            )
            return await async_request_openai_chat_completions_cached(inner_input)
        finally:
            await runner.cleanup()

    try:
        out = asyncio.run(_run())
        assert out.success, f"request failed: {out.error}"
        assert _CAPTURED_PAYLOAD.get("reasoning_effort") == "low", \
            f"payload missing reasoning_effort: {_CAPTURED_PAYLOAD}"
        print("PASS: payload carries reasoning_effort=low via extra_request_body")
    finally:
        _ACTIVE_CHUNKS = SSE_CHUNKS


def test_routing_key_header_sent():
    """会话亲和：RequestFuncInput.routing_key 必须以 X-SMG-Routing-Key 请求头发出。

    1010 报告 round1 命中率仅 34.1%（≈1/3）：多节点随机路由稀释了 prefix
    cache 命中；网关需要该头做会话粘性路由。
    """
    global _ACTIVE_CHUNKS
    _ACTIVE_CHUNKS = SSE_CHUNKS

    async def _run():
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/v1/chat/completions", _handle_chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 18115)
        await site.start()
        try:
            inner_input = RequestFuncInput(
                prompt="hello world",
                api_url="http://127.0.0.1:18115/v1/chat/completions",
                prompt_len=11,
                output_len=4,
                model="test-model",
                lora_name="",
                image_data=None,
                routing_key="bench-session-0",
            )
            return await async_request_openai_chat_completions_cached(inner_input)
        finally:
            await runner.cleanup()

    try:
        out = asyncio.run(_run())
        assert out.success, f"request failed: {out.error}"
        assert _CAPTURED_HEADERS.get("X-SMG-Routing-Key") == "bench-session-0", \
            f"routing header missing: {_CAPTURED_HEADERS}"
        print("PASS: X-SMG-Routing-Key header sent for session affinity")
    finally:
        _ACTIVE_CHUNKS = SSE_CHUNKS


def test_make_extra_request_body_reasoning():
    """make_extra_request_body：--reasoning-effort 设置时进入 extra body，未设置时不携带。"""
    import bench_multi_turn

    args_on = types.SimpleNamespace(
        temperature=0.0, top_p=1.0, ignore_eos=False, reasoning_effort="low")
    body = bench_multi_turn.make_extra_request_body(args_on)
    assert body.get("reasoning_effort") == "low", body

    args_off = types.SimpleNamespace(
        temperature=0.0, top_p=1.0, ignore_eos=False, reasoning_effort=None)
    body = bench_multi_turn.make_extra_request_body(args_off)
    assert "reasoning_effort" not in body, body
    print("PASS: make_extra_request_body forwards reasoning_effort when set")


def test_report_does_not_leak_reasoning_effort():
    """报告与 JSONL 不得出现 reasoning_effort / reasoning-effort。

    generate_markdown 只使用 args.max_tokens_per_turn 与 args.accept_*，
    不转储 extra_request_body / argv / config；bench_multi_turn.py 的 JSONL
    result 同样只有数据集/调度/指标字段。
    """
    s = RequestFuncOutput(success=True, prompt_len=100, output_len=50, ttft=0.05,
                          latency=1.0, itl=[0.02, 0.03], start_time=100.0)
    s.prompt_tokens_actual = 120
    s.cached_tokens = 60
    s.cached_tokens_details = {"cached_tokens": 60}

    md = report.generate_markdown(
        _mk_args("--reasoning-effort", "low"),
        BenchmarkMetrics(completed=1, total_output=50),
        BenchmarkMetrics(completed=1, total_output=50),
        [[s]], [[s]],
        wall_dur=10.0, steady_dur=10.0, steady_sessions=1, ramp_sessions=0,
        terminated=False, termination_reason=None,
        monitor_history=[], cum_429=0,
        backend="sglang", model="test-model",
        target_rps=1.0, start_rps=0.1, output_lens=[50],
    )
    assert "reasoning_effort" not in md, "report must not mention reasoning_effort"
    assert "reasoning-effort" not in md, "report must not mention reasoning-effort"
    assert "effort" not in md, "report must not mention effort at all"
    print("PASS: report contains no reasoning-effort information")


def test_wrapper_propagates_routing_and_effort():
    """端到端：经过安装版 wrapper 的请求仍携带 X-SMG-Routing-Key 头与 reasoning_effort。

    安装版 sglang wrapper 只透传它认识的字段，丢弃 extra_request_body /
    routing_key（1008_2 报告确认：亲和与 reasoning_effort 均未生效），
    request_client 兜底补偿：
    - routing_key 从首条消息内容哈希推导——同一会话各轮首条消息相同 →
      同一 key → 网关亲和路由依然成立
    - reasoning_effort 从 serving.args 读取（全测试同值）

    同时验证 wrapper 逐轮累积对话历史（round1+ 请求携带前轮内容 →
    prefix cache 命中前提），且不触碰 start_time（1008_4 根因）与
    lora_name/image_data（1008_5 根因，服务器版必填）。
    """
    global _ACTIVE_CHUNKS, _CAPTURED_HEADERS, _CAPTURED_PAYLOAD, _ROUTING_KEYS
    import bench_multi_turn

    _ACTIVE_CHUNKS = SSE_CHUNKS
    _CAPTURED_HEADERS = {}
    _CAPTURED_PAYLOAD = {}
    _ROUTING_KEYS = []

    async def _run():
        from aiohttp import web

        app = web.Application()
        app.router.add_post("/v1/chat/completions", _handle_chat_completions)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 18116)
        await site.start()
        try:
            wrapped = bench_multi_turn.wrap_multi_turn_request_func(
                async_request_openai_chat_completions_cached, backend="sglang"
            )
            # 安装版 wrapper 未见过这些自定义字段 —— outer 有无均可
            outer_input = types.SimpleNamespace(
                prompt=["turn 1", "turn 2"],
                api_url="http://127.0.0.1:18116/v1/chat/completions",
                prompt_len=11,
                output_len=6,
                model="test-model",
            )
            return await wrapped(outer_input, pbar=None)
        finally:
            await runner.cleanup()

    # 安装版 wrapper 丢弃 extra_request_body → request_client 从 serving.args 兜底
    _serving_ns.reasoning_effort = "low"
    try:
        outs = asyncio.run(_run())
        assert len(outs) == 2, f"expected 2 results, got {len(outs)}"
        assert _CAPTURED_PAYLOAD.get("reasoning_effort") == "low", \
            f"payload missing reasoning_effort: {_CAPTURED_PAYLOAD}"
        # 内容哈希路由：两轮请求的 X-SMG-Routing-Key 相同（同一会话）
        assert len(_ROUTING_KEYS) == 2, \
            f"expected 2 captured routing keys, got {_ROUTING_KEYS}"
        assert all(k and k.startswith("bench-") for k in _ROUTING_KEYS), \
            f"content-hash routing key missing: {_ROUTING_KEYS}"
        assert _ROUTING_KEYS[0] == _ROUTING_KEYS[1], \
            f"routing key must be stable across rounds: {_ROUTING_KEYS}"
        # 历史累积：最后一轮请求的 messages 含第一轮内容 + 助手回复
        msgs = _CAPTURED_PAYLOAD.get("messages", [])
        texts = [m.get("content") for m in msgs]
        assert "turn 1" in texts, f"history not accumulated: {texts}"
        assert "turn 2" in texts, f"latest turn missing: {texts}"
        assert any(m.get("role") == "assistant" for m in msgs), \
            f"assistant reply not in history: {msgs}"
        # 每轮 output 的 start_time 由 request_client 设置（真实请求时间），
        # wrapper 不得覆盖为 0（稳态窗口过滤依赖它）
        assert all(getattr(o, "start_time", None) for o in outs), \
            "wrapper must not zero out output.start_time"
        print("PASS: wrapper path keeps affinity header, reasoning_effort, "
              "and multi-turn history")
    finally:
        _serving_ns.reasoning_effort = None
        _ACTIVE_CHUNKS = SSE_CHUNKS
        _CAPTURED_HEADERS = {}
        _CAPTURED_PAYLOAD = {}
        _ROUTING_KEYS = []


if __name__ == "__main__":
    test_fork_captures_cached_tokens()
    test_round_breakdown_uses_server_values()
    test_round_breakdown_no_cache_flag()
    test_round_breakdown_cold_round_zero()
    test_generate_markdown_cache_metrics()
    test_generate_markdown_flag_off_shows_na()
    test_tpm_concurrency_judgments()
    test_request_payload_uses_max_tokens()
    test_verify_output_cap()
    test_ttft_first_token_includes_reasoning()
    test_reasoning_effort_in_payload()
    test_routing_key_header_sent()
    test_make_extra_request_body_reasoning()
    test_report_does_not_leak_reasoning_effort()
    test_wrapper_propagates_routing_and_effort()
    print("\nAll smoke tests passed.")

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import requests
from tqdm.asyncio import tqdm

import sglang.benchmark.serving as serving
from sglang.benchmark.serving import (
    RequestFuncInput,
    calculate_metrics,
    flush_server_cache,
    wait_for_endpoint,
    wrap_multi_turn_request_func,
)
from sglang.benchmark.utils import get_tokenizer

from config import build_parser, compute_rps, load_env, make_serving_namespace, _preparse_env_file
from request_client import async_request_openai_chat_completions_cached
from monitor import DegradationMonitor
from ramp_scheduler import get_ramp_request
from report import generate_markdown
from sharegpt_multiturn import load_sharegpt_multiturn


def resolve_model(base_url, fallback="default"):
    try:
        resp = requests.get(base_url.rstrip("/") + "/v1/models", timeout=10)
        data = resp.json()
        items = data.get("data") or []
        if items:
            return items[0].get("id", fallback)
    except Exception as e:
        print(f"[warn] auto-detect model failed: {e}", file=sys.stderr)
    return fallback


def build_api_url(base_url):
    return base_url.rstrip("/") + "/v1/chat/completions"


def make_extra_request_body(args):
    body = {"temperature": args.temperature}
    if args.top_p < 1.0:
        body["top_p"] = args.top_p
    if args.ignore_eos:
        body["ignore_eos"] = True
    if getattr(args, "reasoning_effort", None):
        body["reasoning_effort"] = args.reasoning_effort
    return body


def verify_output_cap(results, cap, tolerance=8):
    """校验 warmup 结果是否遵守 max_tokens 上限（1007 事故回归）。

    事故：网关忽略生成上限字段，每轮实际输出 ~1.7K token（计划 256），
    decode 预算超 6.8x，TPM/TPOT/轮周期全部失真。超限时立即告警，
    避免烧完整场才发现数据不可用。

    返回告警文案（str）；正常时返回 None。
    """
    if cap <= 0:
        return None
    if not isinstance(results, list):
        results = [results]
    oversize = []
    for o in results:
        if o is None or not getattr(o, "success", False):
            continue
        ol = getattr(o, "output_len", 0) or 0
        if ol > cap + tolerance:
            oversize.append(ol)
    if not oversize:
        return None
    worst = max(oversize)
    return (
        f"服务器疑似未遵守 max_tokens 上限：cap={cap}，warmup 超标轮输出 "
        f"{oversize}（最坏 {worst}，为上限的 {worst / cap:.1f} 倍）。"
        "输出长度失控会使 decode 预算/轮周期/TPM 全部失真，验收结果不可用。"
        f"请先单发一条 max_tokens={cap} 的请求，检查 usage.completion_tokens "
        "是否 ≤ 上限，并排查网关是否透传该字段。"
    )


def preflight_capacity(args, target_rps, loaded):
    if args.sustain_seconds is None:
        return
    # 爬坡期速率从 0 线性升至目标，平均为一半；×1.2 覆盖泊松到达的随机波动
    window = args.ramp_seconds / 2.0 + args.sustain_seconds
    needed = window * target_rps * 1.2
    if loaded < needed:
        print(
            f"[warn] 数据集容量可能不足：爬坡+稳态窗口 {window:.0f}s(等效满速) @ {target_rps:.3f} rps "
            f"约需 {needed:.0f} 会话，实际 {loaded}；测试可能在稳态结束前耗尽会话。"
            f"请增大 --num-sessions 或缩短窗口。",
            file=sys.stderr,
        )


async def run_benchmark(args):
    serving.args = make_serving_namespace(args)

    if args.ready_check_timeout_sec > 0:
        ok = wait_for_endpoint(args.base_url.rstrip("/") + "/health", args.ready_check_timeout_sec)
        if not ok:
            print("[error] server not ready", file=sys.stderr)
            return 1

    tokenizer_id = args.tokenizer or args.model
    if not tokenizer_id:
        print("[error] need --tokenizer or --model", file=sys.stderr)
        return 1
    tokenizer = get_tokenizer(tokenizer_id)

    input_requests = load_sharegpt_multiturn(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        num_sessions=args.num_sessions,
        num_turns=args.num_turns,
        max_tokens_per_turn=args.max_tokens_per_turn,
        system_prompt_len=args.system_prompt_len,
        context_len=args.context_len,
        min_turns=args.min_turns,
        seed=args.seed,
        apply_chat_template=args.apply_chat_template,
        num_shared_prefixes=args.num_shared_prefixes,
    )
    if not input_requests:
        print("[error] no sessions loaded", file=sys.stderr)
        return 1

    if args.avg_tokens_per_request <= 0:
        total_est = 0.0
        for row in input_requests:
            n = len(row.prompt)
            total_est += n * (row.prompt_len + row.output_len) + \
                row.output_len * n * (n - 1) / 2.0
        args.avg_tokens_per_request = total_est / len(input_requests)
        print(
            f"[auto] avg_tokens_per_request = {args.avg_tokens_per_request:.0f} "
            f"(estimated from {len(input_requests)} sessions, "
            f"prompt_len~{input_requests[0].prompt_len}, turns~"
            f"{len(input_requests[0].prompt)})"
        )

    base_url = args.base_url.rstrip("/")
    api_url = build_api_url(args.base_url)
    backend = args.backend
    model = args.model or resolve_model(base_url)

    safe_model = re.sub(r'[^\w\-.]', '_', model or "unknown")
    subdir_name = f"{safe_model}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_subdir = os.path.join(args.output_dir, subdir_name)
    os.makedirs(output_subdir, exist_ok=True)
    args._output_subdir = output_subdir
    print(f"[output] -> {output_subdir}")

    start_rps, target_rps = compute_rps(args)
    preflight_capacity(args, target_rps, len(input_requests))

    request_func = wrap_multi_turn_request_func(
        async_request_openai_chat_completions_cached, backend=backend
    )

    sem = asyncio.Semaphore(args.max_concurrency if args.max_concurrency > 0 else 10 ** 9)

    # 并发跟踪：信号量占用数 = 实际同时在执行的会话数（每个会话同一时刻只有 1 个在途
    # HTTP 请求）。peak_concurrency 即真实峰值并发（服务端视角）。
    current_concurrency = 0
    peak_concurrency = 0

    async def limited(rfi, pbar):
        nonlocal current_concurrency, peak_concurrency
        async with sem:
            current_concurrency += 1
            if current_concurrency > peak_concurrency:
                peak_concurrency = current_concurrency
            try:
                return await request_func(rfi, pbar=pbar)
            finally:
                current_concurrency -= 1

    extra_body = make_extra_request_body(args)

    if not args.no_warmup:
        warm = input_requests[0]
        warm_input = RequestFuncInput(
            prompt=warm.prompt, api_url=api_url, prompt_len=warm.prompt_len,
            output_len=warm.output_len, model=model, lora_name="",
            image_data=warm.image_data, extra_request_body=extra_body,
        )
        print("[warmup] one multi-turn session ...")
        try:
            warm_result = await request_func(warm_input, pbar=None)
        except Exception as e:
            print(f"[warn] warmup failed: {e}", file=sys.stderr)
        else:
            cap_warning = verify_output_cap(warm_result, args.max_tokens_per_turn)
            if cap_warning:
                print(f"[warn] {cap_warning}", file=sys.stderr)
        if not args.no_flush_cache:
            try:
                flush_server_cache(base_url, backend)
            except Exception as e:
                print(f"[warn] flush cache failed: {e}", file=sys.stderr)
        await asyncio.sleep(1.0)

    print(
        f"[ramp] start_rps={start_rps:.4f} target_rps={target_rps:.4f} "
        f"ramp_seconds={args.ramp_seconds} sustain_seconds={args.sustain_seconds}"
    )

    bench_start = time.perf_counter()
    ramp_end_time = bench_start + args.ramp_seconds
    stop_event = asyncio.Event()
    monitor = None if args.disable_monitor else DegradationMonitor(
        ramp_end_time=ramp_end_time,
        bench_start=bench_start,
        start_rps=start_rps,
        target_rps=target_rps,
        ramp_seconds=args.ramp_seconds,
        max_concurrency=args.max_concurrency,
        interval=args.monitor_interval,
        window=args.monitor_window,
        max_error_rate=args.max_error_rate,
        throughput_drop_ratio=args.throughput_drop_ratio,
        max_ttft_p99_ms=args.max_ttft_p99_ms,
        baseline_warmup_seconds=args.baseline_warmup_seconds,
    )

    inflight = 0

    async def _run_one(rfi, pbar):
        nonlocal inflight
        inflight += 1
        if monitor:
            monitor.note_concurrency(inflight)
        try:
            out = await limited(rfi, pbar)
        except Exception:
            out = None
        finally:
            inflight -= 1
        if monitor is not None:
            if isinstance(out, list):
                for o in out:
                    monitor.feed(o)
            else:
                monitor.feed(out)
        return out

    async def watchdog():
        while not stop_event.is_set():
            await asyncio.sleep(args.monitor_interval)
            monitor.snapshot()
            reason = monitor.check()
            if reason:
                print(f"\n[terminate] 衰减信号: {reason}", file=sys.stderr)
                stop_event.set()
                return

    watchdog_task = asyncio.create_task(watchdog()) if monitor else None

    pbar = tqdm(total=len(input_requests), desc="sessions")
    tasks = []

    async def terminate_listener():
        # stop_event 触发后立即取消所有未完成任务。
        # 调度循环只在循环体内检查 stop_event；监控若在调度结束后才触发
        # （过载场景常态），旧逻辑无人取消任务，排空可拖数十小时。
        await stop_event.wait()
        for t in tasks:
            if not t.done():
                t.cancel()

    listener_task = asyncio.create_task(terminate_listener())

    session_idx = 0
    async for req in get_ramp_request(
        input_requests, start_rps, target_rps, args.ramp_seconds,
        args.sustain_seconds, seed=args.seed, stop_event=stop_event,
    ):
        if stop_event.is_set():
            break
        # 会话亲和路由：同一会话各轮带相同 routing_key，网关将其固定到同一
        # 后端节点——prefix cache 命中的前提。1010 报告中 round1 命中率仅
        # 34.1%（≈1/3）：多节点随机路由下，前一轮的前缀只在 1/N 概率的
        # 同节点请求上可命中；亲和后 round1+ 理论命中 ~96%+。
        affinity_key = (
            None if args.no_session_affinity else f"bench-session-{session_idx}"
        )
        session_idx += 1
        rfi = RequestFuncInput(
            prompt=req.prompt, api_url=api_url, prompt_len=req.prompt_len,
            output_len=req.output_len, model=model, lora_name="",
            image_data=req.image_data, extra_request_body=extra_body,
            routing_key=affinity_key,
        )
        tasks.append(asyncio.create_task(_run_one(rfi, pbar)))

    # 排空在途会话，drain_timeout 后强制取消剩余任务，
    # 防止服务器挂起/过载时测试无限等待
    if tasks:
        await asyncio.wait(tasks, timeout=args.drain_timeout)
        for t in tasks:
            if not t.done():
                t.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    listener_task.cancel()
    if watchdog_task:
        watchdog_task.cancel()
    pbar.close()
    wall_dur = time.perf_counter() - bench_start

    all_session_outputs = []
    for r in results:
        if isinstance(r, (Exception, asyncio.CancelledError)) or r is None:
            continue
        all_session_outputs.append(r)

    steady_session_outputs = [
        o for o in all_session_outputs
        if o and (o[0].start_time if isinstance(o, list) else o.start_time) >= ramp_end_time
    ]
    ramp_session_count = len(all_session_outputs) - len(steady_session_outputs)

    all_flat = [x for o in all_session_outputs for x in (o if isinstance(o, list) else [o]) if x is not None]
    steady_flat = [x for o in steady_session_outputs for x in (o if isinstance(o, list) else [o]) if x is not None]

    if steady_flat:
        steady_start = min(o.start_time for o in steady_flat if o.start_time > 0)
        _ok_steady = [o for o in steady_flat if o.success and o.start_time > 0]
        if _ok_steady:
            steady_end = max(o.start_time + o.latency for o in _ok_steady)
            steady_dur = steady_end - steady_start
        else:
            steady_dur = 0.0
    else:
        steady_dur = 0.0

    terminated = stop_event.is_set()
    termination_reason = monitor.stop_reason if monitor else None
    monitor_history = monitor.history_dicts() if monitor else []
    cum_429 = monitor.cum_429 if monitor else 0

    metrics_full, output_lens_full = calculate_metrics(
        None, all_flat, wall_dur, tokenizer, backend
    )
    if steady_flat:
        metrics_steady, output_lens_steady = calculate_metrics(
            None, steady_flat, steady_dur, tokenizer, backend
        )
    else:
        from sglang.benchmark.serving import BenchmarkMetrics
        metrics_steady = BenchmarkMetrics(
            completed=0, total_input=0, total_input_text=0, total_input_vision=0,
            total_output=0, total_output_retokenized=0,
            request_throughput=0, input_throughput=0, output_throughput=0,
            output_throughput_retokenized=0, total_throughput=0, total_throughput_retokenized=0,
            mean_ttft_ms=0, median_ttft_ms=0, std_ttft_ms=0,
            p90_ttft_ms=0, p95_ttft_ms=0, p99_ttft_ms=0,
            mean_tpot_ms=0, median_tpot_ms=0, std_tpot_ms=0,
            p90_tpot_ms=0, p95_tpot_ms=0, p99_tpot_ms=0,
            mean_itl_ms=0, median_itl_ms=0, std_itl_ms=0,
            p90_itl_ms=0, p95_itl_ms=0, p99_itl_ms=0, max_itl_ms=0,
            mean_e2e_latency_ms=0, median_e2e_latency_ms=0, std_e2e_latency_ms=0,
            p90_e2e_latency_ms=0, p95_e2e_latency_ms=0, p99_e2e_latency_ms=0,
            concurrency=0, max_concurrent_requests=0,
        )
        output_lens_steady = []

    print_console_summary(
        metrics_full, metrics_steady, wall_dur, steady_dur,
        len(all_session_outputs), len(steady_session_outputs),
        ramp_session_count, args.num_turns, terminated, termination_reason, cum_429,
        peak_concurrency, args.max_concurrency,
    )
    write_jsonl(
        args, metrics_full, metrics_steady, all_flat, steady_flat,
        output_lens_full, output_lens_steady, backend, model, target_rps,
        wall_dur, steady_dur, len(steady_session_outputs), ramp_session_count,
        terminated, termination_reason, monitor_history, cum_429,
        peak_concurrency,
    )
    write_markdown_report(
        args, metrics_full, metrics_steady,
        all_session_outputs, steady_session_outputs,
        wall_dur, steady_dur, len(steady_session_outputs), ramp_session_count,
        terminated, termination_reason, monitor_history, cum_429,
        backend, model, target_rps, start_rps, output_lens_full,
        peak_concurrency,
    )
    return 0


def print_console_summary(metrics_full, metrics_steady, wall_dur, steady_dur,
                          total_sessions, steady_sessions, ramp_sessions,
                          num_turns, terminated, termination_reason, cum_429,
                          peak_concurrency=0, max_concurrency=0):
    print("\n" + "=" * 70)
    print("Multi-turn long-context benchmark results")
    print("=" * 70)
    print(f"  [全程] duration={wall_dur:.3f}s  sessions={total_sessions}  "
          f"completed={metrics_full.completed}  rps={metrics_full.request_throughput:.4f}")
    print(f"         output_tps={metrics_full.output_throughput:.2f}  "
          f"total_tps={metrics_full.total_throughput:.2f}")
    print(f"         peak_concurrency={peak_concurrency}"
          + (f" (cap={max_concurrency})" if max_concurrency > 0 else " (uncapped)"))
    print(f"  [稳态] duration={steady_dur:.3f}s  sessions={steady_sessions}  "
          f"ramp_excluded={ramp_sessions}")
    print(f"         rps={metrics_steady.request_throughput:.4f}  "
          f"output_tps={metrics_steady.output_throughput:.2f}")
    if steady_sessions > 0:
        print(f"  [稳态延迟] TTFT p50={metrics_steady.median_ttft_ms:.2f} "
              f"p99={metrics_steady.p99_ttft_ms:.2f} ms | "
              f"TPOT p50={metrics_steady.median_tpot_ms:.2f} "
              f"p99={metrics_steady.p99_tpot_ms:.2f} ms")
    if terminated:
        print(f"  [终止] {termination_reason}")
    if cum_429 > 0:
        print(f"  [429] {cum_429} 次")
    print("=" * 70)


def write_jsonl(args, metrics_full, metrics_steady, all_flat, steady_flat,
               output_lens_full, output_lens_steady, backend, model, target_rps,
               wall_dur, steady_dur, steady_sessions, ramp_sessions,
               terminated, termination_reason, monitor_history, cum_429,
               peak_concurrency=0):
    start_rps, _ = compute_rps(args)
    result = {
        "tag": args.tag,
        "backend": backend,
        "model": model,
        "dataset_name": "sharegpt-multiturn",
        "num_sessions": args.num_sessions,
        "num_turns": args.num_turns,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "system_prompt_len": args.system_prompt_len,
        "ramp": {
            "start_rps": start_rps,
            "target_rps": target_rps,
            "ramp_seconds": args.ramp_seconds,
            "sustain_seconds": args.sustain_seconds,
        },
        "wall_duration": round(wall_dur, 3),
        "steady_duration": round(steady_dur, 3),
        "steady_sessions": steady_sessions,
        "ramp_sessions_excluded": ramp_sessions,
        "terminated": terminated,
        "termination_reason": termination_reason,
        "cum_429": cum_429,
        "peak_concurrency": peak_concurrency,
        "monitor_history": monitor_history,
        "full": {
            "completed": metrics_full.completed,
            "total_output_tokens": metrics_full.total_output,
            "request_throughput": metrics_full.request_throughput,
            "output_throughput": metrics_full.output_throughput,
            "total_throughput": metrics_full.total_throughput,
            "ttft_ms": {"mean": metrics_full.mean_ttft_ms, "p50": metrics_full.median_ttft_ms,
                        "p90": metrics_full.p90_ttft_ms, "p95": metrics_full.p95_ttft_ms,
                        "p99": metrics_full.p99_ttft_ms},
            "tpot_ms": {"mean": metrics_full.mean_tpot_ms, "p50": metrics_full.median_tpot_ms,
                        "p99": metrics_full.p99_tpot_ms},
            "itl_ms": {"mean": metrics_full.mean_itl_ms, "p99": metrics_full.p99_itl_ms,
                       "max": metrics_full.max_itl_ms},
            "e2e_ms": {"mean": metrics_full.mean_e2e_latency_ms, "p50": metrics_full.median_e2e_latency_ms,
                       "p99": metrics_full.p99_e2e_latency_ms},
            "concurrency": metrics_full.concurrency,
            "max_concurrent_requests": metrics_full.max_concurrent_requests,
        },
        "steady": {
            "completed": metrics_steady.completed,
            "total_output_tokens": metrics_steady.total_output,
            "request_throughput": metrics_steady.request_throughput,
            "output_throughput": metrics_steady.output_throughput,
            "total_throughput": metrics_steady.total_throughput,
            "ttft_ms": {"mean": metrics_steady.mean_ttft_ms, "p50": metrics_steady.median_ttft_ms,
                        "p90": metrics_steady.p90_ttft_ms, "p95": metrics_steady.p95_ttft_ms,
                        "p99": metrics_steady.p99_ttft_ms},
            "tpot_ms": {"mean": metrics_steady.mean_tpot_ms, "p50": metrics_steady.median_tpot_ms,
                        "p99": metrics_steady.p99_tpot_ms},
            "concurrency": metrics_steady.concurrency,
            "max_concurrent_requests": metrics_steady.max_concurrent_requests,
        },
    }
    if args.output_details:
        result["details"] = {
            "input_lens": [o.prompt_len for o in all_flat],
            "output_lens": output_lens_full,
            "ttfts": [o.ttft for o in all_flat],
            "itls": [o.itl for o in all_flat],
            "errors": [o.error for o in all_flat],
            "cached_tokens": [getattr(o, "cached_tokens", 0) or 0 for o in all_flat],
            "prompt_tokens_actual": [getattr(o, "prompt_tokens_actual", 0) or 0 for o in all_flat],
        }

    out_file = args.output_file
    if not out_file:
        out_file = f"multi_turn_{backend}_{args.num_sessions}s_{args.num_turns}t.jsonl"
    out_file = os.path.join(getattr(args, "_output_subdir", ""), out_file)
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"[result] JSONL -> {out_file}")


def write_markdown_report(args, metrics_full, metrics_steady,
                          all_session_outputs, steady_session_outputs,
                          wall_dur, steady_dur, steady_sessions, ramp_sessions,
                          terminated, termination_reason, monitor_history, cum_429,
                          backend, model, target_rps, start_rps, output_lens,
                          peak_concurrency=0):
    md = generate_markdown(
        args, metrics_full, metrics_steady,
        all_session_outputs, steady_session_outputs,
        wall_dur, steady_dur, steady_sessions, ramp_sessions,
        terminated, termination_reason, monitor_history, cum_429,
        backend, model, target_rps, start_rps, output_lens,
        peak_concurrency,
    )
    base = args.report_md
    if not base:
        base = "report.md"
    base = os.path.join(getattr(args, "_output_subdir", ""), base)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    with open(base, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[result] Markdown report -> {base}")


def main():
    env_file = _preparse_env_file()
    env = load_env(env_file)
    parser = build_parser(env)
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url is required (可通过 configs/.env 或 --base-url 提供)")
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    np.random.seed(args.seed)
    return asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    sys.exit(main())

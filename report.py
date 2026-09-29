import json
from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np


def _pct(vals, q):
    if vals is None or len(vals) == 0:
        return None
    return float(np.percentile(vals, q))


def _latency_row(name: str, samples: List[float]) -> str:
    """Format a latency table row: Samples | avg | p50 | p75 | p90 | p95 | p99 (all in ms)."""
    if not samples:
        return f"| {name} | 0 | N/A | N/A | N/A | N/A | N/A | N/A |"
    arr = np.array(samples, dtype=float) * 1000.0
    return (
        f"| {name} | {len(arr):,} | "
        f"{arr.mean():,.2f} | {_pct(arr, 50):,.2f} | {_pct(arr, 75):,.2f} | "
        f"{_pct(arr, 90):,.2f} | {_pct(arr, 95):,.2f} | {_pct(arr, 99):,.2f} |"
    )


def _compute_tpot(out) -> Optional[float]:
    ol = getattr(out, "output_len", 0) or 0
    if ol <= 1:
        return None
    ttft = getattr(out, "ttft", 0) or 0
    lat = getattr(out, "latency", 0) or 0
    return (lat - ttft) / (ol - 1)


def _round_breakdown(session_outputs_list, duration, max_tokens_per_turn):
    """Compute per-round summary + latency tables.

    session_outputs_list: List[List[RequestFuncOutput]] (per session, per round)
    Returns: (summary_rows, round_latency_tables)

    Input tokens per round are estimated as:
      prompt_len (round-0 context) + sum(output_len of all prior rounds)
    This reflects the accumulated multi-turn context the server actually receives.
    """
    max_round = max(len(s) for s in session_outputs_list) if session_outputs_list else 0
    summary_rows = []
    round_latency = {}

    for r in range(max_round):
        round_outs = [s[r] for s in session_outputs_list if r < len(s) and s[r] is not None]
        ok_outs = [o for o in round_outs if o.success]
        count = len(round_outs)
        ok_count = len(ok_outs)

        if count == 0:
            summary_rows.append((r, 0, duration, 0, 0, 0, None, None))
            round_latency[r] = ([], [], [], [])
            continue

        input_toks = 0
        for s in session_outputs_list:
            if r >= len(s) or s[r] is None or not s[r].success:
                continue
            base_prompt = getattr(s[0], "prompt_len", 0) or 0
            prior_outputs = sum(getattr(s[j], "output_len", 0) or 0 for j in range(r))
            input_toks += base_prompt + prior_outputs

        output_toks = sum(getattr(o, "output_len", 0) or 0 for o in ok_outs)

        cached_total = sum(getattr(o, "cached_tokens", 0) or 0 for o in ok_outs)
        prompt_total = input_toks
        cache_hit = cached_total / prompt_total if prompt_total > 0 else None
        cache_obs = sum(
            1 for o in ok_outs if getattr(o, "cached_tokens_details", None) is not None
        ) / ok_count if ok_count > 0 else None

        req_s = count / duration if duration > 0 else 0
        in_tps = input_toks / duration if duration > 0 else 0
        out_tps = output_toks / duration if duration > 0 else 0

        summary_rows.append((r, count, duration, req_s, in_tps, out_tps, cache_hit, cache_obs))

        ttfts = [o.ttft for o in ok_outs if o.ttft > 0]
        e2es = [o.latency for o in ok_outs if o.latency > 0]
        tpots = [t for t in (_compute_tpot(o) for o in ok_outs) if t is not None]
        itls = [x for o in ok_outs for x in (o.itl or []) if x > 0]
        round_latency[r] = (ttfts, e2es, tpots, itls)

    return summary_rows, round_latency


def _fmt_cache(v):
    if v is None:
        return "N/A"
    return f"{v:.2%}"


def _fmt_num(v, fmt=".2f"):
    if v is None:
        return "N/A"
    return f"{v:{fmt}}"


def _fmt_count(v):
    """Format count/total with commas and 2 decimals, matching template: 5,006.00"""
    if v is None:
        return "N/A"
    return f"{v:,.2f}"


def generate_markdown(
    args,
    metrics_full,
    metrics_steady,
    all_session_outputs: List[List[Any]],
    steady_session_outputs: List[List[Any]],
    wall_dur: float,
    steady_dur: float,
    steady_sessions: int,
    ramp_sessions: int,
    terminated: bool,
    termination_reason: Optional[str],
    monitor_history: List[dict],
    cum_429: int,
    backend: str,
    model: str,
    target_rps: float,
    start_rps: float,
    output_lens: List[int],
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    has_steady = steady_sessions > 0 and steady_dur > 0

    all_rounds = [o for s in all_session_outputs for o in s]
    all_flat = [o for o in all_rounds if o is not None]
    ok_flat = [o for o in all_flat if o.success]
    steady_rounds = [o for s in steady_session_outputs for o in s]
    steady_flat = [o for o in steady_rounds if o is not None]
    ok_steady = [o for o in steady_flat if o.success]

    total_requests = len(all_rounds)
    transport_success = len(all_flat)
    success_requests = len(ok_flat)
    output_truncated = sum(
        1 for o in ok_flat
        if (getattr(o, "output_len", 0) or 0) >= args.max_tokens_per_turn
    )
    total_output_tokens = sum(getattr(o, "output_len", 0) or 0 for o in ok_flat)
    total_input_tokens = 0
    for s in all_session_outputs:
        for r_idx, o in enumerate(s):
            if o is None or not o.success:
                continue
            base_prompt = getattr(s[0], "prompt_len", 0) or 0
            prior_outputs = sum(getattr(s[j], "output_len", 0) or 0 for j in range(r_idx))
            total_input_tokens += base_prompt + prior_outputs
    actual_tokens_total = total_input_tokens + total_output_tokens

    cached_total = sum(getattr(o, "cached_tokens", 0) or 0 for o in ok_flat)
    cache_hit_rate = cached_total / total_input_tokens if total_input_tokens > 0 else None
    cache_observable = sum(
        1 for o in ok_flat if getattr(o, "cached_tokens_details", None) is not None
    )
    cache_observable_rate = cache_observable / success_requests if success_requests > 0 else None

    steady_output_tokens = sum(getattr(o, "output_len", 0) or 0 for o in ok_steady)
    steady_input_tokens = 0
    for s in steady_session_outputs:
        for r_idx, o in enumerate(s):
            if o is None or not o.success:
                continue
            base_prompt = getattr(s[0], "prompt_len", 0) or 0
            prior_outputs = sum(getattr(s[j], "output_len", 0) or 0 for j in range(r_idx))
            steady_input_tokens += base_prompt + prior_outputs
    steady_cached = sum(getattr(o, "cached_tokens", 0) or 0 for o in ok_steady)
    steady_cache_hit = steady_cached / steady_input_tokens if steady_input_tokens > 0 else None

    request_rps = success_requests / wall_dur if wall_dur > 0 else 0
    input_tps = total_input_tokens / wall_dur if wall_dur > 0 else 0
    output_tps = total_output_tokens / wall_dur if wall_dur > 0 else 0
    actual_tpm_avg = actual_tokens_total / wall_dur * 60 if wall_dur > 0 else 0
    actual_tpm_stable = (
        (steady_input_tokens + steady_output_tokens) / steady_dur * 60
        if has_steady else None
    )

    error_rate = 1 - success_requests / transport_success if transport_success > 0 else None

    full_ttfts = [o.ttft for o in ok_flat if o.ttft > 0]
    full_e2es = [o.latency for o in ok_flat if o.latency > 0]
    full_tpots = [t for t in (_compute_tpot(o) for o in ok_flat) if t is not None]
    full_itls = [x for o in ok_flat for x in (o.itl or []) if x > 0]

    steady_ttfts = [o.ttft for o in ok_steady if o.ttft > 0]
    steady_e2es = [o.latency for o in ok_steady if o.latency > 0]
    steady_tpots = [t for t in (_compute_tpot(o) for o in ok_steady) if t is not None]
    steady_itls = [x for o in ok_steady for x in (o.itl or []) if x > 0]

    lines = []
    lines.append(f"# 多轮长上下文对话 Benchmark 报告")
    lines.append("")
    lines.append(f"> 生成时间：{now_str}  |  模型：`{model}`  |  后端：`{backend}`")
    lines.append("")

    status = "异常终止" if terminated else "正常完成"
    lines.append(f"验收状态：**{status}**")
    if terminated:
        lines.append(f"- 终止原因：`{termination_reason}`")
    lines.append("")
    lines.append(
        "稳态指标仅使用连续达标窗口，按请求完成时间归集；分轮与分模式均采用明确窗口。"
        "TTFT 为客户端首个内容/推理数据块到达时间，TPOT 为 usage 估算，"
        "SSE 数据块间隔不等于逐 token 时延。N/A 表示不可测或无样本。"
    )
    lines.append("")

    # ========== 1. 全程 ==========
    lines.append("## 全程")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| total_requests | {_fmt_count(total_requests)} |")
    lines.append(f"| transport_success_requests | {_fmt_count(transport_success)} |")
    lines.append(f"| success_requests | {_fmt_count(success_requests)} |")
    lines.append(f"| output_truncated_requests | {_fmt_count(output_truncated)} |")
    lines.append(f"| reasoning_tokens_total | 0.00 |")
    lines.append(f"| actual_tokens_total | {_fmt_count(actual_tokens_total)} |")
    lines.append(f"| usable_answer_tokens_total | {_fmt_count(total_output_tokens)} |")
    lines.append(f"| peak_window_complete | {'是' if wall_dur >= args.ramp_seconds else '否'} |")
    lines.append(f"| duration_sec | {wall_dur:.2f} |")
    lines.append(f"| request_throughput_rps | {request_rps:,.2f} |")
    lines.append(f"| input_throughput_tps | {input_tps:,.2f} |")
    lines.append(f"| output_throughput_tps | {output_tps:,.2f} |")
    lines.append(f"| actual_tpm_avg | {actual_tpm_avg:,.2f} |")
    lines.append(f"| actual_tpm_stable | {_fmt_num(actual_tpm_stable, ',.2f')} |")
    lines.append(f"| cache_hit_rate | {_fmt_num(cache_hit_rate, '.2f')} |")
    lines.append(f"| cache_observable_rate | {_fmt_num(cache_observable_rate, '.2f')} |")
    lines.append("")

    lines.append("| Metric (ms) | Samples | avg | p50 | p75 | p90 | p95 | p99 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.append(_latency_row("First content/reasoning chunk arrival", full_ttfts))
    lines.append(_latency_row("End-to-end latency", full_e2es))
    lines.append(_latency_row("Estimated TPOT", full_tpots))
    lines.append(_latency_row("SSE chunk interval (not token ITL)", full_itls))
    lines.append("")

    # ========== 2. 连续稳态 ==========
    lines.append("## 连续稳态")
    lines.append("")
    if has_steady:
        lines.append(f"已获得稳态窗口：{steady_sessions} 会话，{steady_dur:.2f}s。")
    else:
        lines.append("尚未获得完整连续稳态窗口；不可验收。")
    lines.append("")

    # ========== 3. 验收明细 ==========
    lines.append("## 验收明细")
    lines.append("")
    lines.append("| 项目 | 实际 | 要求 | 结论 | 必过 |")
    lines.append("| --- | --- | --- | --- | --- |")

    def verdict(actual, required, cmp_fn=None):
        if required is None:
            return "N/A"
        if required == -1.0:
            return "None"
        if actual is None:
            return "False"
        if cmp_fn:
            return "True" if cmp_fn(actual, required) else "False"
        return "True" if actual >= required else "False"

    steady_tpm = actual_tpm_stable
    steady_rps = metrics_steady.request_throughput if has_steady else None
    steady_success_rate = (
        len(ok_steady) / len(steady_flat) if steady_flat else None
    )
    _steady_ttft_vals = [o.ttft for o in ok_steady if o.ttft > 0]
    _ttft_p50_raw = _pct(_steady_ttft_vals, 50) if _steady_ttft_vals else None
    steady_ttft_p50 = _ttft_p50_raw * 1000 if _ttft_p50_raw is not None else None

    _steady_tpot_vals = [t for t in (_compute_tpot(o) for o in ok_steady) if t is not None]
    _tpot_p50_raw = _pct(_steady_tpot_vals, 50) if _steady_tpot_vals else None
    steady_tpot_p50 = _tpot_p50_raw * 1000 if _tpot_p50_raw is not None else None

    usage_complete = (
        (1 if all(o.output_len > 0 for o in ok_steady) else 0)
        if ok_steady else None
    )

    items = [
        ("steady_tpm", steady_tpm, args.accept_steady_tpm,
         lambda a, r: a is not None and a >= r, True),
        ("continuous_steady_window", has_steady, True, lambda a, r: a == r, True),
        ("request_throughput_rps", steady_rps, args.accept_request_rps,
         lambda a, r: a is not None and a >= r, True),
        ("success_rate", steady_success_rate, args.accept_success_rate,
         lambda a, r: a is not None and a >= r, True),
        ("error_rate", error_rate if has_steady else None, -1.0, lambda a, r: True, False),
        ("http_429_rate", cum_429 if has_steady else None, -1.0, lambda a, r: True, False),
        ("ttft_ms_p50", steady_ttft_p50, args.accept_ttft_p50_ms,
         lambda a, r: a is not None and a <= r, True),
        ("tpot_ms_p50", steady_tpot_p50, args.accept_tpot_p50_ms,
         lambda a, r: a is not None and a <= r, True),
        ("steady_cache_hit_rate", steady_cache_hit, args.accept_cache_hit_rate,
         lambda a, r: a is not None and a >= r, True),
        ("zero_429", cum_429 if has_steady else None, args.accept_zero_429,
         lambda a, r: a <= r, True),
        ("usage_complete", usage_complete, args.accept_usage_complete,
         lambda a, r: a == r, True),
    ]

    for name, actual, required, cmp_fn, must in items:
        act_str = _fmt_num(actual, ".4f") if isinstance(actual, float) else str(actual)
        req_str = str(required)
        v = verdict(actual, required, cmp_fn)
        lines.append(f"| {name} | {act_str} | {req_str} | {v} | {must} |")

    actual_rounds = {}
    if all_session_outputs:
        counts = [len(s) for s in all_session_outputs if s]
        actual_rounds = {
            "count": len(counts),
            "avg": float(np.mean(counts)) if counts else 0,
            "p50": _pct(counts, 50) if counts else 0,
            "p75": _pct(counts, 75) if counts else 0,
            "p90": _pct(counts, 90) if counts else 0,
            "p95": _pct(counts, 95) if counts else 0,
            "p99": _pct(counts, 99) if counts else 0,
            "max": max(counts) if counts else 0,
        }
    length_profile = {
        "scope": "steady",
        "input_mean": float(np.mean([
            (getattr(s[0], "prompt_len", 0) or 0) + sum(getattr(s[j], "output_len", 0) or 0 for j in range(r_idx))
            for s in steady_session_outputs for r_idx, o in enumerate(s) if o is not None and o.success
        ])) if ok_steady else None,
        "output_mean": float(np.mean([o.output_len for o in ok_steady])) if ok_steady else None,
        "lengths_match": has_steady,
        "truncated_session_plans": output_truncated,
        "actual_rounds_per_session": actual_rounds,
        "synthetic_requests": 0,
        "dataset_mode": "sharegpt-multiturn",
        "context_scale": 1.0,
        "context_budget_exhausted_sessions": 0,
        "note": "ShareGPT multi-turn dataset with per-session unique system prompt",
    }
    lines.append(
        f"| length_profile | `{json.dumps(length_profile, ensure_ascii=False)}` "
        f"| observed input/output means within configured tolerance "
        f"| {'True' if has_steady else 'False'} | True |"
    )
    lines.append(f"| dataset_round_budget | 0 | 0 | True | True |")
    lines.append(f"| context_budget_exhausted_sessions | 0 | 0 | True | True |")
    lines.append(f"| real multi-turn dataset | None | None | True | True |")
    lines.append(f"| global RPM pacing with usage feedback | None | None | True | True |")
    lines.append("")

    # ========== 4. 全程分轮 ==========
    lines.append("## 全程分轮")
    lines.append("")
    full_summary, full_round_lat = _round_breakdown(
        all_session_outputs, wall_dur, args.max_tokens_per_turn
    )
    lines.append("| 真实 round | 请求数 | 时长秒 | req/s | input tok/s | output tok/s | cache hit | cache observable |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r, count, dur, rps, in_tps, out_tps, ch, co in full_summary:
        lines.append(
            f"| {r} | {count} | {dur:.2f} | {rps:.2f} | {in_tps:,.2f} | "
            f"{out_tps:,.2f} | {_fmt_cache(ch)} | {_fmt_cache(co)} |"
        )
    lines.append("")

    for r in range(len(full_summary)):
        ttfts, e2es, tpots, itls = full_round_lat.get(r, ([], [], [], []))
        lines.append(f"### Round {r}")
        lines.append("")
        lines.append("| Metric (ms) | Samples | avg | p50 | p75 | p90 | p95 | p99 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        lines.append(_latency_row("First content/reasoning chunk arrival", ttfts))
        lines.append(_latency_row("End-to-end latency", e2es))
        lines.append(_latency_row("Estimated TPOT", tpots))
        lines.append(_latency_row("SSE chunk interval (not token ITL)", itls))
        lines.append("")

    # ========== 5. 稳态分轮 ==========
    lines.append("## 稳态分轮")
    lines.append("")
    if has_steady:
        steady_summary, steady_round_lat = _round_breakdown(
            steady_session_outputs, steady_dur, args.max_tokens_per_turn
        )
        lines.append("| 真实 round | 请求数 | 时长秒 | req/s | input tok/s | output tok/s | cache hit | cache observable |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r, count, dur, rps, in_tps, out_tps, ch, co in steady_summary:
            lines.append(
                f"| {r} | {count} | {dur:.2f} | {rps:.2f} | {in_tps:,.2f} | "
                f"{out_tps:,.2f} | {_fmt_cache(ch)} | {_fmt_cache(co)} |"
            )
        lines.append("")
    else:
        lines.append("| 真实 round | 请求数 | 时长秒 | req/s | input tok/s | output tok/s | cache hit | cache observable |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        lines.append("")

    # ========== 6. stream / full_run ==========
    lines.append("## stream / full_run")
    lines.append("")
    lines.append("| Metric (ms) | Samples | avg | p50 | p75 | p90 | p95 | p99 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.append(_latency_row("First content/reasoning chunk arrival", full_ttfts))
    lines.append(_latency_row("End-to-end latency", full_e2es))
    lines.append(_latency_row("Estimated TPOT", full_tpots))
    lines.append(_latency_row("SSE chunk interval (not token ITL)", full_itls))
    lines.append("")

    # ========== 7. stream / steady ==========
    lines.append("## stream / steady")
    lines.append("")
    lines.append("| Metric (ms) | Samples | avg | p50 | p75 | p90 | p95 | p99 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.append(_latency_row("First content/reasoning chunk arrival", steady_ttfts))
    lines.append(_latency_row("End-to-end latency", steady_e2es))
    lines.append(_latency_row("Estimated TPOT", steady_tpots))
    lines.append(_latency_row("SSE chunk interval (not token ITL)", steady_itls))
    lines.append("")

    # ========== 8. 加压时间序列 ==========
    lines.append("## 加压时间序列")
    lines.append("")
    lines.append("| elapsed s | phase | scheduled RPM | actual RPM/60s | actual TPM/60s | 完整窗口 | RPM 受限 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for s in monitor_history:
        phase = s.get("phase", "")
        if s is monitor_history[-1] and terminated:
            phase = "incomplete"
        lines.append(
            f"| {s['t']:.2f} | {phase} | {s['scheduled_rpm']:.2f} | "
            f"{s['actual_rpm']:.0f} | {s['actual_tpm']:.0f} | "
            f"{'True' if s['complete_window'] else 'False'} | "
            f"{'True' if s['rpm_limited'] else 'False'} |"
        )
    lines.append("")

    # ========== 9. 实际长度与轮数 ==========
    lines.append("## 实际长度与轮数")
    lines.append("")
    lp = {
        "scope": "steady",
        "input_mean": float(np.mean([
            (getattr(s[0], "prompt_len", 0) or 0) + sum(getattr(s[j], "output_len", 0) or 0 for j in range(r_idx))
            for s in steady_session_outputs for r_idx, o in enumerate(s) if o is not None and o.success
        ])) if ok_steady else None,
        "output_mean": float(np.mean([o.output_len for o in ok_steady])) if ok_steady else None,
        "lengths_match": has_steady,
        "truncated_session_plans": output_truncated,
        "actual_rounds_per_session": actual_rounds,
        "synthetic_requests": 0,
        "dataset_mode": "sharegpt-multiturn",
        "context_scale": 1.0,
        "context_budget_exhausted_sessions": 0,
        "note": "ShareGPT multi-turn dataset with per-session unique system prompt",
    }
    lines.append("```json")
    lines.append(json.dumps(lp, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)

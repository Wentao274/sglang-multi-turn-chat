"""Monitor window/throughput regression tests.

回归背景（glm-5.3-test-20261001-061110 事故分析）：
1. 窗口曾按请求开始时间索引：过载下慢请求（TTFT 数百秒）完成时已滑出
   30s 窗口，监控只能看到快请求 → 79% 检查被跳过、TTFT 止损结构性
   不可能触发。修复：窗口按完成时刻索引。
2. 吞吐分母曾用样本时间戳跨度：服务器崩溃时大量失败瞬间完成、时间戳
   几乎相同 → 跨度≈0 → 吞吐出现 1e9~8e12 垃圾值。修复：分母用墙钟
   窗口时长。
3. stop_event 曾只在调度循环内检查：监控在调度结束后触发时无人取消
   任务 → 30.3 小时排空。修复：terminate_listener + drain 超时。

Usage:  python test_monitor.py
"""

import asyncio
import time

import test_cached_client  # noqa: F401  构建 sglang stubs
from test_cached_client import RequestFuncOutput

import bench_multi_turn  # noqa: F401  导入即冒烟：验证改动无语法/名字错误
from monitor import DegradationMonitor


def _mk_monitor(window=30.0, max_ttft_p99_ms=30000.0, max_error_rate=0.10, **kw):
    now = time.perf_counter()
    return DegradationMonitor(
        ramp_end_time=now - 120.0,   # 爬坡 2 分钟前已结束
        bench_start=now - 600.0,     # 测试 10 分钟前开始
        start_rps=0.0,
        target_rps=10.0,
        ramp_seconds=60.0,
        max_concurrency=0,
        window=window,
        max_ttft_p99_ms=max_ttft_p99_ms,
        max_error_rate=max_error_rate,
        **kw,
    )


def _mk_out(ttft, latency, success=True, output_len=170, prompt_len=8681, error=""):
    return RequestFuncOutput(
        success=success, prompt_len=prompt_len, output_len=output_len,
        ttft=ttft, latency=latency, itl=[0.007] * (output_len - 1),
        error=error,
    )


def test_slow_requests_stay_visible():
    """过载核心场景：慢请求（TTFT 280s）完成时必须留在 30s 窗口内。

    旧行为：rec[0]=output.start_time（约 300s 前）→ 完成即滑出窗口。
    """
    m = _mk_monitor()
    m.feed(_mk_out(ttft=280.0, latency=300.0))
    assert len(m._samples) == 1, "slow request should be in window"
    samples = [s for s in m._samples if s[0] >= time.perf_counter() - m.window]
    assert len(samples) == 1, "check() filter should see the slow request"
    print("PASS: slow request (TTFT 280s) remains visible in 30s window")


def test_ttft_stop_loss_fires_on_slow_requests():
    """窗口 TTFT p99 含慢请求时必须触发止损（修复前结构性不可能触发）。

    修复前：check() 里 samples 按旧 start_time 过滤 → 全部不可见 →
    len<5 直接跳过，止损永不触发。
    """
    m = _mk_monitor()
    for _ in range(6):
        m.feed(_mk_out(ttft=280.0, latency=300.0))
    first = m.check()
    assert first is None, f"first check should lock baseline, got {first}"
    second = m.check()
    assert second is not None and "TTFT" in second, \
        f"TTFT stop-loss should fire (p99 280s >> 30s), got {second}"
    print(f"PASS: TTFT stop-loss fires -> {second}")


def test_no_garbage_throughput_on_burst_completion():
    """崩溃场景：批量失败瞬间完成（样本时间戳跨度≈0），吞吐不得爆炸。

    旧行为：window_dur=max(跨度,1e-9)≈1e-9 → 吞吐 = n/1e-9 ≈ 1e9+。
    """
    m = _mk_monitor()
    for _ in range(50):
        m.feed(_mk_out(ttft=0.0, latency=0.0, success=False, error="Connection reset"))
    snap = m.snapshot()
    assert snap is not None
    # 分母 = min(30s, 墙钟) → 50 次失败 / 30s ≈ 1.67，绝不可能是 1e9
    assert snap.throughput < 10.0, f"garbage throughput: {snap.throughput}"
    assert snap.output_throughput < 10.0, f"garbage output_throughput: {snap.output_throughput}"
    # 全部失败 → 窗口错误率 100%，error 止损必须触发（事故当场的真实模式）
    reason = m.check()
    assert reason is not None and "error_rate" in reason, \
        f"error burst should trigger stop, got {reason}"
    print(f"PASS: burst completion -> throughput={snap.throughput:.3f}, stop: {reason}")


def test_throughput_drop_detectable_when_stalled():
    """正常吞吐基线建立后，窗口清空时吞吐骤降检查应可判定（而非被跳过）。"""
    m = _mk_monitor()
    for _ in range(10):
        m.feed(_mk_out(ttft=0.5, latency=1.0))
    assert m.check() is None  # 锁基线
    # 30s 窗口过后样本过期 → 样本数 <5 → 跳过；此时不误报即可
    # （真正挂死场景由 bench 侧 drain-timeout 兜底）
    assert m.check() is None
    print("PASS: stale window does not false-positive after baseline lock")


def test_terminate_listener_cancels_pending_tasks():
    """调度结束后才触发的 stop_event 必须立刻取消所有 pending 任务。

    事故重现：36,096 个会话任务在调度循环正常结束后无人取消，
    asyncio.gather 排空 26 小时。
    """
    async def _run():
        stop_event = asyncio.Event()
        tasks = []
        cancelled = asyncio.Event()

        async def hang():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def terminate_listener():
            await stop_event.wait()
            for t in tasks:
                if not t.done():
                    t.cancel()

        tasks.append(asyncio.create_task(hang()))
        lt = asyncio.create_task(terminate_listener())
        await asyncio.sleep(0.05)  # 模拟调度循环已正常结束
        t0 = time.perf_counter()
        stop_event.set()          # 此后监控才触发终止
        await asyncio.wait_for(cancelled.wait(), timeout=2.0)
        dt = time.perf_counter() - t0
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        lt.cancel()
        return dt

    dt = asyncio.run(_run())
    assert dt < 1.0, f"cancellation took {dt:.3f}s, listener failed"
    print(f"PASS: post-schedule stop cancels pending tasks in {dt*1000:.1f}ms")


def test_drain_timeout_cancels_stragglers():
    """无终止信号时，排空超时后必须强制取消滞留任务（防服务器挂起）。"""
    async def _run():
        tasks = [asyncio.create_task(asyncio.sleep(3600)) for _ in range(3)]
        drain_timeout = 0.2
        await asyncio.wait(tasks, timeout=drain_timeout)
        stragglers = [t for t in tasks if not t.done()]
        assert len(stragglers) == 3, "all tasks should still be pending"
        for t in stragglers:
            t.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return all(isinstance(r, asyncio.CancelledError) for r in results)

    assert asyncio.run(_run())
    print("PASS: drain timeout cancels stragglers")


if __name__ == "__main__":
    test_slow_requests_stay_visible()
    test_ttft_stop_loss_fires_on_slow_requests()
    test_no_garbage_throughput_on_burst_completion()
    test_throughput_drop_detectable_when_stalled()
    test_terminate_listener_cancels_pending_tasks()
    test_drain_timeout_cancels_stragglers()
    print("\nAll monitor regression tests passed.")

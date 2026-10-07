import math
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Deque, List, Optional, Tuple

import numpy as np


@dataclass
class MetricsSnapshot:
    t: float
    phase: str
    scheduled_rpm: float
    actual_rpm: float
    actual_tpm: float
    complete_window: bool
    rpm_limited: bool
    cumulative_completed: int
    cumulative_failed: int
    window_completed: int
    window_failed: int
    error_rate: float
    throughput: float
    output_throughput: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    tpot_mean_ms: float
    itl_mean_ms: float
    itl_p99_ms: float
    e2e_p99_ms: float
    concurrency: float
    baseline_throughput: Optional[float]

    def to_dict(self):
        return asdict(self)


class DegradationMonitor:
    """滑动窗口衰减检测器 + 时间序列快照采集。

    同时维护两个窗口：
    - _samples: monitor_window 秒窗口，用于衰减检测
    - _ramp_samples: 60 秒窗口，用于加压时间序列的 RPM/TPM 统计
    """

    def __init__(
        self,
        ramp_end_time: float,
        bench_start: float,
        start_rps: float,
        target_rps: float,
        ramp_seconds: float,
        max_concurrency: int,
        interval: float = 10.0,
        window: float = 30.0,
        max_error_rate: float = 0.10,
        throughput_drop_ratio: float = 0.5,
        max_ttft_p99_ms: float = 30000.0,
        baseline_warmup_seconds: float = 30.0,
    ):
        self.ramp_end_time = ramp_end_time
        self.bench_start = bench_start
        self.start_rps = start_rps
        self.target_rps = target_rps
        self.ramp_seconds = ramp_seconds
        self.max_concurrency = max_concurrency if max_concurrency > 0 else 10 ** 9
        self.interval = interval
        self.window = window
        self.max_error_rate = max_error_rate
        self.throughput_drop_ratio = throughput_drop_ratio
        self.max_ttft_p99_ms = max_ttft_p99_ms
        self.baseline_warmup_seconds = baseline_warmup_seconds

        self._samples: Deque[Tuple] = deque()
        self._ramp_samples: Deque[Tuple] = deque()
        self.baseline_throughput: Optional[float] = None
        self.baseline_locked_at: Optional[float] = None
        self.stop_reason: Optional[str] = None
        self.history: List[MetricsSnapshot] = []
        self._cum_completed = 0
        self._cum_failed = 0
        self._cum_429 = 0
        self._peak_concurrent = 0

    def feed(self, output) -> None:
        if output is None:
            return
        ok = bool(output.success)
        latency = getattr(output, "latency", 0.0)
        ttft = getattr(output, "ttft", 0.0)
        output_len = getattr(output, "output_len", 0)
        prompt_len = getattr(output, "prompt_len", 0)
        itl = getattr(output, "itl", []) or []
        tpot = (latency - ttft) / max(output_len - 1, 1) if output_len > 1 else 0.0
        cached = getattr(output, "cached_tokens", 0) or 0
        error_str = getattr(output, "error", "") or ""
        is_429 = "429" in error_str or "Too Many Requests" in error_str

        if ok:
            self._cum_completed += 1
        else:
            self._cum_failed += 1
        if is_429:
            self._cum_429 += 1

        now = time.perf_counter()
        # 窗口按完成时刻索引：过载时慢请求（TTFT 数百秒）完成时才进入窗口，
        # 才能被衰减检测看见。若按开始时间索引，慢请求完成时已滑出窗口，
        # 监控只能看到快请求，止损永远不触发。
        rec = (now, latency, ttft, tpot, itl, output_len, prompt_len, ok, cached, is_429)
        self._samples.append(rec)
        self._ramp_samples.append(rec)

        deg_cutoff = now - self.window
        while self._samples and self._samples[0][0] < deg_cutoff:
            self._samples.popleft()
        ramp_cutoff = now - 60.0
        while self._ramp_samples and self._ramp_samples[0][0] < ramp_cutoff:
            self._ramp_samples.popleft()

    def note_concurrency(self, current_inflight: int):
        if current_inflight > self._peak_concurrent:
            self._peak_concurrent = current_inflight

    def _phase(self, elapsed: float) -> str:
        if elapsed < self.ramp_seconds:
            return "ramp"
        return "stabilizing"

    def _scheduled_rpm(self, elapsed: float) -> float:
        if elapsed >= self.ramp_seconds or self.ramp_seconds <= 0:
            return self.target_rps * 60.0
        frac = elapsed / self.ramp_seconds
        rps = self.start_rps + (self.target_rps - self.start_rps) * frac
        return rps * 60.0

    def _build_snapshot(self) -> Optional[MetricsSnapshot]:
        now = time.perf_counter()
        elapsed = now - self.bench_start
        samples = [s for s in self._samples if s[0] >= now - self.window]
        ramp_samples = list(self._ramp_samples)
        t = round(elapsed, 2)

        if not samples:
            ok_ramp = [s for s in ramp_samples if s[7]]
            ramp_dur = max(now - self.bench_start, 1e-9)
            ramp_dur = min(ramp_dur, 60.0)
            return MetricsSnapshot(
                t=t, phase=self._phase(elapsed),
                scheduled_rpm=self._scheduled_rpm(elapsed),
                actual_rpm=len(ramp_samples) / ramp_dur * 60.0,
                actual_tpm=sum(s[5] + s[6] for s in ok_ramp) / ramp_dur * 60.0,
                complete_window=elapsed >= 60.0,
                rpm_limited=self._peak_concurrent >= self.max_concurrency,
                cumulative_completed=self._cum_completed,
                cumulative_failed=self._cum_failed,
                window_completed=0, window_failed=0, error_rate=0.0,
                throughput=0.0, output_throughput=0.0,
                ttft_p50_ms=0, ttft_p90_ms=0, ttft_p99_ms=0,
                tpot_mean_ms=0, itl_mean_ms=0, itl_p99_ms=0,
                e2e_p99_ms=0, concurrency=0.0,
                baseline_throughput=self.baseline_throughput,
            )

        arr_ok = np.array([s[7] for s in samples], dtype=bool)
        ok_samples = [s for s in samples if s[7]]
        ttfts = [s[2] for s in ok_samples]
        tpots = [s[3] for s in ok_samples]
        itls = [x for s in ok_samples for x in s[4]]
        e2es = [s[1] for s in ok_samples]
        out_lens = [s[5] for s in ok_samples]
        prompt_lens = [s[6] for s in ok_samples]
        # 吞吐分母用墙钟窗口时长，而非样本时间戳跨度：
        # 崩溃场景下大量失败瞬间完成、时间戳几乎相同，样本跨度≈0 会
        # 除出 1e9 量级的垃圾吞吐值。
        window_dur = min(self.window, max(now - self.bench_start, 1e-9))
        concurrency = float(np.sum([s[1] for s in samples]) / window_dur) if samples else 0.0

        ok_ramp = [s for s in ramp_samples if s[7]]
        ramp_dur = min(max(now - self.bench_start, 1e-9), 60.0)
        actual_rpm = len(ramp_samples) / ramp_dur * 60.0
        actual_tpm = sum(s[5] + s[6] for s in ok_ramp) / ramp_dur * 60.0

        def pct(vals, q):
            return float(np.percentile(vals, q)) * 1000.0 if vals else 0.0

        def mean_ms(vals):
            return float(np.mean(vals)) * 1000.0 if vals else 0.0

        return MetricsSnapshot(
            t=t, phase=self._phase(elapsed),
            scheduled_rpm=self._scheduled_rpm(elapsed),
            actual_rpm=actual_rpm,
            actual_tpm=actual_tpm,
            complete_window=elapsed >= 60.0,
            rpm_limited=self._peak_concurrent >= self.max_concurrency,
            cumulative_completed=self._cum_completed,
            cumulative_failed=self._cum_failed,
            window_completed=int(arr_ok.sum()),
            window_failed=int((~arr_ok).sum()),
            error_rate=float(1.0 - arr_ok.mean()) if len(arr_ok) else 0.0,
            throughput=len(ok_samples) / window_dur,
            output_throughput=float(np.sum(out_lens)) / window_dur if out_lens else 0.0,
            ttft_p50_ms=pct(ttfts, 50),
            ttft_p90_ms=pct(ttfts, 90),
            ttft_p99_ms=pct(ttfts, 99),
            tpot_mean_ms=mean_ms(tpots),
            itl_mean_ms=mean_ms(itls),
            itl_p99_ms=pct(itls, 99),
            e2e_p99_ms=pct(e2es, 99),
            concurrency=concurrency,
            baseline_throughput=self.baseline_throughput,
        )

    def snapshot(self) -> Optional[MetricsSnapshot]:
        snap = self._build_snapshot()
        if snap is not None:
            self.history.append(snap)
        return snap

    def check(self) -> Optional[str]:
        if self.stop_reason:
            return self.stop_reason
        now = time.perf_counter()
        steady_elapsed = now - self.ramp_end_time
        samples = [s for s in self._samples if s[0] >= now - self.window]
        if len(samples) < 5:
            return None

        arr_ok = np.array([s[7] for s in samples], dtype=bool)
        error_rate = 1.0 - float(arr_ok.mean())
        if error_rate >= self.max_error_rate:
            self.stop_reason = (
                f"error_rate {error_rate:.1%} >= {self.max_error_rate:.1%} "
                f"(window {len(samples)} rounds)"
            )
            return self.stop_reason

        if steady_elapsed < self.baseline_warmup_seconds:
            return None

        window_dur = min(self.window, max(now - self.bench_start, 1e-9))
        if self.baseline_throughput is None:
            ok_samples = [s for s in samples if s[7]]
            self.baseline_throughput = len(ok_samples) / window_dur
            self.baseline_locked_at = now
            return None

        ok_samples = [s for s in samples if s[7]]
        ttfts = [s[2] for s in ok_samples]
        if ttfts:
            p99 = float(np.percentile(ttfts, 99)) * 1000.0
            if p99 >= self.max_ttft_p99_ms:
                self.stop_reason = f"TTFT p99 {p99:.0f}ms >= {self.max_ttft_p99_ms:.0f}ms"
                return self.stop_reason

        throughput = len(ok_samples) / window_dur
        if self.baseline_throughput > 0 and throughput < self.throughput_drop_ratio * self.baseline_throughput:
            self.stop_reason = (
                f"throughput {throughput:.3f} < "
                f"{self.throughput_drop_ratio:.0%} of baseline "
                f"{self.baseline_throughput:.3f}"
            )
            return self.stop_reason

        return None

    def history_dicts(self) -> List[dict]:
        return [s.to_dict() for s in self.history]

    @property
    def cum_429(self) -> int:
        return self._cum_429

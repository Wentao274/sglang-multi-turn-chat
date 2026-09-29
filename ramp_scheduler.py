import asyncio
import math
import time
from typing import AsyncGenerator, List, Optional

import numpy as np

from sglang.benchmark.datasets.common import DatasetRow


def _lambda_of_t(t, r0, r1, T):
    if T <= 0 or t >= T:
        return r1
    return r0 + (r1 - r0) * (t / T)


def _lambda_T(r0, r1, T):
    return T * (r0 + r1) / 2.0


def _invert_lambda(c, r0, r1, T):
    if c <= 0:
        return 0.0
    if T <= 0:
        return c / max(r1, 1e-9)
    lam_T = _lambda_T(r0, r1, T)
    if c <= lam_T:
        a = (r1 - r0) / (2.0 * T)
        if a > 1e-12:
            return (-r0 + math.sqrt(r0 * r0 + 4.0 * a * c)) / (2.0 * a)
        return c / max(r0, 1e-9)
    return T + (c - lam_T) / max(r1, 1e-9)


def rate_at(elapsed, start_rps, target_rps, ramp_seconds):
    return _lambda_of_t(elapsed, start_rps, target_rps, ramp_seconds)


async def get_ramp_request(
    input_requests: List[DatasetRow],
    start_rps: float,
    target_rps: float,
    ramp_seconds: float,
    sustain_seconds: Optional[float] = None,
    seed: Optional[int] = None,
    stop_event: Optional["asyncio.Event"] = None,
) -> AsyncGenerator[DatasetRow, None]:
    if start_rps <= 0 and target_rps <= 0:
        for req in input_requests:
            if stop_event is not None and stop_event.is_set():
                return
            yield req
        return

    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()
    cumulative = 0.0
    window = ramp_seconds + sustain_seconds if sustain_seconds is not None else None

    for req in input_requests:
        if stop_event is not None and stop_event.is_set():
            return
        cumulative += float(rng.exponential(1.0))
        t = _invert_lambda(cumulative, start_rps, target_rps, ramp_seconds)
        if window is not None and t > window:
            return
        target = start_time + t
        delay = target - time.perf_counter()
        while delay > 0:
            if stop_event is not None and stop_event.is_set():
                return
            await asyncio.sleep(min(delay, 1.0))
            delay = target - time.perf_counter()
        yield req

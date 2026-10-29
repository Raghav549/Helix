"""Observability primitives for the HELIX web service.

Everything here is privacy-conscious by construction: access logs contain a
request id, the route template, a status code, a latency and an error code.
Bodies, prompts, file contents, credentials and environment values are never
logged.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

REQUEST_ID_HEADER = 'X-Request-ID'
_REQUEST_ID_RE = re.compile(r'^[A-Za-z0-9._-]{8,80}$')


def valid_request_id(value: Optional[str]) -> Optional[str]:
    """Accept only a client-supplied id that cannot inject into logs or headers."""
    if isinstance(value, str) and _REQUEST_ID_RE.fullmatch(value):
        return value
    return None


class AccessLogger:
    """Emit one JSON line per request to stdout (Render captures stdout)."""

    def __init__(self, stream=None, clock=time.time):
        self._stream = stream
        self._clock = clock
        self._lock = threading.Lock()

    def log(self, **fields) -> None:
        record = {'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self._clock())),
                  'service': 'helix-backend'}
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            stream = self._stream
            if stream is None:
                print(line, flush=True)
            else:
                stream.write(line + '\n')


class TokenBucket:
    """A monotonic clock token bucket used for cheap per-client rate limiting."""

    __slots__ = ('capacity', 'refill_per_second', 'tokens', 'updated')

    def __init__(self, capacity: float, refill_per_second: float, now: float):
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.tokens = capacity
        self.updated = now

    def consume(self, now: float, amount: float = 1.0) -> Tuple[bool, float]:
        elapsed = max(0.0, now - self.updated)
        self.updated = now
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        if self.tokens >= amount:
            self.tokens -= amount
            return True, 0.0
        missing = amount - self.tokens
        return False, missing / self.refill_per_second if self.refill_per_second > 0 else 60.0


class RateLimiter:
    """In-process token-bucket limiter.

    Scoped per worker process, which is honest: Render runs a single gunicorn
    worker in the deployment blueprint, so the limit is global there. With
    several workers the effective limit multiplies, and the docs say so.
    """

    def __init__(self, per_minute: int, burst: int, clock=time.monotonic, max_keys: int = 4096):
        self.per_minute = max(1, int(per_minute))
        self.burst = max(1, int(burst))
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: Dict[str, TokenBucket] = {}
        self._order: Deque[str] = deque()
        self._max_keys = max_keys

    def check(self, key: str, cost: float = 1.0) -> Tuple[bool, float]:
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(self.burst, self.per_minute / 60.0, now)
                self._buckets[key] = bucket
                self._order.append(key)
                while len(self._order) > self._max_keys:
                    self._buckets.pop(self._order.popleft(), None)
            return bucket.consume(now, cost)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()
            self._order.clear()


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, '') or default)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))

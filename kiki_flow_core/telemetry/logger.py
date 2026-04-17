"""Structured JSON logger for kiki-flow steps."""

from __future__ import annotations

import json
import sys
import time
from typing import IO, Any


class StructuredLogger:
    """Write one JSON object per record to the configured stream."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        self.stream = stream or sys.stdout

    def record(
        self,
        *,
        track: str,
        tau: int,
        step_phase: str,
        status: str,
        duration_ms: float,
        errors: list[str] | None = None,
        **extra: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "track": track,
            "tau": tau,
            "step_phase": step_phase,
            "status": status,
            "duration_ms": duration_ms,
        }
        if errors is not None:
            payload["errors"] = errors
        payload.update(extra)
        self.stream.write(json.dumps(payload) + "\n")
        self.stream.flush()

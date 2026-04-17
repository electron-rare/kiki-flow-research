"""In-memory metrics registry with Prometheus text export."""

from __future__ import annotations

import threading
from typing import Literal

MetricKind = Literal["counter", "gauge"]


class Metrics:
    """Thread-safe metrics aggregator with Prometheus text exposition export."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[tuple[str, str], float] = {}
        self._kinds: dict[str, MetricKind] = {}

    def record(self, *, track: str, metric_name: str, value: float, kind: MetricKind) -> None:
        with self._lock:
            self._kinds[metric_name] = kind
            key = (track, metric_name)
            if kind == "counter":
                self._values[key] = self._values.get(key, 0.0) + value
            else:
                self._values[key] = value

    def snapshot(self) -> dict[tuple[str, str], float]:
        with self._lock:
            return dict(self._values)

    def export_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (track, name), value in self._values.items():
                kind = self._kinds.get(name, "gauge")
                metric_full = f"kiki_flow_{name}"
                lines.append(f"# TYPE {metric_full} {kind}")
                lines.append(f'{metric_full}{{track="{track}"}} {value}')
        return "\n".join(lines) + "\n"

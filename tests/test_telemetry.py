import io
import json
import threading

from kiki_flow_core.telemetry.logger import StructuredLogger
from kiki_flow_core.telemetry.metrics import Metrics


def test_structured_logger_writes_valid_json():
    buf = io.StringIO()
    log = StructuredLogger(stream=buf)
    log.record(track="T1", tau=0, step_phase="advection", status="ok", duration_ms=12.5)
    line = buf.getvalue().strip()
    payload = json.loads(line)
    assert payload["track"] == "T1"
    assert payload["status"] == "ok"
    assert "ts" in payload


def test_structured_logger_includes_errors_field_when_provided():
    buf = io.StringIO()
    log = StructuredLogger(stream=buf)
    log.record(
        track="T2",
        tau=5,
        step_phase="jko",
        status="error",
        duration_ms=99.0,
        errors=["mass_drift"],
    )
    payload = json.loads(buf.getvalue().strip())
    assert payload["errors"] == ["mass_drift"]


def test_metrics_record_increments_counter():
    m = Metrics()
    m.record(track="T1", metric_name="steps_total", value=1, kind="counter")
    m.record(track="T1", metric_name="steps_total", value=2, kind="counter")
    assert m.snapshot()[("T1", "steps_total")] == 3  # noqa: PLR2004


def test_metrics_record_gauge_overwrites():
    m = Metrics()
    m.record(track="T1", metric_name="F_value", value=1.5, kind="gauge")
    m.record(track="T1", metric_name="F_value", value=0.8, kind="gauge")
    assert m.snapshot()[("T1", "F_value")] == 0.8  # noqa: PLR2004


def test_metrics_thread_safe_under_concurrent_writes():
    m = Metrics()

    def worker() -> None:
        for _ in range(100):
            m.record(track="T1", metric_name="steps_total", value=1, kind="counter")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.snapshot()[("T1", "steps_total")] == 1000  # noqa: PLR2004


def test_metrics_export_prometheus_format():
    m = Metrics()
    m.record(track="T1", metric_name="steps_total", value=5, kind="counter")
    text = m.export_prometheus()
    assert "kiki_flow_steps_total" in text
    assert 'track="T1"' in text

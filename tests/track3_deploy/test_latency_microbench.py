import time
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from kiki_flow_core.hooks import RoutingAdapter
from kiki_flow_core.state import FlowState
from kiki_flow_core.track3_deploy.neural_surrogate import NeuralSurrogate
from kiki_flow_core.track3_deploy.query_encoder import QueryEncoder
from kiki_flow_core.track3_deploy.streaming_runner import StreamingRunner


@pytest.mark.slow
def test_latency_sla(tmp_path: Path):
    state_dim = 64
    hidden = 128
    embed_dim = 384
    rng = np.random.default_rng(0)
    tensors = {
        "w1": (rng.standard_normal((state_dim + embed_dim, hidden)) * 0.001).astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "w2": (rng.standard_normal((hidden, hidden)) * 0.001).astype(np.float32),
        "b2": np.zeros(hidden, dtype=np.float32),
        "w3": (rng.standard_normal((hidden, state_dim)) * 0.001).astype(np.float32),
        "b3": np.zeros(state_dim, dtype=np.float32),
    }
    path = tmp_path / "w.safetensors"
    save_file(tensors, str(path))
    surr = NeuralSurrogate.load(path, state_dim=state_dim, embed_dim=embed_dim, hidden=hidden)
    enc = QueryEncoder(use_stub=True)
    routing = RoutingAdapter(publisher=lambda x: None)

    state = FlowState(
        rho={f"{o}:code": np.full(16, 1.0 / 16) for o in ["phono", "lex", "syntax", "sem"]},
        P_theta=np.zeros(8),
        mu_curr=np.full(16, 1.0 / 16),
        tau=0,
        metadata={"track_id": "T3"},
    )
    runner = StreamingRunner(
        surrogate=surr, encoder=enc, routing_adapter=routing, initial_state=state
    )

    # Warm up
    for _ in range(50):
        runner.on_query("warmup")

    latencies_ms: list[float] = []
    for i in range(500):
        t0 = time.perf_counter()
        runner.on_query(f"query_{i}")
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99)]
    assert p50 < 25.0, f"p50 latency regression: {p50:.1f} ms"  # noqa: PLR2004
    assert p99 < 100.0, f"p99 latency regression: {p99:.1f} ms"  # noqa: PLR2004

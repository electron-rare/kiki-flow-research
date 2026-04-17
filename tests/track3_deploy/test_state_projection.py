import numpy as np

from kiki_flow_core.state import FlowState
from kiki_flow_core.track3_deploy.state_projection import flatten, unflatten


def make_state(n: int = 8) -> FlowState:
    return FlowState(
        rho={
            f"{o}:code": np.random.default_rng(0).dirichlet(np.ones(n))
            for o in ["phono", "lex", "syntax", "sem"]
        },
        P_theta=np.zeros(16),
        mu_curr=np.full(n, 1.0 / n),
        tau=7,
        metadata={"track_id": "T3"},
    )


def test_flatten_shape():
    s = make_state()
    flat = flatten(s)
    assert flat.ndim == 1


def test_unflatten_roundtrip():
    s = make_state()
    flat = flatten(s)
    s_back = unflatten(flat, reference=s)
    for k in s.rho:
        np.testing.assert_allclose(s_back.rho[k], s.rho[k], atol=1e-8)
    assert s_back.tau == s.tau
    assert s_back.tau == 7  # noqa: PLR2004

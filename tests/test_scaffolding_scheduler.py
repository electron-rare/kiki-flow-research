import numpy as np
import pytest

from kiki_flow_core.modules.scaffolding_scheduler import ScaffoldingScheduler


def test_scheduler_h_within_bounds():
    s = ScaffoldingScheduler(h_min=1e-3, h_max=1.0, zpd_oracle=lambda errs: 0.5)
    h, _mu = s.next_step(error_profile=np.array([0.1, 0.1, 0.2]))
    assert s.h_min <= h <= s.h_max


def test_scheduler_empty_zpd_returns_h_max():
    """Nothing to learn (errors all near zero) => large step."""
    s = ScaffoldingScheduler(h_min=1e-3, h_max=1.0, zpd_oracle=lambda errs: float(errs.mean()))
    h, _mu = s.next_step(error_profile=np.zeros(5))
    assert h == pytest.approx(s.h_max, rel=1e-2)


def test_scheduler_saturated_zpd_returns_h_min():
    """Errors at ceiling => small cautious step."""
    s = ScaffoldingScheduler(h_min=1e-3, h_max=1.0, zpd_oracle=lambda errs: float(errs.mean()))
    h, _mu = s.next_step(error_profile=np.ones(5))
    assert h == pytest.approx(s.h_min, rel=1e-2)


def test_scheduler_mu_curr_is_normalized_distribution():
    s = ScaffoldingScheduler(h_min=1e-3, h_max=1.0, zpd_oracle=lambda errs: 0.5)
    _h, mu = s.next_step(error_profile=np.array([0.1, 0.5, 0.2, 0.05, 0.3]))
    assert abs(mu.sum() - 1.0) < 1e-6  # noqa: PLR2004
    assert (mu >= 0).all()


def test_scheduler_mu_concentrates_on_high_error_regions():
    s = ScaffoldingScheduler(h_min=1e-3, h_max=1.0, zpd_oracle=lambda errs: 0.5)
    errors = np.array([0.0, 0.0, 0.9, 0.0, 0.0])
    _h, mu = s.next_step(error_profile=errors)
    assert mu[2] == mu.max(), "Curriculum should concentrate where errors are largest"

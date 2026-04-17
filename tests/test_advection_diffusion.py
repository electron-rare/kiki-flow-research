import numpy as np
import pytest

from kiki_flow_core.modules.advection_diffusion import AdvectionDiffusion


@pytest.mark.golden
def test_pure_advection_matches_analytical():
    data = np.load("tests/golden/advection_1d_gaussian.npz")
    x = data["x"]
    rho0 = data["rho0"]
    v_speed = float(data["v_speed"])
    t_final = float(data["t_final"])
    n_steps = 500
    dt = t_final / n_steps
    solver = AdvectionDiffusion(species=None, x_grid=x, diffusion=0.0)
    rho = rho0.copy()
    v_field = np.full_like(x, v_speed)
    for _ in range(n_steps):
        rho = solver.step_1d(rho, v_field=v_field, dt=dt)
    l2 = float(np.sqrt(((rho - data["rho_expected"]) ** 2).sum()))
    # First-order upwind is dissipative; relaxed tolerance reflects numerical diffusion.
    # Tightening below 5e-2 would require a higher-order scheme (WENO, MacCormack).
    assert l2 < 5e-2, f"L2 error too large: {l2}"  # noqa: PLR2004


@pytest.mark.golden
def test_pure_diffusion_matches_analytical():
    data = np.load("tests/golden/diffusion_gaussian.npz")
    x = data["x"]
    rho0 = data["rho0"]
    diff = float(data["D"])
    t_final = float(data["t_final"])
    n_steps = 1000
    dt = t_final / n_steps
    solver = AdvectionDiffusion(species=None, x_grid=x, diffusion=diff)
    rho = rho0.copy()
    v_field = np.zeros_like(x)
    for _ in range(n_steps):
        rho = solver.step_1d(rho, v_field=v_field, dt=dt)
    l2 = float(np.sqrt(((rho - data["rho_expected"]) ** 2).sum()))
    assert l2 < 5e-3, f"L2 error too large: {l2}"  # noqa: PLR2004


def test_mass_conservation_over_1000_steps():
    n = 128
    x = np.linspace(-1, 1, n)
    rho = np.exp(-0.5 * (x / 0.2) ** 2)
    rho /= rho.sum()
    solver = AdvectionDiffusion(species=None, x_grid=x, diffusion=0.005)
    v_field = np.full_like(x, 0.1)
    for _ in range(1000):
        rho = solver.step_1d(rho, v_field=v_field, dt=0.001)
        assert abs(rho.sum() - 1.0) < 1e-4  # noqa: PLR2004


def test_non_negative_density_after_step():
    n = 64
    x = np.linspace(-1, 1, n)
    rho = np.zeros(n)
    rho[n // 2] = 1.0  # Dirac
    solver = AdvectionDiffusion(species=None, x_grid=x, diffusion=0.01)
    v_field = np.full_like(x, 0.5)
    for _ in range(50):
        rho = solver.step_1d(rho, v_field=v_field, dt=0.005)
        assert (rho >= -1e-6).all()  # noqa: PLR2004

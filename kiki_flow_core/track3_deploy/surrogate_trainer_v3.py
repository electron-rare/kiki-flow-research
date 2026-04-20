"""Joint trainer v3 for (encoder + bridge head) with MSE + λ·KL loss.

The encoder is frozen during JAX backprop (only its .encode() output is used).
The bridge head is a fresh JAX MLP (512 -> 256 -> 256 -> 128 tanh) whose
weights are updated via optax AdamW. This keeps the encoder-side flexible
(pure-NumPy hash/distilled, or flax tiny-tf with its own training loop).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from safetensors.numpy import load_file, save_file

from kiki_flow_core.track3_deploy.encoders.base import TextEncoder

_BRIDGE_INPUT_DIM = 512
_BRIDGE_HIDDEN_DIM = 256
_BRIDGE_OUTPUT_DIM = 128
_N_SPECIES = 4
_N_STACKS = 32
_EPS = 1e-8


def _kl_per_species(rho_target: jnp.ndarray, rho_pred: jnp.ndarray) -> jnp.ndarray:
    """KL(target || pred), mean over batch and species. Shapes (B, 4, 32)."""
    return jnp.mean(
        jnp.sum(
            rho_target * (jnp.log(rho_target + _EPS) - jnp.log(rho_pred + _EPS)),
            axis=-1,
        )
    )


def _softmax_per_species(delta: jnp.ndarray) -> jnp.ndarray:
    """Reshape 128-dim to (B, 4, 32) and softmax each species axis."""
    shaped = delta.reshape(-1, _N_SPECIES, _N_STACKS)
    return jax.nn.softmax(shaped, axis=-1)


class _BridgeHead:
    """Pure-JAX MLP: 512 -> 256 -> 256 -> 128 (tanh), skip around H2."""

    @staticmethod
    def init_params(
        seed: int,
        input_dim: int = _BRIDGE_INPUT_DIM,
        hidden: int = _BRIDGE_HIDDEN_DIM,
        output_dim: int = _BRIDGE_OUTPUT_DIM,
    ) -> dict[str, jnp.ndarray]:
        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)
        scale_in = (2.0 / input_dim) ** 0.5
        scale_h = (2.0 / hidden) ** 0.5
        return {
            "W1": jax.random.normal(k1, (input_dim, hidden)) * scale_in,
            "b1": jnp.zeros(hidden),
            "W2": jax.random.normal(k2, (hidden, hidden)) * scale_h,
            "b2": jnp.zeros(hidden),
            "W3": jax.random.normal(k3, (hidden, output_dim)) * scale_h,
            "b3": jnp.zeros(output_dim),
        }

    @staticmethod
    def forward(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
        h1 = jax.nn.gelu(x @ params["W1"] + params["b1"])
        h2 = jax.nn.gelu(h1 @ params["W2"] + params["b2"]) + h1  # skip
        return jnp.tanh(h2 @ params["W3"] + params["b3"])


class JointTrainer:
    """Trainer for bridge head on top of a (frozen) encoder.

    Loss: MSE(delta_pred, delta_target) + lam * KL_per_species(rho_pred, rho_target).
    """

    def __init__(
        self,
        encoder: TextEncoder,
        lam: float = 0.5,
        lr: float = 3e-4,
        seed: int = 0,
    ) -> None:
        self.encoder = encoder
        self.lam = lam
        self.params = _BridgeHead.init_params(seed=seed)
        self.optim = optax.adamw(lr)
        self.opt_state = self.optim.init(self.params)
        # JIT'd inner fns
        self._loss_fn = jax.jit(self._loss_impl)
        self._step_fn = jax.jit(self._step_impl)

    def _features(self, texts: list[str], state_pre: np.ndarray) -> jnp.ndarray:
        """Concat state_pre (128) with encoder output (384) -> 512-dim input."""
        enc = self.encoder.encode(texts)
        return jnp.concatenate([jnp.asarray(state_pre), jnp.asarray(enc)], axis=-1)

    def _loss_impl(
        self,
        params: dict[str, jnp.ndarray],
        features: jnp.ndarray,
        state_pre: jnp.ndarray,
        state_post: jnp.ndarray,
        rho_target: jnp.ndarray,
        lam: float,
    ) -> jnp.ndarray:
        delta_pred = _BridgeHead.forward(params, features)
        target_delta = state_post - state_pre
        mse = jnp.mean((delta_pred - target_delta) ** 2)
        pred_state = state_pre + delta_pred
        rho_pred = _softmax_per_species(pred_state)
        kl = _kl_per_species(rho_target, rho_pred)
        return mse + lam * kl

    def _step_impl(
        self,
        params: dict[str, jnp.ndarray],
        opt_state: optax.OptState,
        features: jnp.ndarray,
        spre: jnp.ndarray,
        spost: jnp.ndarray,
        rho: jnp.ndarray,
        lam: float,
    ) -> tuple[dict[str, jnp.ndarray], optax.OptState, jnp.ndarray]:
        loss_val, grads = jax.value_and_grad(self._loss_impl)(
            params, features, spre, spost, rho, lam
        )
        updates, opt_state = self.optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    def loss(
        self,
        texts: list[str],
        state_pre: np.ndarray,
        state_post: np.ndarray,
        rho_target: np.ndarray,
    ) -> float:
        feats = self._features(texts, state_pre)
        return float(
            self._loss_fn(
                self.params,
                feats,
                jnp.asarray(state_pre),
                jnp.asarray(state_post),
                jnp.asarray(rho_target),
                self.lam,
            )
        )

    def loss_components(
        self,
        texts: list[str],
        state_pre: np.ndarray,
        state_post: np.ndarray,
        rho_target: np.ndarray,
    ) -> tuple[float, float]:
        feats = self._features(texts, state_pre)
        delta_pred = _BridgeHead.forward(self.params, feats)
        target_delta = jnp.asarray(state_post) - jnp.asarray(state_pre)
        mse = float(jnp.mean((delta_pred - target_delta) ** 2))
        pred_state = jnp.asarray(state_pre) + delta_pred
        kl = float(
            _kl_per_species(
                jnp.asarray(rho_target),
                _softmax_per_species(pred_state),
            )
        )
        return mse, kl

    def step(
        self,
        texts: list[str],
        state_pre: np.ndarray,
        state_post: np.ndarray,
        rho_target: np.ndarray,
    ) -> float:
        feats = self._features(texts, state_pre)
        self.params, self.opt_state, loss_val = self._step_fn(
            self.params,
            self.opt_state,
            feats,
            jnp.asarray(state_pre),
            jnp.asarray(state_post),
            jnp.asarray(rho_target),
            self.lam,
        )
        return float(loss_val)

    def save_checkpoint(self, path: Path | str) -> None:
        flat = {f"bridge/{k}": np.asarray(v) for k, v in self.params.items()}
        save_file(flat, str(path))
        # Encoder weights saved alongside
        enc_path = Path(path).with_suffix(".encoder.safetensors")
        self.encoder.save(enc_path)

    def load_checkpoint(self, path: Path | str) -> None:
        flat = load_file(str(path))
        self.params = {k.split("/", 1)[1]: jnp.asarray(v) for k, v in flat.items()}
        enc_path = Path(path).with_suffix(".encoder.safetensors")
        if enc_path.exists():
            self.encoder.load(enc_path)
        self.opt_state = self.optim.init(self.params)

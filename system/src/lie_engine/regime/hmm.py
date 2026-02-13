from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if axis is not None:
        s = np.squeeze(s, axis=axis)
    return s


def _build_features(df: pd.DataFrame) -> np.ndarray:
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    r5 = np.log(close / close.shift(5)).replace([np.inf, -np.inf], np.nan)
    sigma20 = close.pct_change().rolling(20).std(ddof=0)
    dvol20 = vol.pct_change(20)
    feat = pd.concat([r5, sigma20, dvol20], axis=1)
    feat.columns = ["r5", "sigma20", "dvol20"]
    feat = feat.dropna()
    if feat.empty:
        return np.empty((0, 3), dtype=float)
    x = feat.to_numpy(dtype=float)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (x - mu) / sd


@dataclass(slots=True)
class GaussianHMM3:
    n_states: int = 3
    n_iter: int = 30
    tol: float = 1e-4
    random_state: int = 42
    startprob_: np.ndarray = field(init=False)
    transmat_: np.ndarray = field(init=False)
    means_: np.ndarray = field(init=False)
    vars_: np.ndarray = field(init=False)
    _fitted: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states)
        self.transmat_ = np.full((self.n_states, self.n_states), 0.05)
        np.fill_diagonal(self.transmat_, 0.90)
        self.means_ = np.zeros((self.n_states, 3))
        self.vars_ = np.ones((self.n_states, 3))
        self._fitted = False

    def _init_params(self, x: np.ndarray) -> None:
        if x.shape[0] < self.n_states:
            self.means_ = np.vstack([x.mean(axis=0) for _ in range(self.n_states)])
            self.vars_ = np.vstack([np.maximum(x.var(axis=0), 1e-2) for _ in range(self.n_states)])
            return

        q = np.quantile(x[:, 0], [0.2, 0.5, 0.8])
        idx = [np.argmin(np.abs(x[:, 0] - qi)) for qi in q]
        self.means_ = x[idx]
        base_var = np.maximum(x.var(axis=0), 1e-2)
        self.vars_ = np.vstack([base_var.copy() for _ in range(self.n_states)])

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        t = x.shape[0]
        out = np.zeros((t, self.n_states), dtype=float)
        for k in range(self.n_states):
            var = np.maximum(self.vars_[k], 1e-6)
            log_det = np.sum(np.log(2.0 * math.pi * var))
            diff = x - self.means_[k]
            maha = np.sum((diff * diff) / var, axis=1)
            out[:, k] = -0.5 * (log_det + maha)
        return out

    def _forward_backward(self, log_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        t, n = log_b.shape
        log_a = np.log(np.maximum(self.transmat_, 1e-12))
        log_pi = np.log(np.maximum(self.startprob_, 1e-12))

        alpha = np.zeros((t, n), dtype=float)
        beta = np.zeros((t, n), dtype=float)

        alpha[0] = log_pi + log_b[0]
        for i in range(1, t):
            alpha[i] = _logsumexp(alpha[i - 1][:, None] + log_a, axis=0) + log_b[i]

        beta[t - 1] = 0.0
        for i in range(t - 2, -1, -1):
            beta[i] = _logsumexp(log_a + log_b[i + 1][None, :] + beta[i + 1][None, :], axis=1)

        log_prob = float(_logsumexp(alpha[t - 1], axis=0))
        gamma = np.exp(alpha + beta - log_prob)

        xi = np.zeros((t - 1, n, n), dtype=float)
        for i in range(t - 1):
            val = alpha[i][:, None] + log_a + log_b[i + 1][None, :] + beta[i + 1][None, :] - log_prob
            xi[i] = np.exp(val)

        return gamma, xi, log_prob

    def fit(self, x: np.ndarray) -> "GaussianHMM3":
        if x.shape[0] < 10:
            self._init_params(x if x.size else np.zeros((3, 3)))
            self._fitted = True
            return self

        self._init_params(x)
        prev = -np.inf

        for _ in range(self.n_iter):
            log_b = self._log_emission(x)
            gamma, xi, ll = self._forward_backward(log_b)

            self.startprob_ = np.clip(gamma[0], 1e-9, 1.0)
            self.startprob_ /= self.startprob_.sum()

            denom = gamma[:-1].sum(axis=0)[:, None]
            trans = xi.sum(axis=0) / np.maximum(denom, 1e-12)
            trans = np.clip(trans, 1e-9, 1.0)
            self.transmat_ = trans / trans.sum(axis=1, keepdims=True)

            gsum = gamma.sum(axis=0)
            self.means_ = (gamma.T @ x) / np.maximum(gsum[:, None], 1e-12)

            for k in range(self.n_states):
                diff = x - self.means_[k]
                v = (gamma[:, k][:, None] * (diff * diff)).sum(axis=0) / max(gsum[k], 1e-12)
                self.vars_[k] = np.maximum(v, 1e-4)

            if abs(ll - prev) < self.tol:
                break
            prev = ll

        self._fitted = True
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return np.array([[1 / 3, 1 / 3, 1 / 3]], dtype=float)
        if not self._fitted:
            self.fit(x)
        gamma, _, _ = self._forward_backward(self._log_emission(x))
        return gamma


def infer_hmm_state(df: pd.DataFrame) -> dict[str, float]:
    x = _build_features(df)
    if x.shape[0] < 10:
        return {"bull": 0.33, "range": 0.34, "bear": 0.33}

    model = GaussianHMM3()
    model.fit(x)
    probs = model.predict_proba(x)[-1]

    # Map latent states to bull/range/bear by mean return factor ordering.
    returns = model.means_[:, 0]
    idx_sorted = np.argsort(returns)
    bear_idx = idx_sorted[0]
    range_idx = idx_sorted[1]
    bull_idx = idx_sorted[2]

    return {
        "bull": float(probs[bull_idx]),
        "range": float(probs[range_idx]),
        "bear": float(probs[bear_idx]),
    }

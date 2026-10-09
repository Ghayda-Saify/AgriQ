"""Minimal LSTM (NumPy, explicit BPTT) for agricultural price forecasting.

No deep-learning framework is reachable from this environment, so the network
is implemented directly: standard forget / input / output / candidate gates,
trained by backpropagation through time with Adam and gradient clipping.
Single layer, small hidden state -- appropriate for the very short series
available (28 annual observations per crop).
"""
from __future__ import annotations
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


class LSTM:
    def __init__(self, n_in, n_hidden=16, seed=0):
        rng = np.random.default_rng(seed)
        D, H = n_in, n_hidden
        s = 1.0 / np.sqrt(H)
        self.H = H
        self.W = rng.uniform(-s, s, (4 * H, D + H))
        self.b = np.zeros(4 * H)
        self.b[:H] = 1.0                       # forget bias -> remember by default
        self.Wy = rng.uniform(-s, s, (1, H))
        self.by = np.zeros(1)
        self.m = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params().items()}
        self.t = 0

    def params(self):
        return {"W": self.W, "b": self.b, "Wy": self.Wy, "by": self.by}

    def forward(self, X):
        B, T, D = X.shape
        H = self.H
        h = np.zeros((B, H)); c = np.zeros((B, H))
        cache = []
        for t in range(T):
            xh = np.concatenate([X[:, t, :], h], axis=1)
            z = xh @ self.W.T + self.b
            f = sigmoid(z[:, :H]); i = sigmoid(z[:, H:2*H])
            o = sigmoid(z[:, 2*H:3*H]); g = np.tanh(z[:, 3*H:])
            c_prev = c
            c = f * c_prev + i * g
            tc = np.tanh(c)
            h = o * tc
            cache.append((xh, f, i, o, g, c_prev, tc))
        y = (h @ self.Wy.T + self.by).ravel()
        self._h_last = h
        return y, cache

    def backward(self, X, cache, dy):
        B, T, D = X.shape
        H = self.H
        gW = np.zeros_like(self.W); gb = np.zeros_like(self.b)
        gWy = dy[None, :] @ self._h_last
        gby = np.array([dy.sum()])
        dh = dy[:, None] @ self.Wy
        dc = np.zeros((B, H))
        for t in reversed(range(T)):
            xh, f, i, o, g, c_prev, tc = cache[t]
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            df = dc * c_prev; di = dc * g; dg = dc * i
            dz = np.concatenate([df * f * (1 - f), di * i * (1 - i),
                                 do * o * (1 - o), dg * (1 - g ** 2)], axis=1)
            gW += dz.T @ xh
            gb += dz.sum(axis=0)
            dxh = dz @ self.W
            dh = dxh[:, D:]
            dc = dc * f
        return {"W": gW, "b": gb, "Wy": gWy, "by": gby}

    def step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8, clip=5.0):
        self.t += 1
        for k, p in self.params().items():
            g = np.clip(grads[k], -clip, clip)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def fit(self, X, y, epochs=400, lr=0.01, batch=64, seed=0, verbose=False):
        rng = np.random.default_rng(seed)
        n = len(X)
        for ep in range(epochs):
            idx = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, batch):
                b = idx[s:s + batch]
                yh, cache = self.forward(X[b])
                err = yh - y[b]
                tot += float((err ** 2).sum())
                grads = self.backward(X[b], cache, 2 * err / len(b))
                self.step(grads, lr)
            if verbose and ep % 100 == 0:
                print(f"    epoch {ep:4d}  mse {tot/n:.5f}")
        return self

    def predict(self, X):
        return self.forward(X)[0]

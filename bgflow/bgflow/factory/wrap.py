from collections.abc import Iterable
import torch
import numpy as np
from bgflow import Flow


__all__ = ["WrapPeriodic", "SetConstantFlow"]


class WrapPeriodic(torch.nn.Module):
    def __init__(self, net, left=0.0, right=1.0, indices=slice(None)):
        super().__init__()
        self.net = net
        self.left = left
        self.right = right
        self.indices = indices

    def forward(self, x):
        indices = np.arange(x.shape[-1])[self.indices]
        other_indices = np.setdiff1d(np.arange(x.shape[-1]), indices)
        y = x[..., indices]
        cos = torch.cos(2 * np.pi * (y - self.left) / (self.right - self.left))
        sin = torch.sin(2 * np.pi * (y - self.left) / (self.right - self.left))
        x = torch.cat([x[..., other_indices], cos, sin], dim=-1)
        return self.net.forward(x)


class SetConstantFlow(Flow):

    def __init__(self, indices, values, n_event_dims0=1):
        """

        Parameters
        ----------
        indices
        values
        n_event_dims0 : int
            The number of event dims of x[0]. Required to infer the batch shape.
        """
        super().__init__()
        argsort = np.argsort(indices)
        self.indices = [indices[i] for i in argsort]
        values = [values[i] for i in argsort]
        for i, v in enumerate(values):
            self.register_buffer(f"_values_{i}", v)
        self.n_event_dims0 = n_event_dims0

    @property
    def values(self):
        result = []
        i = 0
        while hasattr(self, f"_values_{i}"):
            result.append(getattr(self, f"_values_{i}"))
            i += 1
        return result

    def _forward(self, *xs, **kwargs):
        """insert constants"""
        batch_shape = list(xs[0].shape[:self.n_event_dims0])
        y = list(xs)
        for i, v in zip(self.indices, self.values):
            y.insert(i, v.repeat([*batch_shape, *np.ones_like(v.shape)]))
        dlogp = torch.zeros(batch_shape + [1], device=xs[0].device, dtype=xs[0].dtype)
        return (*y, dlogp)

    def _inverse(self, *xs, **kwargs):
        """remove constants"""
        y = tuple(xs[i] for i, z in enumerate(xs) if i not in self.indices)
        batch_shape = list(y[0].shape[:self.n_event_dims0])
        dlogp = torch.zeros(batch_shape + [1], device=y[0].device, dtype=y[0].dtype)
        return (*y, dlogp)


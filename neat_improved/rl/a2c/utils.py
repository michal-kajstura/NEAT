import numpy as np
import torch


def init(module, weight_init, bias_init, gain: float = 1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def explained_variance(ypred: torch.Tensor, y: torch.Tensor):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    ypred = ypred.detach().cpu().numpy().swapaxes(1, 0).flatten()
    y = y.detach().cpu().numpy().swapaxes(1, 0).flatten()

    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

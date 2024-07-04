import numpy as np
import torch
import sklearn

from pytabkit.models.training.metrics import Metrics


def test_pinball():
    torch.manual_seed(0)
    y_pred = torch.randn(100)[:, None]
    y = torch.randn(100)[:, None]
    loss = Metrics.apply(y_pred, y, 'pinball(0.95)').item()
    sklearn_loss = sklearn.metrics.mean_pinball_loss(y.numpy(), y_pred.numpy(), alpha=0.95)
    assert np.isclose(loss, sklearn_loss)

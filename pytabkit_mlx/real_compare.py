"""Real pytabkit vs MLX port on IDENTICAL train/val sets.

Real model gets val_idxs = the port's exact split indices, same n_epochs.
RNG still differs (torch vs numpy init/shuffle), so this checks ballpark +
curve behavior, not exact equality.

Run: ./.venv/bin/python -m pytabkit_mlx.real_compare
"""
import sys
from pathlib import Path
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import load_diabetes, load_breast_cancer
from pytabkit import RealMLP_TD_Regressor, RealMLP_TD_Classifier
from pytabkit_mlx.api import (RealMLP_MLX_Regressor, RealMLP_MLX_Classifier,
                              pytabkit_split)

N_EPOCHS = 64


def compare_reg():
    X, y = load_diabetes(return_X_y=True)
    itr, ite = pytabkit_split(len(y), 0, 0.8)
    va = pytabkit_split(len(itr), 0, 0.8)[1]
    t0 = time.time()
    real = RealMLP_TD_Regressor(random_state=0, n_epochs=N_EPOCHS, device='cpu',
                                verbosity=0)
    real.fit(X[itr], y[itr], val_idxs=va)
    t_real = time.time() - t0
    pred_real = real.predict(X[ite])
    rmse_real = float(np.sqrt(np.mean((pred_real - y[ite]) ** 2)))
    t0 = time.time()
    mine = RealMLP_MLX_Regressor(random_state=0, n_epochs=N_EPOCHS)
    mine.fit(X[itr], y[itr])
    t_mine = time.time() - t0
    pred_mine = mine.predict(X[ite])
    rmse_mine = float(np.sqrt(np.mean((pred_mine - y[ite]) ** 2)))
    corr = float(np.corrcoef(pred_real, pred_mine)[0, 1])
    print(f'diabetes[{N_EPOCHS}ep] real_rmse={rmse_real:.2f} ({t_real:.1f}s) '
          f'mlx_rmse={rmse_mine:.2f} ({t_mine:.1f}s) pred_corr={corr:.4f}',
          flush=True)
    return rmse_real, rmse_mine


def compare_class():
    X, y = load_breast_cancer(return_X_y=True)
    itr, ite = pytabkit_split(len(y), 0, 0.8)
    va = pytabkit_split(len(itr), 0, 0.8)[1]
    t0 = time.time()
    real = RealMLP_TD_Classifier(random_state=0, n_epochs=N_EPOCHS, device='cpu',
                                 verbosity=0)
    real.fit(X[itr], y[itr], val_idxs=va)
    t_real = time.time() - t0
    acc_real = float((real.predict(X[ite]) == y[ite]).mean())
    t0 = time.time()
    mine = RealMLP_MLX_Classifier(random_state=0, n_epochs=N_EPOCHS)
    mine.fit(X[itr], y[itr])
    t_mine = time.time() - t0
    acc_mine = float((mine.predict(X[ite]) == y[ite]).mean())
    agree = float((real.predict(X[ite]) == mine.predict(X[ite])).mean())
    print(f'cancer[{N_EPOCHS}ep] real_acc={acc_real:.4f} ({t_real:.1f}s) '
          f'mlx_acc={acc_mine:.4f} ({t_mine:.1f}s) agree={agree:.4f}', flush=True)
    return acc_real, acc_mine


if __name__ == '__main__':
    compare_reg()
    compare_class()

"""End-to-end smoke test: fit/predict on sklearn datasets + categoricals.

Run: .venv/bin/python -m pytabkit_mlx.e2e
"""
import sys
from pathlib import Path
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pytabkit_mlx.api import RealMLP_MLX_Classifier, RealMLP_MLX_Regressor


def test_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    t0 = time.time()
    clf = RealMLP_MLX_Classifier(n_epochs=32, random_state=0)
    clf.fit(Xtr, ytr)
    acc = (clf.predict(Xte) == yte).mean()
    print(f'breast_cancer acc={acc:.4f} ({time.time()-t0:.1f}s)', flush=True)
    assert acc > 0.90, acc
    proba = clf.predict_proba(Xte)
    assert proba.shape == (len(yte), 2) and np.allclose(proba.sum(1), 1)


def test_categoricals():
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        'num1': rng.normal(size=n),
        'num2': rng.normal(size=n),
        'color': rng.choice(['r', 'g', 'b'], size=n),           # one-hot path
        'city': rng.choice([f'c{i}' for i in range(30)], size=n),  # embedding path
        'flag': pd.array(rng.choice([True, False], size=n), dtype='boolean'),  # nullable bool -> cat
    })
    y = ((df['num1'] > 0).astype(int) ^ (df['color'] == 'r').astype(int)).to_numpy()
    clf = RealMLP_MLX_Classifier(n_epochs=64, random_state=0)
    clf.fit(df, y)
    acc = (clf.predict(df) == y).mean()
    print(f'categorical train acc={acc:.4f}', flush=True)
    assert acc > 0.9, acc
    # unseen + missing categories must not crash
    df2 = df.copy()
    df2.loc[0, 'city'] = 'never-seen'
    df2.loc[1, 'color'] = None
    p = clf.predict(df2)
    assert len(p) == n


def test_diabetes():
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    X, y = load_diabetes(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    t0 = time.time()
    reg = RealMLP_MLX_Regressor(n_epochs=64, random_state=0)
    reg.fit(Xtr, ytr)
    pred = reg.predict(Xte)
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    base = float(np.sqrt(np.mean((ytr.mean() - yte) ** 2)))
    print(f'diabetes rmse={rmse:.2f} (mean-pred {base:.2f}, {time.time()-t0:.1f}s)', flush=True)
    assert rmse < base


def test_seed_determinism():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, ytr, Xte = X[:400], y[:400], X[400:450]
    a = RealMLP_MLX_Classifier(n_epochs=8, random_state=0).fit(Xtr, ytr).predict_proba(Xte)
    b = RealMLP_MLX_Classifier(n_epochs=8, random_state=0).fit(Xtr, ytr).predict_proba(Xte)
    assert (a == b).all(), 'same seed must give identical predictions'
    c = RealMLP_MLX_Classifier(n_epochs=8, random_state=1).fit(Xtr, ytr).predict_proba(Xte)
    assert (a != c).any(), 'different seeds must differ'
    # None = nondeterministic but must run (all streams from fresh entropy)
    n = RealMLP_MLX_Classifier(n_epochs=2, random_state=None).fit(Xtr, ytr).predict_proba(Xte)
    assert n.shape == (50, 2) and np.allclose(n.sum(1), 1)
    print('seed determinism OK', flush=True)


if __name__ == '__main__':
    test_breast_cancer()
    test_categoricals()
    test_diabetes()
    test_seed_determinism()
    print('E2E PASSED')

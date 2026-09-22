"""RealMLP-TD architecture in MLX.

Faithful port of the default RealMLP construction
(pytabkit/models/nn_models/models.py NNFactory + nn.py fitters) for the
TD (tuned-default) configuration:

  per-num-feature PBLD embedding (cos-bias, linear, densenet):
      z = cos(2*pi*x*w1 + b1)            w1 ~ N(0, 0.1^2) [F,1,16]
                                         b1 ~ U(-pi, pi) [F,1,16]
      h = z@w2 + b2                      w2, b2 ~ U(-1,1)/sqrt(16), [F,16,3]/[F,1,3]
      out = concat(h.flatten, x)         -> 4 dims per numerical feature
  + one-hot cats (cat_size <= 9) / learned embeddings (dim 8, N(0,1) init)
  + front diagonal scale (init 1)
  + 3 x [NTK linear 256 + he+5 bias + parametric act + dropout]
  + head linear (NTK + he+5 bias)

  NTK param:  forward uses W_eff = W_raw / sqrt(in_features).
  'std' init: W = randn / per-col-std(X_train_layer @ (W_raw/sqrt(in))).
  he+5 bias:  bias_j = -weighted avg of 5 random train pre-activations
              (exponential weights), per output unit.
  parametric act: out = x + (f(x) - x) * w, w init 1.
      f = SELU (classification) or Mish (regression).

Parameters are a nested dict of mx arrays; metadata (lr/wd factors) lives in
PARAM_META. This makes torch<->MLX conversion mechanical (see convert.py).
"""

import math

import mlx.core as mx
import numpy as np

SELU_SCALE = 1.0507009873554805
SELU_ALPHA = 1.6732632423543772

PLR_H1 = 16   # plr_hidden_1
PLR_H2 = 3    # plr_hidden_2 (4) minus densenet feature
PLR_SIGMA = 0.1
EMB_DIM = 8

# (lr_factor, wd_factor) per parameter, matching pytabkit hyper_factors.
# wd applies to everything except biases (bias_wd_factor=0). NOTE: in the
# decoupled wd step, pytabkit multiplies by these factors twice
# (see train.py), which is replicated for exactness.
PARAM_META = {
    'plr_w1': (0.1, 1.0), 'plr_b1': (0.1, 1.0),
    'plr_w2': (0.1, 1.0), 'plr_b2': (0.1, 1.0),
    'emb': (1.0, 1.0),
    'front_scale': (6.0, 1.0),
    'w': (1.0, 1.0), 'b': (0.1, 0.0), 'act_w': (0.1, 1.0),
    'head_w': (1.0, 1.0), 'head_b': (0.1, 0.0),
}


def selu(x):
    return mx.where(x > 0, SELU_SCALE * x,
                    SELU_SCALE * SELU_ALPHA * (mx.exp(x) - 1))


def mish(x):
    return x * mx.tanh(mx.logaddexp(0, x))


def get_act(name):
    if name == 'selu':
        return selu
    if name == 'mish':
        return mish
    raise ValueError(f'Unknown act "{name}"')


def init_params(n_num, n_onehot, emb_cat_sizes, n_out, hidden_sizes,
                act_name, rng):
    """Build params dict. Xn_init: preprocessed train numericals (numpy)."""
    P = {}

    def var(name, arr, kind):
        P[name] = mx.array(np.asarray(arr, dtype=np.float32))

    # --- PBLD numerical embeddings (data-independent init) ---
    F = n_num
    if F > 0:
        var('plr_w1', PLR_SIGMA * rng.randn((F, 1, PLR_H1)), 'plr_w1')
        var('plr_b1', math.pi * rng.uniform(-1, 1, (F, 1, PLR_H1)), 'plr_b1')
        var('plr_w2', rng.uniform(-1, 1, (F, PLR_H1, PLR_H2)) / math.sqrt(PLR_H1), 'plr_w2')
        var('plr_b2', rng.uniform(-1, 1, (F, 1, PLR_H2)) / math.sqrt(PLR_H1), 'plr_b2')
    # --- categorical embeddings ---
    for i, cs in enumerate(emb_cat_sizes):
        P[f'emb_table_{i}'] = mx.array(
            rng.randn((cs, EMB_DIM)).astype(np.float32))
    P['n_emb_tables'] = len(emb_cat_sizes)
    P['hidden_sizes'] = list(hidden_sizes)
    P['act_name'] = act_name
    P['n_num'] = F
    return P


def pbld_forward(P, Xn):
    """Xn: [B, F] -> [B, F*4]. Feature dim is treated as batch (as in
    pytabkit's PLREmbeddingsLayerCosBias: transpose, matmul, transpose)."""
    if P['n_num'] == 0:
        return mx.zeros((Xn.shape[0], 0))
    x = Xn.transpose(1, 0)[:, :, None]  # [F, B, 1]
    z = mx.cos(2 * math.pi * x * P['plr_w1'] + P['plr_b1'])  # [F,B,16]
    h = z @ P['plr_w2'] + P['plr_b2']  # [F,B,3]
    h = h.transpose(1, 0, 2).reshape(h.shape[1], -1)  # [B, F*3]
    return mx.concatenate([h, Xn], axis=1)


def build_input(P, Xn, Xo, Xc):
    parts = [pbld_forward(P, Xn)]
    if Xo.shape[1] > 0:
        parts.append(Xo)
    for i in range(P['n_emb_tables']):
        parts.append(P[f'emb_table_{i}'][Xc[:, i]])
    if len(parts) == 1:
        return parts[0]
    return mx.concatenate(parts, axis=1)


def layer_forward(P, x, i, act_fn, training, p_drop, mask=None):
    in_f = x.shape[-1]
    W_eff = P[f'w{i}'] / math.sqrt(in_f)
    z = x @ W_eff + P[f'b{i}']
    a = z + (act_fn(z) - z) * P[f'act_w{i}']
    if training and p_drop > 0.0:
        if mask is None:
            m = mx.random.uniform(shape=a.shape) >= p_drop
        else:
            m = mask
        a = mx.where(m, a / (1.0 - p_drop), mx.zeros_like(a))
    return a


def forward(P, Xn, Xo, Xc, training=False, p_drop=0.0, masks=None):
    act_fn = get_act(P['act_name'])
    x = build_input(P, Xn, Xo, Xc) * P['front_scale']
    for i in range(len(P['hidden_sizes'])):
        x = layer_forward(P, x, i, act_fn, training, p_drop,
                          mask=None if masks is None else masks[i])
    in_f = x.shape[-1]
    out = x @ (P['head_w'] / math.sqrt(in_f)) + P['head_b']
    return out


def _heplus_bias(Z, rng):
    """he+5 bias init, matching BiasFitter.heplus_bias draw order exactly:
    one randint (out,5) block + one exponential (out,5) block."""
    out_f = Z.shape[1]
    n = Z.shape[0]
    idx = rng.integers(0, n, (out_f, 5))
    e = rng.exponential((out_f, 5))
    e /= e.sum(axis=1, keepdims=True)
    return -(e * Z[idx, np.arange(out_f)[:, None]]).sum(axis=1)


def init_data_dependent(P, S_init, rng):
    """Init front scale (ones), hidden + head weights ('std') and biases
    ('he+5') using train activations pushed through layer by layer.

    S_init: numpy [N, D] model-input matrix (PBLD + one-hot + embeddings
    with initial tables) for a sample of training rows.
    """
    S = np.asarray(S_init, dtype=np.float64)
    D = S.shape[1]
    P['front_scale'] = mx.ones((D,), dtype=mx.float32)
    sizes = P['hidden_sizes']
    for i, out_f in enumerate(sizes):
        in_f = S.shape[1]
        W_raw = rng.randn((in_f, out_f))
        eff = W_raw / math.sqrt(in_f)
        Z = S @ eff
        s = Z.std(axis=0, keepdims=True)  # population std, like torch correction=0
        s[s == 0] = 1.0
        P[f'w{i}'] = mx.array((W_raw / s).astype(np.float32))
        Z = S @ (W_raw / s / math.sqrt(in_f))
        b = _heplus_bias(Z, rng)
        P[f'b{i}'] = mx.array(b.astype(np.float32))
        P[f'act_w{i}'] = mx.ones((out_f,), dtype=mx.float32)
        S = S @ (W_raw / s / math.sqrt(in_f)) + b[None, :]
        S = _numpy_act(S, P['act_name'], np.ones(out_f))
    # head
    in_f = S.shape[1]
    n_out = P['_n_out']
    W_raw = rng.randn((in_f, n_out))
    Z = S @ (W_raw / math.sqrt(in_f))
    s = Z.std(axis=0, keepdims=True)
    s[s == 0] = 1.0
    P['head_w'] = mx.array((W_raw / s).astype(np.float32))
    Z = S @ (W_raw / s / math.sqrt(in_f))
    b = _heplus_bias(Z, rng)
    P['head_b'] = mx.array(b.astype(np.float32))
    mx.eval(P)
    return P


def _numpy_act(S, act_name, w):
    if act_name == 'selu':
        f = np.where(S > 0, SELU_SCALE * S,
                     SELU_SCALE * SELU_ALPHA * (np.exp(S) - 1))
    elif act_name == 'mish':
        f = S * np.tanh(np.logaddexp(0, S))
    else:
        raise ValueError(act_name)
    return S + (f - S) * w[None, :]


def param_meta(name):
    """Metadata (lr_factor, wd_factor) for a parameter key."""
    if name in ('plr_w1', 'plr_b1', 'plr_w2', 'plr_b2'):
        return PARAM_META[name]
    if name.startswith('emb_table_'):
        return PARAM_META['emb']
    if name == 'front_scale':
        return PARAM_META['front_scale']
    if name.startswith('head_w'):
        return PARAM_META['head_w']
    if name.startswith('head_b'):
        return PARAM_META['head_b']
    if name.startswith('act_w'):
        return PARAM_META['act_w']
    if name.startswith('w'):
        return PARAM_META['w']
    if name.startswith('b'):
        return PARAM_META['b']
    raise KeyError(name)


def array_params(P):
    """Dict of only the mx.array parameters (excludes config scalars)."""
    return {k: v for k, v in P.items() if isinstance(v, mx.array)}

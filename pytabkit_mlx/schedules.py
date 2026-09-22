"""Learning-rate-style schedules ported from pytabkit.models.training.scheduling.

Only the schedules used by RealMLP-TD are implemented:
  - 'coslog4' (lr):  0.5 * (1 - cos(2*pi*log2(1 + 15*t)))
  - 'flat_cos' (dropout p, weight decay): 1 for t < 0.5, then cosine 1 -> 0
  - 'constant' / 'flat': 1

All take progress t in [0, 1] (fraction of total training steps) and return
a multiplier for the base value. Matches pytabkit's FunctionSchedule /
ScheduleSequence semantics exactly (verified in parity.py).
"""

import math


def coslog4(t: float) -> float:
    return 0.5 * (1.0 - math.cos(2 * math.pi * math.log2(1 + 15 * t)))


def flat_cos(t: float) -> float:
    if t < 0.5:
        return 1.0
    u = (t - 0.5) / 0.5
    return 0.5 * (1.0 + math.cos(math.pi * u))


def constant(t: float) -> float:
    return 1.0


def get_schedule(name: str):
    if name == 'coslog4':
        return coslog4
    if name in ('flat_cos', 'flat-cos'):
        return flat_cos
    if name in ('constant', 'flat'):
        return constant
    raise ValueError(f'Unknown schedule "{name}" (pytabkit_mlx supports coslog4/flat_cos/constant)')

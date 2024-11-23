from typing import List, Tuple, Callable


def bilin_int(x: float, values: List[Tuple[float, float]]) -> float:
    # integrates a bilinear interpolation of the values
    sum_of_integrals = 0.0
    x0, y0 = values[0]
    for x1, y1 in values[1:]:
        if x <= x0:
            return sum_of_integrals
        if x <= x1:
            y1 = y0 + (x-x0)/(x1-x0)*(y1-y0)
            x1 = x

        sum_of_integrals += (x1-x0) * (y1+y0) / 2
        x0, y0 = x1, y1

    return sum_of_integrals


def bisection_find(f: Callable[[float], float], y: float, xmin: float, xmax: float, n=50) -> float:
    # find x with f(x) = y, assuming increasing f
    a = xmin
    b = xmax
    c = (a+b)/2  # middle

    fa = f(a)
    fb = f(b)
    fc = f(c)

    if fa >= y:
        return a
    if fb <= y:
        return b

    for _ in range(n):
        if fc >= y:
            b, fb = c, fc
        else:
            a, fa = c, fc

        c = (a+b)/2
        fc = f(c)

    return c


def more_percep_uniform_hue(x: float) -> float:
    """
    Returns a hue-value that should change perceptually somewhat uniformly with x
    :param x: a value between 0 and 1.
    :return: Hue value for HSV space.
    """
    # eye-balled perceptual "rate of change" scores at different hues
    hue_percep_deriv = [(0, 0.3), (30, 0.6), (60, 1.0), (90, 0.3), (150, 0.3), (180, 0.8), (220, 0.4), (260, 0.4), (280, 0.8),
           (300, 0.6), (360, 0.3)]
    f = lambda val: bilin_int(val, hue_percep_deriv)
    fmax = f(360)
    return bisection_find(f, x*fmax, 0, 360)/360

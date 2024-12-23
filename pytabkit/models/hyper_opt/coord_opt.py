from pathlib import Path

import numpy as np
from typing import Union, Callable, Any, Optional, Dict, Tuple

from pytabkit.models import utils
from pytabkit.models.hyper_opt.hyper_optimizers import HyperOptimizer


# implementing a custom coordinate-descent style hyperparameter optimizer


def identity(x):
    return x


class Hyperparameter:
    def __init__(self, start_value: Union[int, float], min_step_size: Union[int, float], importance: float,
                 log_scale: bool = False, only_int: bool = False,
                 min_value: Union[int, float] = -np.inf, max_value: Union[int, float] = np.inf,
                 out_func: Callable[[Any], Any] = None, max_step_size: float = np.inf):
        # if log_scale=True, min_value, max_value, min_step_size, and max_step_size are on the log scale,
        # i.e., min_value can still be negative
        # in this case, the values will be exponentiated at the end
        self.start_value = start_value
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.importance = importance
        self.log_scale = log_scale
        self.only_int = only_int
        self.min_value = min_value
        self.max_value = max_value
        self.out_func = out_func or identity
        self.tfm = (lambda x: np.exp(x)) if log_scale else identity
        self.inv_tfm = (lambda x: np.log(x)) if log_scale else identity
        self.quant_tfm = (lambda x: round(x)) if only_int else identity
        # if log_scale:
        #     self.min_value = np.log(min_value) if 0 < min_value < np.inf else -np.inf
        #     self.max_value = np.log(max_value) if 0 < max_value < np.inf else np.inf
        if self.log_scale and self.only_int:
            # need to avoid having values < 0 for which round(exp(value)) = 0, which is not representable in log-space
            self.min_value = max(self.min_value, 0.0)

    def adjust_step_size(self, current_value: float, step_size: float) -> Optional[float]:
        # should return suggested step size that satisfies all constraints, or None if no suitable step size is found

        # We have three constraints: step size limit, min_value/max-value, and quantization.
        # Updating each of them could violate one of the others.
        # do a loop and check if all three are satisfied
        # if it doesn't work after a certain number of iterations, we fail and return None
        for i in range(5):
            updated = False
            step_size_sign = np.sign(step_size)

            # check min_step_size / max_step_size
            if np.abs(step_size) < self.min_step_size - 1e-8:
                step_size = step_size_sign * self.min_step_size
                updated = True
            if np.abs(step_size) > self.max_step_size + 1e-8:
                step_size = step_size_sign * self.max_step_size
                updated = True

            # check min_value / max_value
            candidate = current_value + step_size
            if candidate < self.min_value - 1e-8:
                candidate = self.min_value
                updated = True
            elif candidate > self.max_value + 1e-8:
                candidate = self.max_value
                updated = True
            step_size = candidate - current_value
            
            print(f'CoordOpt: {self.min_value=}, {self.max_value=}, {self.start_value=}')
            print(f'CoordOpt: {current_value=}, {candidate=}')

            curr_t = self.tfm(current_value)
            cand_t = self.tfm(candidate)
            curr_q = self.quant_tfm(curr_t)
            cand_q = self.quant_tfm(cand_t)

            if curr_q == cand_q:
                cand_q = curr_q + step_size_sign
                if self.log_scale and self.only_int and cand_q <= 0.5:
                    return None  # curr_q is 1 and we want to make cand_q = 0 but this doesn't exist in log scale
                step_size = self.inv_tfm(cand_q) - current_value
                updated = True

            if not updated:
                # step size fulfilled all three constraints in this loop and hence has not been updated
                return step_size
        return None  # did not find a step size that fulfills all constraints

    def apply_tfms(self, x: Any) -> Any:
        return self.out_func(self.quant_tfm(self.tfm(x)))


class CoordOptimizerImpl:
    # potential improvements:
    # increase the importances in an UCB-style
    # in coord_opt_idx allow to explore the reverse direction if the first step in the previous direction fails
    def __init__(self, f: Callable[[Dict], Tuple[float, Any]], space: Dict[str, Hyperparameter], n_steps: int,
                 beta: float = 0.5, step_dec_factor: float = 0.5, step_inc_factor: float = 2.0,
                 initial_step_multiplier: float = 8.0):
        self.f = f
        self.space = space
        self.n_steps = n_steps
        self.n_f_evals = 0

        if n_steps <= 0:
            raise ValueError(f'CoordOptimizerImpl: Got {n_steps=} but need n_steps > 0')

        # hyperparameters of the HPO method
        self.beta = beta
        self.step_dec_factor = step_dec_factor
        self.step_inc_factor = step_inc_factor
        self.initial_step_multiplier = initial_step_multiplier
        self.max_coord_opt_steps = 10

        self.keys = [k for k, v in space.items()]  # preserve the order in space
        self.d = len(self.keys)
        self.hps = [space[key] for key in self.keys]
        self.prior_importances = [hp.importance for hp in self.hps]
        self.priorities = np.argsort(np.asarray(self.prior_importances))[::-1]
        self.importances = np.zeros(self.d)
        self.min_step_sizes = np.asarray([hp.min_step_size for hp in self.hps])
        self.hp_values = np.asarray([hp.start_value for hp in self.hps])
        self.step_sizes = self.initial_step_multiplier * self.min_step_sizes
        for idx in range(self.d):
            # adjust direction of step sizes
            if self.hp_values[idx] - self.hps[idx].min_value > self.hps[idx].max_value - self.hp_values[idx]:
                # there is more space in the negative direction, start in the other direction
                self.step_sizes[idx] *= -1

        # current best hyperparameter values  (before transformation, i.e., can be in log-space)
        self.evaluated_hp_values = []  # to avoid evaluating the same point twice
        # eval loss on starting values
        self.loss, self.additional_info = self.eval(self.hp_values)
        self.blocked_directions = np.zeros(self.d, dtype=np.int32)

    def suggest(self, new_hp_values) -> float:
        # return loss difference, update optimum if necessary etc.
        # unblock variables if new optimum is found
        new_loss, new_additional_info = self.eval(new_hp_values)
        loss_diff = new_loss - self.loss
        if new_loss < self.loss:
            # update parameters
            self.loss = new_loss
            self.additional_info = new_additional_info
            self.hp_values = new_hp_values
            # unblock all coordinates
            print(f'CoordOpt: Unblocking all coordinates')
            self.blocked_directions = np.zeros(self.d, dtype=np.int32)

        return loss_diff

    def convert_hp_values(self, values: np.ndarray) -> Dict[str, Any]:
        return {key: hp.apply_tfms(value) for (key, value, hp) in zip(self.keys, values, self.hps)}

    def eval(self, new_hp_values: np.ndarray) -> Tuple[float, Any]:
        # convert hyperparameters, call function, increase step counter, raise error if step count is full
        if self.n_f_evals >= self.n_steps:
            raise StopIteration()
        self.n_f_evals += 1
        print(f'CoordOpt: Evaluating hyperparameters in step {self.n_f_evals}: {new_hp_values}')
        self.evaluated_hp_values.append(new_hp_values)
        converted = {key: hp.apply_tfms(value) for (key, value, hp) in zip(self.keys, new_hp_values, self.hps)}
        return self.f(converted)

    def already_evaluated(self, new_hp_values: np.ndarray) -> bool:
        """
        :param new_hp_values: New hyperparameter values that should be tried.
        :return: True if these hyperparameters have already been evaluated before
        """
        for old_hp_values in self.evaluated_hp_values:
            if np.allclose(new_hp_values, old_hp_values):
                return True

        return False

    def coord_opt_idx(self, idx: int):
        # implicitly update importance
        # keep track of step? or use an exception to break when the step count has finished?

        for i in range(self.max_coord_opt_steps):
            print(f'CoordOpt: Optimizing coordinate {idx}, step {i}')
            # loop while line search over coordinate still finds an improvement

            # adjust step size
            adj_step = self.hps[idx].adjust_step_size(current_value=self.hp_values[idx], step_size=self.step_sizes[idx])
            if adj_step is None:
                print(f'CoordOpt: adj_step is None')
                # no suitable step size was found, for example because the boundary is reached
                self.step_sizes[idx] = -self.step_dec_factor * self.step_sizes[idx]
                # if this would bring us below the minimum step size, block the variable
                if np.abs(self.step_sizes[idx]) < self.hps[idx].min_step_size:
                    print(f'CoordOpt: Blocking coordinate {idx}')
                    self.blocked_directions[idx] += 1
                return

            # make step with suggest()
            new_hp_values = np.copy(self.hp_values)
            new_hp_values[idx] += adj_step
            if self.already_evaluated(new_hp_values):
                print(f'CoordOpt: Already evaluated hyperparameters')
                self.step_sizes[idx] = -self.step_dec_factor * adj_step
                self.blocked_directions[idx] += 1
                return
            loss_diff = self.suggest(new_hp_values)

            # update importance
            self.importances[idx] = self.beta * self.importances[idx] + (1 - self.beta) * np.abs(loss_diff)

            if loss_diff < 0:
                print(f'CoordOpt: Loss decreased')
                self.step_sizes[idx] = self.step_inc_factor * adj_step
            else:
                print(f'CoordOpt: Loss did not decrease')
                # if loss didn't reduce, *= - step_dec_factor, return
                self.step_sizes[idx] = -self.step_dec_factor * adj_step
                # if this would bring us below the minimum step size, block the variable
                if np.abs(self.step_sizes[idx]) < self.hps[idx].min_step_size:
                    print(f'CoordOpt: Blocking coordinate {idx}')
                    self.blocked_directions[idx] += 1
                return

    def run(self) -> None:
        # wrap everything in try/catch for termination

        try:
            while True:
                # select best index according to importance
                if np.all(self.blocked_directions >= 2):
                    print(f'CoordOpt: Reached a local optimum')
                    return
                if len(self.priorities) > 0:
                    hp_idx = self.priorities[0]
                    self.priorities = self.priorities[1:]
                else:
                    # print(f'{self.importances=}')
                    importances = np.copy(self.importances)
                    importances[self.blocked_directions >= 2] = -1.0
                    hp_idx = np.argmax(importances)
                    if self.blocked_directions[hp_idx] >= 2:
                        raise RuntimeError('CoordOpt: selected blocked index, this should not occur')

                # run coord_opt_idx on the index
                self.coord_opt_idx(hp_idx)
        except StopIteration:
            return


class CoordOptimizer(HyperOptimizer):
    class CoordOptFuncWrapper:
        def __init__(self, f: Callable[[dict], Tuple[float, Any]], fixed_params: Dict[str, Any]):
            self.f = f
            self.fixed_params = fixed_params

        def __call__(self, params: Dict[str, Any], seed: int = 0):
            params = utils.join_dicts(params, self.fixed_params)
            loss, additional_info = self.f(params)
            return np.inf if np.isnan(loss) else loss, None

    def __init__(self, space: Dict[str, Hyperparameter], fixed_params: Dict[str, Any], n_hyperopt_steps: int = 50, **config):
        super().__init__(n_hyperopt_steps=n_hyperopt_steps)
        self.space = space
        self.n_hyperopt_steps = n_hyperopt_steps
        self.fixed_params = fixed_params
        self.config = config

    def _optimize_impl(self, f: Callable[[dict], Tuple[float, Any]], seed: int) -> None:
        fn = CoordOptimizer.CoordOptFuncWrapper(f, self.fixed_params)

        opt = CoordOptimizerImpl(fn, self.space, n_steps=self.n_hyperopt_steps)
        opt.run()

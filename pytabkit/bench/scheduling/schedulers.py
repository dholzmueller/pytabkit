import copy
import sys
import time
from typing import List, Dict, Union

import numpy as np

from pytabkit.bench.scheduling.execution import RayJobManager
from pytabkit.bench.scheduling.jobs import AbstractJob
from pytabkit.bench.scheduling.resource_manager import JobInfo


def format_length_s(duration: float) -> str:
    seconds = int(duration)
    minutes = seconds // 60
    seconds -= minutes * 60
    hours = minutes // 60
    minutes -= hours * 60
    days = hours // 24
    hours -= days * 24

    result = f'{seconds}s'
    if minutes > 0:
        result = f'{minutes}m' + result
    if hours > 0:
        result = f'{hours}h' + result
    if days > 0:
        result = f'{days}d' + result

    return result


def format_date_s(time_s: float) -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_s))


class BaseJobScheduler:
    """
    Base scheduler class where the logic for selecting which jobs should be run next still has to be implemented.
    Contains functionality for printing intermediate states and the main loop in run().
    """
    def __init__(self, job_manager: RayJobManager):
        self.start_time = time.time()
        self.job_manager = job_manager
        self.job_infos: List[JobInfo] = []

    def _submit_more_jobs(self) -> None:
        # to be implemented in subclasses
        raise NotImplementedError()

    def add_jobs(self, jobs: List[AbstractJob]):
        for job in jobs:
            self.job_infos.append(JobInfo(job, job_id=len(self.job_infos)))

    def run(self):
        if len(self.job_infos) == 0:
            print(f'No jobs to run')
            return

        self.job_manager.start()
        self._print_start()

        while self._has_unfinished_jobs():
            self._submit_more_jobs()
            self._print_progress()

            wait_period_s = 30
            finished_job_infos = self.job_manager.pop_finished_job_infos(timeout_s=wait_period_s)
            if len(finished_job_infos) == 0:
                # no jobs finished after wait_period_s, print a running report and then wait for longer
                self._print_running_jobs()
                finished_job_infos = self.job_manager.pop_finished_job_infos()

            for job_info in finished_job_infos:
                # update the status of the job infos that have been finished
                self.job_infos[job_info.job_id] = job_info

            # todo: register finished job infos in self

        self._print_end()

        self.job_manager.terminate()

    def _has_unfinished_jobs(self) -> bool:
        return any(not ji.is_finished() for ji in self.job_infos)

    def _print_start(self):
        self.start_time = time.time()
        print(
            f'############################### START REPORT ##################################\n'
            f'# Start date: {format_date_s(self.start_time)}\n'
            f'# Number of jobs: {len(self.job_infos)}\n'
            f'###############################################################################',
            flush=True
        )

    def _print_end(self):
        end_time = time.time()
        duration = end_time - self.start_time
        group_stats = self._compute_group_stats()
        ram_factors = [ji.job_result.max_cpu_ram_gb / ji.assigned_resources.get_cpu_ram_gb()
                       for ji in self.job_infos]
        ram_factors.sort(reverse=True)
        if len(ram_factors) > 5:
            ram_factors = ram_factors[:5]
        time_factors_string = '\n'.join([f'# Time factor for {key}: {value["time_factor"]}'
                                         for key, value in group_stats.items()])

        n_jobs_failed = len([ji for ji in self.job_infos if ji.is_failed()])

        print(
            f'################################ END REPORT ###################################\n'
            f'# Start date: {format_date_s(self.start_time)}\n'
            f'# End date: {format_date_s(end_time)}\n'
            f'# Duration: {format_length_s(duration)}\n'
            f'# Number of failed jobs: {n_jobs_failed}\n'
            f'# Largest RAM factors: {ram_factors}\n'
            f'{time_factors_string}\n'
            f'###############################################################################',
            flush=True
        )

    def _compute_group_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        job_groups = [ji.job.get_group() for ji in self.job_infos]
        groups = set(job_groups)
        group_stats = {}
        for group in groups:
            job_infos: List[JobInfo] = [ji for ji, jg in zip(self.job_infos, job_groups) if jg == group]
            started_job_infos = [ji for ji in job_infos if not ji.is_remaining()]
            running_job_infos = [ji for ji in started_job_infos if ji.is_running()]
            finished_job_infos = [ji for ji in job_infos if ji.is_finished()]
            finished_job_infos_with_time = [ji for ji in finished_job_infos if ji.job_result.finished_normally]
            n_started = len(started_job_infos)
            n_running = len(running_job_infos)
            n_finished = len(finished_job_infos)
            n_finished_with_time = len(finished_job_infos_with_time)
            if n_started == 0 or (n_finished_with_time == 0 and n_running == 0):
                time_factor = 1.0
            elif n_finished_with_time == 0:
                current_time = time.time()
                elapsed_time = sum([current_time - ji.start_time for ji in running_job_infos])
                predicted_time_units = sum([ji.required_resources.time_s for ji in running_job_infos])
                time_factor = max(1.0, elapsed_time / (predicted_time_units + 1e-8))
            else:
                used_time = sum([ji.job_result.time_s for ji in finished_job_infos_with_time])
                predicted_time_units = sum([ji.required_resources.time_s
                                            for ji in finished_job_infos_with_time])
                time_factor = used_time / (predicted_time_units + 1e-8)
            group_stats[group] = {'time_factor': time_factor,
                                  'n_started': n_started,
                                  'n_running': n_running,
                                  'n_finished': n_finished,
                                  'n_finished_with_time': n_finished_with_time}
        return group_stats

    def _get_time_estimates(self, job_infos: List[JobInfo], group_stats: Dict[str, Dict[str, Union[int, float]]]) \
            -> np.ndarray:
        current_time = time.time()
        startup_time_s = 1.0  # guessed
        time_estimates = []
        for ji in job_infos:
            if ji.is_finished():
                time_estimates.append(0.0)  # job is already finished
                continue
            rr = ji.required_resources
            time_estimate = group_stats[ji.job.get_group()]['time_factor'] * rr.time_s
            if not ji.is_remaining():
                time_estimate = max(0.0, time_estimate - (current_time - ji.start_time))
            else:
                time_estimate += startup_time_s
            time_estimates.append(time_estimate)
        return np.asarray(time_estimates)

    def _print_progress(self):
        group_stats = self._compute_group_stats()
        resource_manager = self.job_manager.get_resource_manager()

        start_time = self.start_time
        current_time = time.time()
        elapsed_time = current_time - start_time

        total_resources = resource_manager.get_total_resources()
        fixed_resources = resource_manager.get_fixed_resources()
        average_fixed_resources = (fixed_resources * total_resources).get_resource_vector() \
                                  / (total_resources.get_resource_vector() + 1e-8)

        job_infos = self.job_infos

        n_jobs_finished = len([ji for ji in job_infos if ji.is_finished()])  # succeeded and failed ones
        n_jobs_remaining = len([ji for ji in job_infos if ji.is_remaining()])
        n_jobs_failed = len([ji for ji in job_infos if ji.is_failed()])
        n_jobs_running = len(job_infos) - n_jobs_finished - n_jobs_remaining

        time_estimates = self._get_time_estimates(job_infos, group_stats=group_stats)
        argmax_time_estimate = np.argmax(time_estimates)
        longest_job_desc = job_infos[argmax_time_estimate].job.get_desc()
        longest_time_estimate: float = time_estimates[argmax_time_estimate]
        system_resource_vec = total_resources.get_resource_vector()
        # estimate \sum_{jobs} job_resources * remaining_job_time
        # (could also do physical cores, but that should be covered by threads)
        total_job_time_resource_vec = sum([ji.required_resources.get_resource_vector(average_fixed_resources) * te
                                           for ji, te in zip(job_infos, time_estimates)])
        # todo: improve this estimate towards the end of a run?
        remaining_time_estimate = np.max(total_job_time_resource_vec / (system_resource_vec + 1e-8))

        elapsed_fraction = elapsed_time / (elapsed_time + remaining_time_estimate)

        end_date_str = format_date_s(current_time + remaining_time_estimate)

        # todo: also print predicted system usage in percent (relative to criticality of resources)?
        #  or print current relative resource usages and remaining task relative resource usages
        # todo: also log this somewhere automatically?
        print(
            f'############################ INTERMEDIATE REPORT ##############################\n'
            f'# {n_jobs_finished} jobs finished ({n_jobs_failed} failed), {n_jobs_running} jobs running, {n_jobs_remaining} jobs remaining\n'
            f'# Elapsed: {format_length_s(elapsed_time)} ({elapsed_time:.2f}s)\n'
            f'# Remaining: {format_length_s(remaining_time_estimate)} ({remaining_time_estimate:.2f}s)\n'
            f'# Percent completed: {100 * elapsed_fraction:.2f}%\n'
            f'# Estimated end time: {end_date_str}\n'
            f'# Current time: {format_date_s(current_time)}\n'
            f'# Longest remaining job: {longest_job_desc} with {format_length_s(longest_time_estimate)}\n'
            f'###############################################################################',
            flush=True
        )

    def _print_running_jobs(self):
        group_stats = self._compute_group_stats()

        current_time = time.time()

        job_infos = self.job_infos

        n_jobs_finished = len([ji for ji in job_infos if ji.is_finished()])
        n_jobs_remaining = len([ji for ji in job_infos if ji.is_remaining()])
        n_jobs_running = len(job_infos) - n_jobs_finished - n_jobs_remaining

        time_estimates = self._get_time_estimates(job_infos, group_stats=group_stats)

        job_strs = []

        sorted_time_idxs = np.argsort(time_estimates)

        # for ji, time_estimate in zip(job_infos, time_estimates):
        for i in sorted_time_idxs:
            ji = job_infos[i]
            time_estimate = time_estimates[i]
            if not ji.is_running():
                continue  # job is not currently running
            job: AbstractJob = ji.job
            job_desc = job.get_desc()
            job_str = (f'# Job {job_desc} has been running for {format_length_s(current_time-ji.start_time)}'
                       f', estimated remaining time: {format_length_s(time_estimate)}')
            job_strs.append(job_str)

        print(
            f'############################### RUNNING REPORT ################################\n'
            f'# Current time: {format_date_s(current_time)}, {n_jobs_running} jobs are running:\n'
            + '\n'.join(job_strs) + '\n' +
            f'###############################################################################',
            flush=True
        )


class SimpleJobScheduler(BaseJobScheduler):
    """
    Simple scheduler. Submits jobs with the largest estimated time. If a job doesn't fit,
    jobs with not too much smaller time can be submitted instead.
    In the beginning, the scheduler ensures that at least three jobs from each group are run
    (e.g. 3x XGB, 3x LGBM, 3x MLP).
    """
    def _submit_more_jobs(self) -> None:
        min_starts_per_group = 3

        job_infos = [ji for ji in self.job_infos if ji.is_remaining()]  # need running jobs as well for n_started_times?

        if len(job_infos) == 0:
            print(f'No job infos remaining')
            return

        group_stats = self._compute_group_stats()
        job_times = self._get_time_estimates(job_infos, group_stats)
        n_started_times = {key: value['n_running'] + value['n_finished_with_time']
                           for key, value in group_stats.items()}
        resource_manager = self.job_manager.get_resource_manager()
        # n_started_times = [group_stats[ji['job'].get_group()]['n_running']
        #                   + group_stats[ji['job'].get_group()]['n_finished_with_time'] for ji in job_infos]

        free_resources = copy.deepcopy(resource_manager.get_free_resources())
        fixed_resources = resource_manager.get_fixed_resources()

        if any(value < min_starts_per_group for value in n_started_times.values()):
            # need to start jobs first from groups where we don't have enough time measurements yet
            # do this by increasing their job_times estimate
            job_times_offset = 2 * np.max(job_times)
            for group, n_started in n_started_times.items():
                if n_started < min_starts_per_group:
                    job_idxs = np.asarray([i for i, ji in enumerate(job_infos) if ji.job.get_group() == group],
                                          dtype=np.int32)
                    sort_perm = np.argsort(job_times[job_idxs])
                    n_offset = min(len(sort_perm), min_starts_per_group - n_started)
                    # add job_times_offset to the n_offset jobs from this group with largest time estimate
                    job_times[job_idxs[sort_perm[-n_offset:]]] += job_times_offset

        # if a job with time estimate t cannot be started,
        # don't start jobs with time estimate less than min_time_factor * t
        # the maximum value of t is tracked in max_non_started_time
        min_time_factor = 0.1
        max_non_started_time = 0.0

        job_idxs_sorted = np.argsort(job_times)[::-1]  # sort descending

        for job_idx in job_idxs_sorted:
            if job_times[job_idx] < min_time_factor * max_non_started_time:
                # don't start too fast jobs if other much slower ones are waiting
                return

            job_info = job_infos[job_idx]

            # otherwise, try assigning the job
            for node_idx, r in enumerate(free_resources.resources):
                assigned_resources = r.try_assign(job_info.required_resources, fixed_resources)
                if assigned_resources is not None:
                    job_info.set_started(assigned_resources)
                    self.job_manager.submit_job(job_info)
                    free_resources.resources[node_idx] -= assigned_resources
                    break
            else:
                # could not assign the job
                max_non_started_time = max(max_non_started_time, job_times[job_idx])


class CustomJobScheduler(BaseJobScheduler):
    """
    More complicated scheduler with different heuristics for which jobs to submit first
    (based on which resources it thinks are scarce, estimated time, which methods have not been run yet, etc.).
    This scheduler can be slow for a large number of jobs (say 10,000 or more).
    """
    def _submit_more_jobs(self) -> None:
        # todo: how to handle OOM errors? Reduce total memory of nodes? Or increase memory of jobs?
        #  Or add constants to free_resources?
        #  maybe check if last error is at least one minute ago or so
        # current error handling: count job as finished, don't rerun

        min_starts_per_group = 3

        job_infos = [ji for ji in self.job_infos if not ji.is_finished()]

        group_stats = self._compute_group_stats()
        job_times = self._get_time_estimates(job_infos, group_stats)
        n_started_time = {key: value['n_running'] + value['n_finished_with_time'] for key, value in group_stats.items()}
        resource_manager = self.job_manager.get_resource_manager()
        # n_started_time = [group_stats[ji['job'].get_group()]['n_running']
        #                   + group_stats[ji['job'].get_group()]['n_finished_with_time'] for ji in job_infos]

        total_resources = resource_manager.get_total_resources()
        free_resources = copy.deepcopy(resource_manager.get_free_resources())
        fixed_resources = resource_manager.get_fixed_resources()

        print('total_resources.get_resource_vector():', total_resources.get_resource_vector())

        system_rv = total_resources.get_resource_vector()
        job_availability = np.asarray([1.0 if ji.is_remaining() else 0.0 for ji in job_infos])
        # n_nodes x 4
        total_node_rvs = np.asarray([r.get_resource_vector() for r in total_resources.resources])
        # shape: 4
        average_fixed_rv = (fixed_resources * total_resources).get_resource_vector() \
                           / (total_resources.get_resource_vector() + 1e-8)
        job_rvs = np.asarray([ji.required_resources.get_resource_vector(average_fixed_rv) for ji in job_infos])
        remaining_job_time_rv = sum([job_rv * job_time for job_rv, job_time in zip(job_rvs, job_times)])
        remaining_times_by_resource = remaining_job_time_rv / (system_rv + 1e-10)
        remaining_distr = remaining_times_by_resource / (np.max(remaining_times_by_resource) + 1e-8)
        criticality = np.exp(5.0 * remaining_distr)
        criticality /= np.sum(criticality)  # tempered softmax
        # max_remaining_time = np.max(remaining_times_by_resource)

        node_job_runability = np.asarray(
            [[r.try_assign(ji.required_resources, fixed_resources) is not None for ji in job_infos]
             for r in total_resources.resources])

        job_runability = np.any(node_job_runability, axis=0)
        # print('job_runability:', job_runability)
        for i in np.argwhere(~job_runability):
            # job i cannot run on any node, even if they are completely empty
            resource_vector = job_infos[int(i)].required_resources.get_resource_vector(average_fixed_rv)
            print(f'The following job does not fit on any node: {job_infos[int(i)].job.get_desc()}'
                  f', its required resource vector is {resource_vector}.',
                  file=sys.stderr, flush=True)

            job_availability[i] = 0.0

        while np.sum(job_availability) > 0.0:
            # if nodes get full before jobs run out, a return statement in the loop is used
            used_resources = total_resources - free_resources
            used_node_rvs = np.asarray([r.get_resource_vector() for r in used_resources.resources])

            # All scores will have shape n_nodes x n_jobs or broadcast to it

            # ----- Assignability -----

            assignments = [[r.try_assign(ji.required_resources, fixed_resources) for ji in job_infos]
                           for r in free_resources.resources]
            assignability_score = np.asarray([[1.0 if a is not None else 0.0 for a in l] for l in assignments])

            # ----- Uncertainty score -----
            uncertainty_score = np.asarray([
                max(0.0, min_starts_per_group - n_started_time[ji.job.get_group()])
                for ji in job_infos])
            uncertainty_score = uncertainty_score[None, :]

            # ----- Short Job Penalty -----

            # only use still available jobs for remaining partial sums
            job_times_rvs = job_times[:, None] * job_rvs * job_availability[:, None]
            perm = np.argsort(job_times)
            time_rv_partial_sums = np.zeros_like(job_times_rvs)
            time_rv_partial_sums[perm] = np.cumsum(job_times_rvs[perm], axis=0)
            time_partial_sums = [np.max(trps / (system_rv + 1e-8)) for trps in time_rv_partial_sums]
            max_time = np.max(job_times)  # todo: use times of all jobs, including currently running ones?
            partial_sum_threshold = 3 * max_time  # heuristic
            # penalty in [0, 1], largest for shortest jobs
            # shape: n_jobs
            short_job_penalty = (partial_sum_threshold - time_partial_sums) / partial_sum_threshold
            short_job_penalty[short_job_penalty < 0.0] = 0.0
            short_job_penalty = short_job_penalty[None, :]  # extend by node dimension

            # ----- Time score -----

            # could also use max_remaining_time in denominator instead
            time_score = job_times[None, :] / (max_time + 1e-8)  # in [0, 1]

            # ----- Resource score -----

            resource_score = np.sum(job_rvs[None, :, :] * criticality[None, None, :], axis=-1)
            resource_score /= (np.max(resource_score) + 1e-8)  # now in [0, 1]

            # ----- Utilization score -----

            # use as shape: n_nodes x n_jobs x 4
            new_resources = used_node_rvs[:, None, :] + job_rvs[None, :, :]
            new_utilization = new_resources / (total_node_rvs[:, None, :] + 1e-10)
            # what you could have got with uniform utilization
            new_opt_resources = np.max(new_utilization, axis=-1, keepdims=True) * total_node_rvs[:, None, :]
            # multiplying utilization with resources avoids the 0/0 GPU utilization problem
            new_missed_resources = new_opt_resources - new_resources

            old_resources = used_node_rvs[:, None, :]
            old_utilization = old_resources / (total_node_rvs[:, None, :] + 1e-10)
            # what you could have got with uniform utilization
            old_opt_resources = np.max(old_utilization, axis=-1, keepdims=True) * total_node_rvs[:, None, :]
            # multiplying utilization with resources avoids the 0/0 GPU utilization problem
            old_missed_resources = old_opt_resources - old_resources

            missing_improvement = np.sum((new_missed_resources - old_missed_resources) * criticality[None, None, :],
                                         axis=-1)
            running_improvement = np.sum(job_rvs[None, :, :] * criticality[None, None, :], axis=-1)

            # should be in (-\infty, 1]
            utilization_score = np.max(new_utilization, axis=-1) * missing_improvement / (running_improvement + 1e-8)
            utilization_score = utilization_score / (1.0 + np.abs(utilization_score))  # now in (-1, 1)

            # ----- Joint score -----
            # print(utilization_score.shape, time_score.shape, resource_score.shape, assignability_score.shape,
            #       short_job_penalty.shape, uncertainty_score.shape)
            joint_score = utilization_score + 0.3 * time_score + 0.2 * resource_score - 0.5 * assignability_score \
                          - 5 * short_job_penalty + 1000 * uncertainty_score
            low_value = np.min(joint_score) - 1
            joint_score[:, job_availability <= 0.5] = low_value
            joint_score[~node_job_runability] = low_value

            # ----- Find next node-job pair -----

            # strategy: find next best node-job pair.
            # If no assignment possible, terminate.
            # If job can be run now (assignable), add to list and recompute scores.
            # If job is not assignable to node,
            # block all jobs on node and block job on all nodes where it is not assignable.
            # Then loop back to next best node-job pair.

            while True:  # loop until an assignment is found or all nodes are blocked by unassignable jobs
                best_idxs = np.unravel_index(np.argmax(joint_score), joint_score.shape)
                if joint_score[best_idxs] == low_value:
                    print('No job remaining')
                    return

                node_idx = best_idxs[0]
                job_idx = best_idxs[1]
                assigned_resources = assignments[node_idx][job_idx]
                if assigned_resources is None:  # node is too full to run job now
                    print('Node too full')
                    # block node for now
                    joint_score[node_idx, :] = low_value
                    # make sure that job can only be stolen by other nodes if they are assignable
                    joint_score[assignability_score[:, job_idx] == 0.0, job_idx] = low_value
                else:
                    print('Assigning job')
                    job_availability[job_idx] = 0.0
                    job_info = job_infos[job_idx]
                    job_info.set_started(assigned_resources)
                    self.job_manager.submit_job(job_info)
                    free_resources.resources[node_idx] -= assigned_resources
                    n_started_time[job_info.job.get_group()] += 1
                    break  # leave inner loop, recompute scores


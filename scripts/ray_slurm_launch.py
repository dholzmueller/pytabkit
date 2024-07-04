# from https://docs.ray.io/en/latest/cluster/examples/slurm-launch.html#slurm-launch
# slurm-launch.py
# Usage:
# python slurm-launch.py --exp-name test \
#     --command "rllib train --run PPO --env CartPole-v0"

import argparse
# import subprocess
import sys
import time
import os

from pathlib import Path

from pytabkit.models import utils

template_file = Path(__file__).parent / "ray_slurm_template.sh"
JOB_NAME = "${JOB_NAME}"
NUM_NODES = "${NUM_NODES}"
NUM_GPUS_PER_NODE = "${NUM_GPUS_PER_NODE}"
PARTITION_OPTION = "${PARTITION_OPTION}"
ACCOUNT_OPTION = "${ACCOUNT_OPTION}"
COMMAND_PLACEHOLDER = "${COMMAND_PLACEHOLDER}"
GIVEN_NODE = "${GIVEN_NODE}"
LOAD_ENV = "${LOAD_ENV}"
TIME = "${TIME}"
MEM_CMD = "${MEM_CMD}"
MAIL_USER = "${MAIL_USER}"
LOG_FOLDER = "${LOG_FOLDER}"
CONDA_ENV_NAME = "${CONDA_ENV_NAME}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.out / exp_name.err).")
    parser.add_argument(
        "--conda-env-name",
        type=str,
        required=True,
        help="Conda environment name")
    parser.add_argument(
        "--num_nodes",
        "-n",
        type=int,
        default=1,
        help="Number of nodes to use.")
    parser.add_argument(
        "--mem",
        type=str,
        default=None,
        help="Memory (int + suffix 'mb').")
    parser.add_argument(
        "--time",
        "-t",
        type=str,
        help="Maximum time of job")
    # parser.add_argument(
    #     "--mem",
    #     type=str,
    #     help="Maximum memory of job")
    parser.add_argument(
        "--mail_user",
        "-m",
        type=str,
        default="",
        help="Mail address to which job updates will be sent")
    parser.add_argument(
        "--log_folder",
        "-l",
        type=str,
        default="",
        help="Folder in which to save log files"
    )
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        help="The specified nodes to use. Same format as the "
        "return of 'sinfo'. Default: ''.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use in each node. (Default: 0)")
    parser.add_argument(
        "--queue",
        "-q",
        type=str,
        default=None
    )
    parser.add_argument(
        "--partition",
        "-p",
        type=str,
        default="",
    )
    parser.add_argument(
        "--account",
        "-a",
        type=str,
        default="",
    )
    parser.add_argument(
        "--load-env",
        type=str,
        default="",
        help="The script to load your environment ('module load cuda/10.1')")
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="The command you wish to execute. For example: "
        " --command 'python test.py'. "
        "Note that the command must be a string.")
    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}".format(args.exp_name,
                              time.strftime("%y%m%d-%H%M", time.localtime()))

    partition_option = "#SBATCH --partition={}".format(
        args.partition) if args.partition else ""
    
    account_option = "#SBATCH --account={}".format(
        args.account) if args.account else ""

    # ===== Modified the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = text.replace(JOB_NAME, job_name)
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(NUM_GPUS_PER_NODE, str(args.num_gpus))
    text = text.replace(PARTITION_OPTION, partition_option)
    text = text.replace(ACCOUNT_OPTION, account_option)
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(LOAD_ENV, str(args.load_env))
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(TIME, args.time)
    mem_cmd = '' if args.mem is None else f'SBATCH --mem={args.mem}'
    text = text.replace(MEM_CMD, mem_cmd)
    text = text.replace(MAIL_USER, args.mail_user)
    text = text.replace(LOG_FOLDER, args.log_folder)
    text = text.replace(CONDA_ENV_NAME, args.conda_env_name)
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO "
        "PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!")

    # ===== Save the script =====
    script_file = "slurm_scripts/{}.sh".format(job_name)
    # os.makedirs("slurm_scripts")  # todo: ensure this
    utils.ensureDir(Path('slurm_scripts') / 'test.sh')  # ensure that slurm_scripts directory exists
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Starting to submit job!")
    cmd = f"sbatch {script_file}" if args.queue is None else f"sbatch -p {args.queue} {script_file}"
    # subprocess.Popen(cmd)
    os.system(cmd)
    print(
        "Job submitted! Script file is at: <{}>. Log file is at: <{}>".format(
            script_file, "{}.log".format(job_name)))
    sys.exit(0)

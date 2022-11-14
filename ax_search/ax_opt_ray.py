#!/vast/home/lix/.conda/envs/hippynn/bin/python3.10 -u
# fmt: off
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mail-type=all
#SBATCH -p ml4chem
#SBATCH -J parallel_hyperopt
#SBATCH --qos=long
#SBATCH -o run.log
# black always format pure comments as of now
# add some codes here to keep SLURM derivatives valid
import os
import sys

# SLURM copies the script to a tmp folder
# so to find the local package `training` we need add cwd to path
# per https://stackoverflow.com/a/39574373/7066315
sys.path.append(os.getcwd())
# fmt: on
"""
    B-Opt tuning for HIPNN using AX.

"""

import contextlib
import shutil

import numpy as np
import ray
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ray import tune
from ray.tune.logger import JsonLoggerCallback, LoggerCallback
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.ax import AxSearch
from training import main, read_args

# to make sure ray loads correct the local package
ray.init(runtime_env={"working_dir": "."})


def evaluate(parameter):
    """
    Evaluates a trial for MBO HIPNN.

    Args:
        parameter (dict): Python dictionary for trial values of HIPNN hyperparameters.

    Returns:
        dict : Loss metrics to be minimized.
    """

    try:
        # initialize and override parameters
        params = read_args(
            noprogress=True,
            # training_targets=["energy"],
            init_batch_size=512,
            raise_batch_patience=60,
            termination_patience=200,
            max_batch_size=4096,
            max_epochs=8001,
            n_states=5,
            bypass_cli_args=True,
            init_learning_rate=1e-2,
            **parameter,
        )
        # train model
        with contextlib.redirect_stdout(open(params.log_filename, "w")):
            out = main(params)
        print(f"returned loss is {out['Loss']}. cwd is {os.getcwd()}")

    except Exception as e:
        print("Error encountered")
        if hasattr(e, "output"):
            print(e.output)
        else:
            print(e)
        return {"Loss": np.inf}

    return {"Loss": out["Loss"]}


class AxLogger(LoggerCallback):
    def __init__(self, ax_client: AxClient, flnm: str):
        """
        A logger callback to save the progress to json file after every trial ends.
        Similar to running `ax_client.save_to_json_file` every iteration in sequential
        searches.

        Args:
            ax_client (AxClient): ax client to save
            flnm (str): name for the json file. Append a path if you want to save the \
                json file to somewhere other than cwd.
        """
        self.ax_client = ax_client
        self.flnm = flnm
        self.count = 0

    def log_trial_end(self, trial: "Trial", failed: bool = False):
        shutil.copy(self.flnm, f"{self.flnm}.bk")
        shutil.copy("hyperopt.csv", "hyperopt.csv.bk")
        self.ax_client.save_to_json_file(filepath=self.flnm)
        data_frame = self.ax_client.get_trials_data_frame().sort_values("Loss")
        data_frame.to_csv("hyperopt.csv", header=True)
        self.count += 1
        print("Ax saved", self.count)


# initialize the client and experiment.
if __name__ == "__main__":
    os.chdir("/projects/ml4chem/xinyang/ethene_with_nacr/")
    # TODO: better way to handle restarting of searches
    restart = False
    if restart:
        ax_client = AxClient.load_from_json_file(filepath="hyperopt_ray.json")
    else:
        ax_client = AxClient(
            verbose_logging=False,
            enforce_sequential_optimization=False,
        )
        ax_client.create_experiment(
            name="ethene_opt",
            parameters=[
                {
                    "name": "lower_cutoff",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0.5, 0.95],
                },
                {
                    "name": "upper_cutoff",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [2.5, 3.0],
                },
                {
                    "name": "cutoff_distance",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [3.5, 5.0],
                },
                {
                    "name": "n_sensitivities",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [10, 40],
                },
                {
                    "name": "n_features",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [2, 15],
                },
                {
                    "name": "n_interactions",
                    "type": "fixed",
                    "value_type": "int",
                    "value": 1,
                },
                {
                    "name": "n_atom_layers",
                    "type": "choice",
                    "values": [2, 3, 4],
                },
            ],
            objectives={
                "Loss": ObjectiveProperties(minimize=True),
            },
            overwrite_existing_experiment=True,
            is_test=False,
            parameter_constraints=[
                "lower_cutoff <= upper_cutoff",
                "upper_cutoff <= cutoff_distance",
            ],
        )

    # run the optimization Loop.
    algo = AxSearch(ax_client=ax_client)
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    ax_logger = AxLogger(ax_client, "hyperopt_ray.json")
    tune.run(
        evaluate,
        num_samples=40,
        search_alg=algo,
        verbose=0,
        # TODO: how to checkpoint?
        checkpoint_freq=2,
        # use one GPU per trial
        # ray will automatically set the environment variable `CUDA_VISIBLE_DEVICES`
        # so that only the next available GPU is exposed to torch
        resources_per_trial={"gpu": 1},
        # without specifying `local_dir`, the logs will be saved to ~/ray_results
        local_dir="./test_ray",
        callbacks=[ax_logger, JsonLoggerCallback()],
    )

    print("Script done.")

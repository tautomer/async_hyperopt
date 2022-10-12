#!/vast/home/lix/.conda/envs/hippynn/bin/python3.10 -u
# SBATCH --time=2-00:00:00
# SBATCH --nodes=1
# SBATCH --ntasks=40
# SBATCH --mail-type=all
# SBATCH -p ml4chem
# SBATCH -J hyperopt
# SBATCH --qos=long
# SBATCH -o run.log
"""
    B-Opt tuning for HIPNN using AX.

"""

import contextlib
import os

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
            training_targets=["energy"],
            init_batch_size=512,
            max_epochs=1,
            n_states=1,
            bypass_cli_args=True,
            **parameter
        )
        # train model
        with open(params.log_filename, "w") as log_file:
            with contextlib.redirect_stdout(log_file):
                out = main(params)
    except Exception as e:
        if hasattr(e, "output"):
            print(e.output)
        else:
            print(e)
        return {"Loss": np.inf}

    return {"Loss": out["Loss"]}


class AxLogger(LoggerCallback):
    def __init__(self, ax_client, flnm):
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
        self.ax_client.save_to_json_file(filepath=self.flnm)
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
                    "bounds": [0.5, 1.5],
                },
                {
                    "name": "upper_cutoff",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [3.0, 20.0],
                },
                {
                    "name": "cutoff_distance",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [5.0, 40.0],
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
                    "type": "range",
                    "value_type": "int",
                    "bounds": [1, 3],
                },
                {
                    "name": "n_atom_layers",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [1, 3],
                },
            ],
            objectives={
                "Loss": ObjectiveProperties(minimize=True),
            },
            overwrite_existing_experiment=True,
            is_test=True,
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
        num_samples=10,
        search_alg=algo,
        verbose=1,
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

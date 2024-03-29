#!/usr/bin/env python3
# fmt: off
##SBATCH --time=16:00:00
#SBATCH --time=4-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mail-type=all
#SBATCH -p gpu 
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
    Hyperparameter tuning for HIPNN using AX and Ray.

"""

import contextlib
import gc
import shutil

import numpy as np
import ray
import torch
from ax.core import Trial as AXTrial
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ray import air, tune
from ray.air import session
from ray.tune.experiment.trial import Trial
from ray.tune.logger import JsonLoggerCallback, LoggerCallback
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.ax import AxSearch
from training import main, read_args

# to make sure ray loads correct the local package
ray.init(runtime_env={"working_dir": "."})


def evaluate(parameter: dict, checkpoint_dir=None):
    """
    Evaluates a trial for MBO HIPNN.

    Args:
        parameter (dict): Python dictionary for trial values of HIPNN hyperparameters.
        checkpoint_dir (str, optional): To enable checkpoints for ray. Defaults to None.

    Returns:
        dict : Loss metrics to be minimized.
    """

    gc.collect()
    torch.cuda.empty_cache()
    # initialize and override parameters
    # targets = ["energy"]
    targets = ["energy", "dipole"]
    weights = []
    for i in targets:
        weights.append(parameter.get(i) or 1.0)
        del parameter[i]
    params = read_args(
        noprogress=True,
        db_to_gpu=True,
        training_targets=targets,
        target_weights=weights,
        init_batch_size=32,
        custom_kernel=True,
        raise_batch_patience=50,
        termination_patience=200,
        max_batch_size=2048,
        max_epochs=8001,
        n_states=2,
        bypass_cli_args=True,
        init_learning_rate=5e-4,
        **parameter,
    )
    # train model
    with contextlib.redirect_stdout(open(params.log_filename, "w")):
        out = main(params)

    e_Loss = out["metric"]["valid"]["ENERGY-Loss"]
    d_Loss = out["metric"]["valid"]["DIPOLE-Loss"]
    session.report(
        {
            "Loss": out["Loss"],
            # "Metric": out["Loss"],
            "Metric": e_Loss + d_Loss / 0.8,
            "ENERGY-Loss": e_Loss,
            "DIPOLE-Loss": d_Loss,
        }
    )


class AxLogger(LoggerCallback):
    def __init__(self, ax_client: AxClient, json_name: str, csv_name: str):
        """
        A logger callback to save the progress to json file after every trial ends.
        Similar to running `ax_client.save_to_json_file` every iteration in sequential
        searches.

        Args:
            ax_client (AxClient): ax client to save
            json_name (str): name for the json file. Append a path if you want to save the \
                json file to somewhere other than cwd.
            csv_name (str): name for the csv file. Append a path if you want to save the \
                csv file to somewhere other than cwd.
        """
        self.ax_client = ax_client
        self.json = json_name
        self.csv = csv_name

    def log_trial_end(
        self, trial: Trial, id: int, metric: float, runtime: int, failed: bool = False
    ):
        self.ax_client.save_to_json_file(filepath=self.json)
        shutil.copy(self.json, f"{trial.local_dir}/{self.json}")
        try:
            data_frame = self.ax_client.get_trials_data_frame().sort_values("Metric")
            data_frame.to_csv(self.csv, header=True)
        except KeyError:
            pass
        shutil.copy(self.csv, f"{trial.local_dir}/{self.csv}")
        if failed:
            status = "failed"
        else:
            status = "finished"
        print(
            f"AX trial {id} {status}. Final loss: {metric}. Time taken"
            f" {runtime} seconds. Location directory: {trial.logdir}."
        )

    def on_trial_error(self, iteration: int, trials: list[Trial], trial: Trial, **info):
        id = int(trial.experiment_tag.split("_")[0]) - 1
        ax_trial = self.ax_client.get_trial(id)
        ax_trial.mark_abandoned(reason="Error encountered")
        self.log_trial_end(
            trial, id + 1, "not available", self.calculate_runtime(ax_trial), True
        )

    def on_trial_complete(
        self, iteration: int, trials: list["Trial"], trial: Trial, **info
    ):
        # trial.trial_id is the random id generated by ray, not ax
        # the default experiment_tag starts with ax' trial index
        # but this workaround is totally fragile, as users can
        # customize the tag or folder name
        id = int(trial.experiment_tag.split("_")[0]) - 1
        ax_trial = self.ax_client.get_trial(id)
        failed = False
        try:
            loss = ax_trial.objective_mean
        except ValueError:
            failed = True
            loss = "not available"
        else:
            if np.isnan(loss) or np.isinf(loss):
                failed = True
                loss = "not available"
        if failed:
            ax_trial.mark_failed()
        self.log_trial_end(
            trial, id + 1, loss, self.calculate_runtime(ax_trial), failed
        )

    @classmethod
    def calculate_runtime(cls, trial: AXTrial):
        delta = trial.time_completed - trial.time_run_started
        return int(delta.total_seconds())


# initialize the client and experiment.
if __name__ == "__main__":
    os.chdir("/users/lix/scratch/prosq")
    # TODO: better way to handle restarting of searches
    restart = True
    if restart:
        ax_client = AxClient.load_from_json_file(filepath="hyperopt_ray.json")
        # update existing experiment
        # `immutable_search_space_and_opt_config` has to be False

        # ax_client.set_search_space(
        #     [
        #         {
        #             "name": "n_interactions",
        #             "type": "fixed",
        #             "value_type": "int",
        #             "value": 1,
        #         },
        #         {
        #             "name": "n_atom_layers",
        #             "type": "choice",
        #             "values": [2, 3, 4, 5, 6],
        #         },
        #     ]
        # )
    else:
        ax_client = AxClient(
            verbose_logging=False,
            enforce_sequential_optimization=False,
        )
        ax_client.create_experiment(
            name="prosq_opt",
            parameters=[
                {
                    "name": "lower_cutoff",
                    "type": "fixed",
                    "value_type": "float",
                    "value": 0.6334700267070265,
                },
                {
                    "name": "upper_cutoff",
                    "type": "fixed",
                    "value_type": "float",
                    "value": 7.170363852174164,
                },
                {
                    "name": "cutoff_distance",
                    "type": "fixed",
                    "value_type": "float",
                    "value": 7.426888330783524,
                },
                {
                    "name": "n_interactions",
                    "type": "fixed",
                    "value_type": "int",
                    "value": 3,
                },
                {
                    "name": "n_atom_layers",
                    "type": "fixed",
                    "value": 5,
                },
                {
                    "name": "n_sensitivities",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [20, 40],
                },
                {
                    "name": "n_features",
                    "type": "range",
                    "value_type": "int",
                    "bounds": [20, 60],
                },
                # {
                #     "name": "lower_cutoff",
                #     "type": "range",
                #     "value_type": "float",
                #     "bounds": [0.5, 0.95],
                # },
                # {
                #     "name": "upper_cutoff",
                #     "type": "range",
                #     "value_type": "float",
                #     "bounds": [1.5, 10.0],
                # },
                # {
                #     "name": "cutoff_distance",
                #     "type": "range",
                #     "value_type": "float",
                #     "bounds": [1.75, 15.0],
                # },
                # {
                #     "name": "n_sensitivities",
                #     "type": "range",
                #     "value_type": "int",
                #     "bounds": [20, 40],
                # },
                # {
                #     "name": "n_features",
                #     "type": "range",
                #     "value_type": "int",
                #     "bounds": [20, 60],
                # },
                # {
                #     "name": "n_interactions",
                #     "type": "choice",
                #     "value_type": "int",
                #     "values": [1, 2, 3],
                # },
                # {
                #     "name": "n_atom_layers",
                #     "type": "choice",
                #     "values": [2, 3, 4, 5],
                # },
                {
                    "name": "dipole",
                    "type": "fixed",
                    "value_type": "float",
                    "value": 1.0,
                },
                {
                    "name": "energy",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0.1, 2],
                    # "log_scale": True,
                },
            ],
            objectives={
                "Metric": ObjectiveProperties(minimize=True, threshold=2),
                # "Loss": ObjectiveProperties(minimize=True),
                # "DIPOLE-Loss": ObjectiveProperties(minimize=True, threshold=0.5),
                # "ENERGY-Loss": ObjectiveProperties(minimize=True, threshold=1.0),
            },
            overwrite_existing_experiment=True,
            is_test=False,
            # slightly more overhead
            # but make it possible to adjust the experiment setups
            immutable_search_space_and_opt_config=False,
            # parameter_constraints=[
            #     "lower_cutoff <= upper_cutoff",
            #     "upper_cutoff <= cutoff_distance",
            # ],
        )

    # run the optimization Loop.
    algo = AxSearch(ax_client=ax_client)
    algo = ConcurrencyLimiter(algo, max_concurrent=4)
    ax_logger = AxLogger(ax_client, "hyperopt_ray.json", "hyperopt.csv")
    tuner = tune.Tuner(
        tune.with_resources(evaluate, resources={"gpu": 1}),
        tune_config=tune.TuneConfig(search_alg=algo, num_samples=10),
        run_config=air.RunConfig(
            local_dir="./all_in_one",
            verbose=0,
            callbacks=[ax_logger, JsonLoggerCallback()],
            log_to_file=True,
        ),
    )
    tuner.fit()

    print("Script done.")

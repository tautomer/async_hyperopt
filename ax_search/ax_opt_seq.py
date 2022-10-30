#!/vast/home/lix/miniconda3/envs/jupyter-env/bin/python3.9 -u
# fmt: off
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mail-type=all
#SBATCH -p ml4chem
#SBATCH -J hyperopt
#SBATCH --qos=long
#SBATCH -o run.log
import sys

# fmt: on
"""
    B-Opt tuning for HIPNN using AX.

"""

import json
import os

# Use service client for more control; including checkpointing.
import ax
import numpy as np
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties


def load_metrics():
    """

    Load D1 MAE from metric tracker.

    Args:
        idx (int): Trial index for MBO.
    """
    # Load Test Error
    with open("test/folder", "r") as f:
        dirname = f.readline().rstrip()
    try:
        json_directory = "test/" + dirname + "/training_summary.json"
        if not os.path.exists(json_directory):
            print(os.getcwd())
            print(os.path.exists("test/" + dirname))
        with open(json_directory, "r") as out:
            summary = json.load(out)
    except FileNotFoundError:
        raise FileNotFoundError("Bruh Moment; Summary not found!")

    return summary["loss"]


def evaluate(parameter):
    """Evaluates a trial for MBO hipnn.

    Args:
        parameter (dict): Python dictionary for trial values of hipnn-hyper params.
        trial_index (integer): Index for the Bayesian Optimization step.

    Returns:
        dictionary : Loss metrics to be minimized.
    """
    # Save parameter as a string
    input_values = json.dumps(parameter)

    # Train HIPNN
    os.system(f"python3 -u training.py '{input_values}'")

    mae = load_metrics()

    # return {"mae": (mae, 0.0)}
    # The 0.0 is because we are not doing ensemble averages to get the standard deviation for loss/objective function.
    return {"mae": (mae, 0.0)}


# Initalize the Client and experiment.
os.chdir("/projects/ml4chem/xinyang/ethene/")
restart = False
if restart:
    ax_client = AxClient.load_from_json_file(filepath="hyperopt.json")
else:
    ax_client = AxClient(
        verbose_logging=False,
    )
    ax_client.create_experiment(
        name="ethene_opt",
        parameters=[
            {
                "name": "soft_min",
                "type": "range",
                "value_type": "float",
                "bounds": [0.5, 1.5],
            },
            {
                "name": "soft_max",
                "type": "range",
                "value_type": "float",
                "bounds": [3.0, 20.0],
            },
            {
                "name": "hard_max",
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
                "name": "n_interaction_layers",
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
        ],  # Add more constraints.
        # minimize=True,
        objectives={
            "mae": ObjectiveProperties(minimize=True),
        },
        overwrite_existing_experiment=True,
        is_test=True,
    )

### Run the Optimization Loop.
for k in range(30):
    parameter, trial_index = ax_client.get_next_trial()

    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameter))

    # Save experiment to file as JSON file
    ax_client.save_to_json_file(filepath="./hyperopt.json")
###

print("Script done.")
ax_client.get_next_trials()

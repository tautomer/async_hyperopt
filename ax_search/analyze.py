"""
    Analysis of the the ax_experiment.

    Creates a csv file summarizing the results.
"""

import csv
import json
import sys

import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient

restored_Ax_client = AxClient.load_from_json_file(filepath="hyperopt_ray.json")
try:
    best_parameters, values = restored_Ax_client.get_best_parameters()
except KeyError:
    print("All losses are NaNs.")


data_frame = restored_Ax_client.get_trials_data_frame().sort_values("Loss")
data_frame.to_csv("hyperopt.csv", header=True)

# best_obj_val = min(data_frame['Loss'])
# best_row = data_frame.loc[data_frame['Loss']==best_obj_val]
# print("Best Result is :: ")
# print(best_row)
# print("---------------------------")
for i in range(len(data_frame)):
    print(data_frame.iloc[i])
print(data_frame)

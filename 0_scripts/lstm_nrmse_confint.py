"""
This script calculates and saves point, upper, and lower bounds for ARIMA and ARIMAX models.
"""

import pandas as pd
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

sys.path.append('../')

from model_recurrent import Model

print("RECUERDA ACTIVAR EL VENV")

df = pd.read_csv("/home/marianne.pena/mainMarianne/dengue-model-rec/full_data.csv")

cantons = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 201, -202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 301, 302, 303, 304, 305, 306, 307, 308, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 701, 702, 703, 704, 705, 706]

cantons = [str(x) for x in cantons]

train_points_ind = []
train_upper_ind = []
train_lower_ind = []
train_points_global = []
train_upper_global = []
train_lower_global = []

test_points_ind = []
test_upper_ind = []
test_lower_ind = []
test_points_global = []
test_upper_global = []
test_lower_global = []

canton_list = []

for canton in cantons:
    model_instance = Model(df, canton)
    summary_ind, bootstrap_results_ind = model_instance.bootstrap_lstm_individual(
    epsilon=0.2,
    dif=False,
    block_size=4,
    n_bootstrap=500,
    seed=42
)
    
    train_points_ind.append(summary_ind["point_nrmse"]["train_val"])
    train_upper_ind.append(summary_ind["upper_95"]["train_val"])
    train_lower_ind.append(summary_ind["lower_95"]["train_val"])

    test_points_ind.append(summary_ind["point_nrmse"]["test"])
    test_upper_ind.append(summary_ind["upper_95"]["test"])
    test_lower_ind.append(summary_ind["lower_95"]["test"])

    summary_global, bootstrap_results_global = model_instance.bootstrap_lstm_full(
    block_size=4,
    n_bootstrap=500,
    epsilon=0.2,
    seed=42
)
    if canton == '-202':
        canton = '202'

    train_points_global.append(
        summary_global["train_val"].loc[canton, "point_nrmse"]
    )
    train_upper_global.append(
        summary_global["train_val"].loc[canton, "upper_95"]
    )
    train_lower_global.append(
        summary_global["train_val"].loc[canton, "lower_95"]
    )

    test_points_global.append(
        summary_global["test"].loc[canton, "point_nrmse"]
    )
    test_upper_global.append(
        summary_global["test"].loc[canton, "upper_95"]
    )
    test_lower_global.append(
        summary_global["test"].loc[canton, "lower_95"]
    )

    canton_list.append(canton)


nrmse_df = pd.DataFrame({
    'canton': canton_list,

    'train_point_ind': train_points_ind,
    'train_lower_ind': train_lower_ind,
    'train_upper_ind': train_upper_ind,

    'train_point_global': train_points_global,
    'train_lower_global': train_lower_global,
    'train_upper_global': train_upper_global,

    'test_point_ind': test_points_ind,
    'test_lower_ind': test_lower_ind,
    'test_upper_ind': test_upper_ind,

    'test_point_global': test_points_global,
    'test_lower_global': test_lower_global,
    'test_upper_global': test_upper_global
})

nrmse_df.to_csv("/home/marianne.pena/mainMarianne/dengue-model-rec/data/metrics/nrmse_lstm_confint.csv", index = False)

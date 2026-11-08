"""
This script fits and saves the global LSTM model and its training time.
"""

import pandas as pd
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append('../')

from model_recurrent import Model

print("RECUERDA ACTIVAR EL VENV")

df = pd.read_csv("../full_data.csv")

model0 = Model(df, "full")

results = model0.fit_lstm_full(n_iter = 50)

comp_time = results["elapsed_seconds"]

comp_times = pd.DataFrame({'canton': ['full'], 'comp_time': comp_time})

comp_times.to_csv("../data/comp_time/lstm_comp_time_full.csv", index = False)
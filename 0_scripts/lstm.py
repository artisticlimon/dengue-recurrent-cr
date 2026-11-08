"""
This script fits and saves the individual LSTM models and their training time.
"""

import pandas as pd
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append('../')

from model_recurrent import Model

print("RECUERDA ACTIVAR EL VENV")

df = pd.read_csv("/home/marianne.pena/mainMarianne/dengue-model-rec/full_data.csv")

cantons = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 201, -202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 301, 302, 303, 304, 305, 306, 307, 308, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 701, 702, 703, 704, 705, 706]

cantons = [str(x) for x in cantons]

comp_time_nodif = []
comp_time_dif = []
canton_list = []

for canton in cantons[0:11]:
    model_temp = Model(df, canton)
    comp_time_nodif_temp = model_temp.fit_lstm(dif = False)
    comp_time_nodif.append(comp_time_nodif_temp)
    comp_time_dif_temp = model_temp.fit_lstm(dif = True)
    comp_time_dif.append(comp_time_dif_temp)
    canton_list.append(canton)

comp_times = pd.DataFrame({'canton': canton_list, 'dif': comp_time_dif, "nodif": comp_time_nodif})
comp_times.to_csv("/home/marianne.pena/mainMarianne/dengue-model-rec/data/comp_time/lstm_comp_time_1.csv", index = False)
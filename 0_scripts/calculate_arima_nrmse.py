import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import numpy as np

sys.path.append('../')

from model_recurrent import Model

cantons = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 201, -202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 301, 302, 303, 304, 305, 306, 307, 308, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 602, 603, 604, 605, 606, 607, 608, 610, 701, 702, 703, 704, 705, 706]

cantons = [str(x) for x in cantons]

df = pd.read_csv("../full_data.csv")

nrmse_sarima = []
nrmse_sarimax = []
canton_list = []

for canton in cantons:
    model_temp = Model(df, canton)
    X_combined, y_combined, X_train, X_test, y_test, test, pds, epsilon = model_temp.partition()
    y_pred, lower_bound, upper_bound, nrmse_sarima_temp, order, sorder = model_temp.arima_forecast(y_combined, y_test, epsilon)
    features_test = X_test.drop(columns = ["rr_lag_1", "rr_lag_2", "rr_lag_3", "rr_lag_4"])
    y_pred, lower_bound, upper_bound, nrmse_sarimax_temp, order, sorder = model_temp.arima_forecast(y_combined, y_test, epsilon, exog = features_test)
    nrmse_sarima.append(nrmse_sarima_temp)
    nrmse_sarimax.append(nrmse_sarimax_temp)
    canton_list.append(canton)

nrmse_df = pd.DataFrame({'canton': canton_list, 'sarima': nrmse_sarima, 'sarimax': nrmse_sarimax})

nrmse_df.to_csv("../data/metrics/nrmse_arima_test.csv", index = False)
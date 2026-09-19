import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append('./')

from model_recurrent import Model

df = pd.read_csv("../full_data.csv")

cantons = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 201, -202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 301, 302, 303, 304, 305, 306, 307, 308, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 701, 702, 703, 704, 705, 706]

cantons = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 201, -202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 301, 302, 303, 304, 305, 306, 307, 308, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 701, 702, 703, 704, 705, 706]

cantons = [str(x) for x in cantons]

comp_time_arima = []
comp_time_arimax = []
canton_list = []

for canton in cantons:
    model_temp = Model(df, canton)
    X_combined, y_combined, X_train, X_test, y_test, test, pds, epsilon = model_temp.partition()
    comp_time_arima_temp = model_temp.fit_arima(y_combined)
    features = X_combined.drop(columns = ["rr_lag_1", "rr_lag_2", "rr_lag_3", "rr_lag_4"])
    features_test = X_test.drop(columns = ["rr_lag_1", "rr_lag_2", "rr_lag_3", "rr_lag_4"])
    features = features.astype(float)
    comp_time_arimax_temp = model_temp.choose_arimax(y_combined, y_test, epsilon, exog = features, exog_test = features_test)
    comp_time_arima.append(comp_time_arima_temp)
    comp_time_arimax.append(comp_time_arimax_temp)
    canton_list.append(canton)

comp_times = pd.DataFrame({'canton': canton_list, 'arima': comp_time_arima, 'arimax': comp_time_arimax})
comp_times.to_csv("/home/marianne.pena/mainMarianne/dengue-model-rec/data/comp_time/arima_comp_time.csv", index = False)
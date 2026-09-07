import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append('./')

from model_recurrent import Model

df = pd.read_csv("../full_data.csv")

cantons = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 201, -202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 301, 302, 303, 304, 305, 306, 307, 308, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 701, 702, 703, 704, 705, 706]

cantons = [str(x) for x in cantons]

for canton in cantons:
     model_temp = Model(df, canton)
     X_combined, y_combined, X_train, X_test, y_test, test, pds, epsilon = model_temp.partition()
     model_temp.fit_arima(y_combined)
     df_temp = df[df["week_canton"].str.contains(canton)]
     combined_temp = df_temp[df_temp['week_canton'] < '2025-1-101']
     features = combined_temp[['aedes_aegypti', 'criaderos_de_mosquito', 'dengue', 'fiebre',
             'plaquetas', 'buscar_trabajo', 'desempleo', 'empleo', 'comprar_carro',
             'préstamo', 'tiquetes_avion', 'vacaciones', 'corte_agua',
             'corte_de_agua', 'suspension_de_agua', 'tmin_mean', 'tmax_mean',
             'temp_prom', 'precip_total', 'precip_max', 'precip_min', 'precip_mean',
             'precip_median', 'nino34ssta', 'ndvi']]
     model_temp.fit_arimax(y_combined, exog = features)  
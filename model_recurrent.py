"""
This script contains the Model class used to train and evaluate the different models for relative risk. It includes methods for partitioning the data, training the models, calculating variable importance, calculating prediction intervals, and calculating confidence intervals for the NRMSE metric. The class is designed to be flexible and can be used for different cantons and model types (Random Forest, XGBoost, Hybrid Random Forest, Hybrid XGBoost).
"""

import pandas as pd
from sklearn.metrics import root_mean_squared_error, make_scorer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.model_selection import GridSearchCV, PredefinedSplit, TimeSeriesSplit
import joblib
from sklearn.inspection import permutation_importance
from arch.bootstrap import IIDBootstrap
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from mapie.conformity_scores import AbsoluteConformityScore
from mapie.subsample import BlockBootstrap
import time
import pmdarima as pm 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from neuralforecast.models import TimeXer, NHITS, TSMixerx
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA
import random
from scikeras.wrappers import KerasRegressor
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import pytorch_lightning as pl

class Model:

    """
    Attributes:
    ----
    df: dataframe
        complete dataset with, including lags and target variable, with each row being a week-canton pair
    canton: str
        the canton for which we want to train and evaluate the model. 
    """

    def __init__(self, df, canton):
    
        """
        Method that initializes the class, filters for the desired canton, and drops the columns that are not needed for the models because they correspond to that same week. 

        Parameters
        ----
        df: dataframe
            complete dataset with, including lags and target variable, with each row being a week-canton pair
        canton: str
            the canton for which we want to train and evaluate the model.
        Returns
        ----
            Nothing, but initializes the class with the filtered and cleaned dataframe for the desired canton.
        """
        self.df = df.copy()
        self.canton = canton
        if canton != "full":
            self.df = df[df["week_canton"].str.contains(canton)] 
        else:
            self.df = df    

    def drop_lags(self, i, f, df_temp):

        """
        Method that drops the lag columns that are not needed for the models, depending on which time frame we want to use to make predictions (usual options include immediate, medium term, all lags).

        Parameters
        ----
        i: int
            initial month of lags to drop (e.g., if we want to keep lags 1-4, i would be 5)
        f: str
            last month of lags to drop (e.g., if we want to keep lags 1-4, f would be 9)
        df_temp: dataframe
            dataframe from which to drop the lag columns
        Returns
        ----
            df_temp: dataframe with the specified lag columns dropped
        """

        for i in range(i, f):
            df_temp.drop(columns = [f"tmax_mean_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"tmin_mean_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"temp_prom_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"precip_total_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"precip_max_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"precip_min_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"precip_median_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"precip_mean_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"ndvi_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"nino34ssta_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"aedes_aegypti_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"criaderos_de_mosquito_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"fiebre_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"plaquetas_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"buscar_trabajo_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"desempleo_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"empleo_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"comprar_carro_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"préstamo_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"tiquetes_avion_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"vacaciones_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"corte_agua_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"corte_de_agua_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"suspension_de_agua_lag_{i}"], inplace = True)
            df_temp.drop(columns = [f"dengue_lag_{i}"], inplace = True)
        return df_temp

    def partition(self, epsilon = 0.2):

        """
        Method that created that partitions the dataset into train, validation, and test sets for relative risk as a target value. It drops other target values and relative risk lags. It adds a small value (epsilon) to the relative risk before log-transforming it, to avoid issues with zero values. Finally, it also creates a predefined split object to be used for both hyperparameter tuning and RFECV, specifying which data points belong to train and which to validation.

        Returns
        ----
            X_combined: dataframe
                dataframe with the features for the train and validation set
            y_combined: pandas Series
                pandas Series with the target variable (relative risk) for the train and validation set, log-transformed and with epsilon added
            X_train: dataframe
                dataframe with the features for the train set. It is returned separately from X_combined to be used to create the new predefined splits for the regression part of the hybrid model.
            X_test: dataframe
                dataframe with the features for the test set
            y_test: pandas Series
                pandas Series with the target variable (relative risk) for the test set, log-transformed and with epsilon added
            test: dataframe
                dataframe with the test set, including the week_canton column for plotting purposes
            pds: predefined split
                the predefined split object that specifies which data points belong to train and which to validation, to be used for both hyperparameter tuning and RFECV.
            epsilon: int
                the small value that was added to the relative risk before log-transforming it, to avoid issues with zero values. The default value was found through trial and error, looking at histograms for each value.
        """

        df_temp = self.df.copy()

        df_temp.drop(columns=['clasi_rr'], inplace=True)
        df_temp.drop(columns = ["casos"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0"], inplace = True)

        df_temp.drop(columns = ["nino34ssta"], inplace = True)
        df_temp.drop(columns = ["tmax_mean"], inplace = True)
        df_temp.drop(columns = ["tmin_mean"], inplace = True)
        df_temp.drop(columns = ["temp_prom"], inplace = True)
        df_temp.drop(columns = ["precip_total"], inplace = True)
        df_temp.drop(columns = ["precip_max"], inplace = True)
        df_temp.drop(columns = ["precip_min"], inplace = True)
        df_temp.drop(columns = ["precip_median"], inplace = True)
        df_temp.drop(columns = ["precip_mean"], inplace = True)
        df_temp.drop(columns = ["ndvi"], inplace = True)
        df_temp.drop(columns = ["aedes_aegypti"], inplace = True)
        df_temp.drop(columns = ["criaderos_de_mosquito"], inplace = True)
        df_temp.drop(columns = ["fiebre"], inplace = True)
        df_temp.drop(columns = ["plaquetas"], inplace = True)
        df_temp.drop(columns = ["buscar_trabajo"], inplace = True)
        df_temp.drop(columns = ["desempleo"], inplace = True)
        df_temp.drop(columns = ["empleo"], inplace = True)
        df_temp.drop(columns = ["comprar_carro"], inplace = True)
        df_temp.drop(columns = ["préstamo"], inplace = True)
        df_temp.drop(columns = ["tiquetes_avion"], inplace = True)
        df_temp.drop(columns = ["vacaciones"], inplace = True)
        df_temp.drop(columns = ["corte_agua"], inplace = True)
        df_temp.drop(columns = ["corte_de_agua"], inplace = True)
        df_temp.drop(columns = ["suspension_de_agua"], inplace = True)
        df_temp.drop(columns = ["dengue"], inplace = True)

        df_temp = self.drop_lags(i = 5, f = 9, df_temp = df_temp)

        for i in range(5, 9):
            df_temp.drop(columns = [f"rr_lag_{i}"], inplace = True)
        
        df_temp.drop(columns = ["casos_lag_1"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_1"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_1"], inplace = True)
        df_temp.drop(columns = ["casos_lag_2"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_2"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_2"], inplace = True)
        df_temp.drop(columns = ["casos_lag_3"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_3"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_3"], inplace = True)
        df_temp.drop(columns = ["casos_lag_4"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_4"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_4"], inplace = True)
        df_temp.drop(columns = ["casos_lag_5"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_5"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_5"], inplace = True)
        df_temp.drop(columns = ["casos_lag_6"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_6"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_6"], inplace = True)
        df_temp.drop(columns = ["casos_lag_7"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_7"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_7"], inplace = True)
        df_temp.drop(columns = ["casos_lag_8"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_8"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_8"], inplace = True)

        df_temp = pd.get_dummies(df_temp, columns = ["urb"])

        # 2020-2023 as train, 2024 as validation, 2025 as test  
        train = df_temp[df_temp['week_canton'] < '2024-1-101']
        val = df_temp[(df_temp['week_canton'] >= '2024-1-101') & (df_temp['week_canton'] < '2025-1-101')]
        test = df_temp[df_temp['week_canton'] >= '2025-1-101'] 

        X_train = train.drop(columns = ["rr", "week_canton"]).reset_index(drop=True)
        y_train = np.log(train["rr"] + epsilon).reset_index(drop=True)

        X_val= val.drop(columns = ["rr", "week_canton"]).reset_index(drop=True)
        y_val= np.log(val["rr"]  + epsilon).reset_index(drop=True)

        X_test = test.drop(columns = ["rr", "week_canton"]).reset_index(drop=True)
        y_test = np.log(test["rr"] + epsilon).reset_index(drop=True)

        X_combined = pd.concat([X_train, X_val], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)

        split_indices = np.zeros(len(X_combined))
        split_indices[:len(X_train)] = -1
        split_indices[len(X_train):] = 0

        pds = PredefinedSplit(test_fold=split_indices) # Create predefined split to be used for both hyperparameter tuning and RFECV, specifying which data points belong to train and which to validation

        return X_combined, y_combined, X_train, X_test, y_test, test, pds, epsilon

    def var_importance(self, X_test, y_test, model_type, repeats = 100):

        """
        Method that calculates and plots the permutation importance of the features for the specified model type, using the test set. It reads the trained model from the saved models folder, calculates the permutation importance using the sklearn function, and then creates a horizontal bar plot with the importance scores for the top 10 features. If it is a hurdle model, it calculates the permutation importance for both the classification and regression parts of the model, and plots them separately.

        Parameters
        ----
        X_test: dataframe
            dataframe with the features for the test set
        y_test: pandas Series
            pandas Series with the target variable (relative risk) for the test set, log-transformed and with epsilon added
        model_type: str
            the type of model for which to calculate the variable importance (options: "rf", "xgb", "hrf", "hxgb")
        repeats: int
            the number of times to permute a feature for calculating the permutation importance, to be used as the n_repeats parameter in the sklearn function. 

        Returns
        ----
            imp_df: dataframe
                dataframe with the features, their importance scores, and their standard deviation, sorted by importance score in descending order.
            It also shows the horizontal bar plot with the importance scores for the top 10 features.
        """
        start = time.perf_counter()

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
        try:
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_reg_{self.canton}.joblib')
        except FileNotFoundError: 
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_{self.canton}.joblib')
        
        regressor = grid_reg.best_estimator_

        if model_type == "hrf" or model_type == "hxgb":
            try:
                grid_classi = joblib.load(f'../models/saved_models/{model_type}_classi_{self.canton}.joblib')
            except FileNotFoundError:
                grid_classi = None
        
            if self.canton == "full":
                thresholds = pd.read_csv(f"../data/model_results/thresholds/full_threshold_{model_type}.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
                best_threshold = thresholds[thresholds["canton"] == self.canton]["best_th"].values[0]
                classifier = None if grid_classi is None else grid_classi.best_estimator_
                regressor = grid_reg.best_estimator_
            else: 
                thresholds = pd.read_csv(f"../data/model_results/thresholds/thresholds_{model_type}.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
                best_threshold = thresholds[thresholds["canton"] == self.canton]["best_th"].values[0]
                classifier = None if grid_classi is None else grid_classi.best_estimator_
                regressor = grid_reg.best_estimator_ 

            print("Classifier")

            if grid_classi != None:
                scorer = make_scorer(
                        self.custom_threshold_score,
                        response_method="predict_proba",
                        threshold=best_threshold
                )

                r_classi = permutation_importance(classifier, X_test, y_test, n_repeats=repeats, random_state=42, scoring=scorer, n_jobs = -1)
                
                imp_df_classi = pd.DataFrame({"Feature": X_test.columns, f"imp_{model_type}": r_classi.importances_mean, f"std_{model_type}": r_classi.importances_std}).sort_values(by=f"imp_{model_type}", ascending=False).reset_index(drop=True)

                fig, ax = plt.subplots(figsize=(12, 8))
                imp_df_classi_sorted = imp_df_classi.head(10).sort_values(by=f"imp_{model_type}", ascending=True)
                ax.barh(imp_df_classi_sorted["Feature"], imp_df_classi_sorted[f"imp_{model_type}"])
                ax.tick_params(axis='y', labelsize=14)
                for label in ax.get_yticklabels():
                    label.set_fontweight('bold')
                ax.tick_params(axis='x', labelsize=14)
                for label in ax.get_xticklabels():
                    label.set_fontweight('bold')
                ax.bar_label(ax.containers[0], fmt='%.2e', fontsize = 10, fontweight = "bold")
                ax.set_xlabel("Importance", fontsize = 20, fontweight = "bold")
                # fig.suptitle(f"{model_type} classification permutation importance for RR in {self.canton}")
                plt.show()

            print("Regressor")

        r = permutation_importance(regressor, X_test, y_test, n_repeats=repeats, random_state=42, scoring='neg_mean_squared_error', n_jobs = -1)

        imp_df = pd.DataFrame({"Feature": X_test.columns, f"imp_{model_type}": r.importances_mean, f"std_{model_type}": r.importances_std}).sort_values(by=f"imp_{model_type}", ascending=False).reset_index(drop=True)

        print(imp_df)

        fig, ax = plt.subplots(figsize=(12, 8))
        imp_df_sorted = imp_df.head(10).sort_values(by=f"imp_{model_type}", ascending=True)
        ax.barh(imp_df_sorted["Feature"], imp_df_sorted[f"imp_{model_type}"])
        ax.tick_params(axis='y', labelsize=14)
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.tick_params(axis='x', labelsize=14)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        ax.bar_label(ax.containers[0], fmt='%.2e', fontsize = 10, fontweight = "bold")
        ax.set_xlabel("Importance", fontsize = 20, fontweight = "bold")
        # fig.suptitle(f"{model_type} regression permutation importance for {var} in {self.canton}")
        plt.show()

        end = time.perf_counter()

        print(f"Elapsed time: {end - start:.2f} seconds")

        return imp_df
    
    def fit_arima(self, y_combined, lag = None, exog = None):

        start = time.perf_counter()

        date_idx = pd.date_range(start='2020-02-23', periods=len(y_combined), freq='W-SUN')
        y_combined.index = date_idx

        if exog is not None and lag is None:
            raise ValueError("lag must be specified when exog is provided")

        if lag is not None:
            exog = exog.loc[:, exog.columns.str.contains(f"lag_{lag}")].copy()

        if exog is not None:
            exog = exog.copy()
            scaler = MinMaxScaler().fit(exog)
            exog = scaler.transform(exog)
            pca = PCA(n_components=0.95)
            exog = pca.fit_transform(exog)
            exog = pd.DataFrame(exog)
            exog.index = date_idx

        auto_model = pm.auto_arima(y_combined, X = exog, seasonal=True, m=52, stepwise=True, suppress_warnings=True, error_action='warn', random_state=42)

        order = auto_model.order
        seasonal_order = auto_model.seasonal_order

        model = SARIMAX(y_combined, exog=exog, order=order,seasonal_order=seasonal_order,trend=None).fit(disp=False, maxiter = 200)

        end = time.perf_counter()
        
        comp_time = end - start
        
        if exog is None:
            joblib.dump(model, f"../saved_models/arima/arima_{self.canton}.joblib")
            print(f"ARIMA ready for {self.canton}. Elapsed time: {comp_time:.2f} seconds")
        else: 
            model_artifacts = {'model': model, 'scaler': scaler, 'pca' : pca, 'lag': lag}
            print(f"ARIMAX lag {lag} ready for {self.canton}. Elapsed time: {comp_time:.2f} seconds")

            return comp_time, model_artifacts

        return comp_time

    def choose_arimax(self, y_combined, y_test, epsilon, exog, exog_test):

        date_idx = pd.date_range(start='2020-02-23', periods=len(y_combined), freq='W-SUN')
        y_combined.index = date_idx
        
        models = [] 
        lags = [1, 2, 3, 4]
        scalers = []
        pca = []
        comp_times = []
        nrmses = []

        for lag in lags:
            exog_train_temp = exog.loc[:, exog.columns.str.contains(f"lag_{lag}")].copy()
            exog_test_temp = exog_test.loc[:, exog_test.columns.str.contains(f"lag_{lag}")].copy()
            comp_time_temp, artifact_temp = self.fit_arima(y_combined, lag = lag, exog = exog)
            y_pred, lower_bound, upper_bound, nrmse_temp, order, sorder = self.arima_forecast(y_combined, y_test, epsilon, proposed_artifact = artifact_temp, exog = exog_test_temp) 
            nrmses.append(nrmse_temp)
            models.append(artifact_temp["model"])
            scalers.append(artifact_temp["scaler"])
            pca.append(artifact_temp["pca"])
            comp_times.append(comp_time_temp)
            print(f"NRMSE for {self.canton} lag {lag}: {nrmse_temp}")

        best = np.argmin(nrmses)

        best_model_artifacts = {'model': models[best], 'scaler': scalers[best], 'pca': pca[best], 'lag': lags[best]}
        
        joblib.dump(best_model_artifacts, f"../saved_models/arimax/arimax_{self.canton}.joblib")
        print(f"Best ARIMAX lag {lags[best]} ready for {self.canton}. Elapsed time: {comp_times[best]:.2f}")

        return comp_times[best]

    def arima_forecast(self, y_combined, y_test, epsilon, dataset = "train", proposed_artifact = None, exog = None): 
        if exog is None: 
            model = joblib.load(f'../saved_models/arima/arima_{self.canton}.joblib')
        else: 
            if proposed_artifact is not None:
                artifact = proposed_artifact
            else:
                artifact = joblib.load(
                    f'../saved_models/arimax/arimax_{self.canton}.joblib'
                )
            model = artifact["model"]
            scaler = artifact["scaler"]
            lag = artifact["lag"]
            pca = artifact["pca"]
            exog = exog.loc[:, exog.columns.str.contains(f"lag_{lag}")].copy()

        order = model.model.order
        sorder = model.model.seasonal_order

        train_dates = pd.date_range(
            start='2020-02-23',
            periods=len(y_combined),
            freq='W-SUN'
        )

        test_dates = pd.date_range(
            start=train_dates[-1] + pd.Timedelta(weeks=1),
            periods=len(y_test),
            freq='W-SUN'
        )
        y_test.index = test_dates

        if exog is not None:
            exog = exog.copy()
            exog = scaler.transform(exog)
            exog = pca.transform(exog)
            exog = pd.DataFrame(exog)
            if dataset == "train": 
                exog.index = train_dates
            else:
                exog.index = test_dates

        pred = np.zeros(len(y_test))

        lower_bound = np.zeros(len(y_test))
        upper_bound = np.zeros(len(y_test))

        if dataset == "train":
            steps = len(y_combined)
            first_date = train_dates[0]
        else:
            steps = len(y_test)
            first_date = train_dates[-1] 

        for i in range(steps):
                
            next_date = model.model.data.dates[-1] + pd.Timedelta(weeks=1)
        
            if exog is None:
                X_next = None
            else:
                X_next = exog.iloc[[i]].copy()
                X_next.index = pd.DatetimeIndex([next_date], freq="W-SUN")

            if dataset == "train":
                point = y_combined.iloc[i]
            else:   
                point = y_test.iloc[i]

            y_new = pd.Series(
                [point],
                index=pd.DatetimeIndex([next_date], freq="W-SUN")
            )

            forecast = model.get_forecast(
                steps=1,
                exog=X_next
            )

            pred[i] = forecast.predicted_mean.iloc[0]

            conf_int = forecast.conf_int(alpha=0.05)

            lower_bound[i] = conf_int.iloc[0, 0]
            upper_bound[i] = conf_int.iloc[0, 1]
            
            model = model.extend(
                endog=y_new,
                exog=X_next
            )

        y_pred = np.exp(pred) - epsilon
        y_true = np.exp(y_test) - epsilon

        lower_bound = np.exp(lower_bound) - epsilon
        upper_bound = np.exp(upper_bound) - epsilon

        rmse = root_mean_squared_error(y_true, y_pred)
        nrmse = rmse / (np.mean(y_true) + epsilon)

        return y_pred, lower_bound, upper_bound, nrmse, order, sorder

    def arima_forecast_plot(self, y_combined, y_test, epsilon, exog_train = None, exog_test = None):
    
        if exog_test is None: 
            y_pred_train, lower_bound_train, upper_bound_train, nrmse_train, order, sorder = self.arima_forecast(y_combined, y_combined, epsilon, dataset = "train")
            y_pred_test, lower_bound_test, upper_bound_test, nrmse_test, order, sorder = self.arima_forecast(y_combined, y_test, epsilon, dataset = "test")
            print(f"Auto SARIMA{order}{sorder}, NRMSE train: {nrmse_train:.3f}, NRMSE test: {nrmse_test:.3f}")
        else: 
            y_pred_train, lower_bound_train, upper_bound_train, nrmse_train, order, sorder = self.arima_forecast(y_combined, y_combined, epsilon, dataset = "train", exog = exog_train)
            y_pred_test, lower_bound_test, upper_bound_test, nrmse_test, order, sorder = self.arima_forecast(y_combined, y_test, epsilon, dataset = "test",  exog = exog_test)
            print(f"Auto SARIMAX{order}{sorder}  NRMSE train: {nrmse_train:.3f}, NRMSE test: {nrmse_test:.3f}")

        df_copy = self.df.copy()
        df_copy['week_canton'] = df_copy['week_canton'].str.extract(r'(\d{4}-\d+)').iloc[:, 0].tolist()

        y_true = np.exp(y_test) - epsilon
        y_combined_true = np.exp(y_combined) - epsilon


        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(df_copy['week_canton'].iloc[:len(y_combined)], y_combined_true, label='Real', marker='o', linestyle = "dashed", color = "#000000", linewidth=4)
        ax.plot(df_copy['week_canton'].iloc[len(y_combined):], y_true, marker='o', linestyle = "dashed", color = "#000000", linewidth=4)
        ax.plot(df_copy['week_canton'].iloc[:len(y_combined)], y_pred_train, label='Predicted (train)', marker='o',linestyle = "solid", color = "#99009E", linewidth=4)
        ax.fill_between(x = df_copy['week_canton'].iloc[:len(y_combined)], label = "CI (95%)", y1 = lower_bound_train, y2 = upper_bound_train, alpha = 0.2, color = "#99009E")
        ax.plot(df_copy['week_canton'].iloc[len(y_combined):], y_pred_test, label='Predicted (test)', marker='o',linestyle = "solid", color = "#009E73", linewidth=4)
        ax.fill_between(x = df_copy['week_canton'].iloc[len(y_combined):], label = "CI (95%)", y1 = lower_bound_test, y2 = upper_bound_test, alpha = 0.2, color = "#009E73")
        ax.legend(loc = "upper right", fontsize = 18)
        ax.set_xlabel('Week', fontweight = "bold", fontsize = 20)
        self.ticks_years_top(ax, 25)
        ax.set_ylabel(f'Relative risk', fontweight = "bold", fontsize = 20)
        if exog_test is None:
            ax.set_title(f"SARIMA{order}{sorder} model for {self.canton} NRMSE train: {nrmse_train:.3f}, NRMSE test: {nrmse_test:.3f}")
        else:
            ax.set_title(f"SARIMAX{order}{sorder} for {self.canton}, NRMSE train: {nrmse_train:.3f}, NRMSE test: {nrmse_test:.3f}")

    def set_seed(self, seed=42):
        random.seed(seed)
        
        np.random.seed(seed)
        
        tf.random.set_seed(seed)
        
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

    def set_seed_full(self, seed=42):

        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        pl.seed_everything(seed, workers=True)

    def partition_lstm(self, sequence_length, epsilon=0.2, dif = False):

        df_temp = self.df.copy()

        df_temp = self.drop_lags(i = 1, f = 9, df_temp = df_temp)
        
        for i in range(1, 9):
            df_temp.drop(columns = [f"rr_lag_{i}"], inplace = True)
        
        df_temp.drop(columns=['clasi_rr'], inplace=True)
        df_temp.drop(columns = ["casos"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0"], inplace = True)

        df_temp.drop(columns = ["casos_lag_1"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_1"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_1"], inplace = True)
        df_temp.drop(columns = ["casos_lag_2"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_2"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_2"], inplace = True)
        df_temp.drop(columns = ["casos_lag_3"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_3"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_3"], inplace = True)
        df_temp.drop(columns = ["casos_lag_4"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_4"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_4"], inplace = True)
        df_temp.drop(columns = ["casos_lag_5"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_5"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_5"], inplace = True)
        df_temp.drop(columns = ["casos_lag_6"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_6"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_6"], inplace = True)
        df_temp.drop(columns = ["casos_lag_7"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_7"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_7"], inplace = True)
        df_temp.drop(columns = ["casos_lag_8"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_8"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_8"], inplace = True)

        df_temp = pd.get_dummies(df_temp, columns = ["urb"])

        df_temp["rr_log"] = np.log(df_temp["rr"] + epsilon)
        df_temp["rr_log_raw"] = df_temp["rr_log"]

        if dif:
            df_temp["rr_log"] = df_temp["rr_log"].diff().fillna(0)
            df_temp = df_temp.iloc[1:].reset_index(drop=True)

        parts = df_temp["week_canton"].str.split("-", expand=True)

        df_temp["year"] = parts[0].astype(int)
        df_temp["week"] = parts[1].astype(int)

        df_temp = df_temp.sort_values(
            ["year", "week"]
        ).reset_index(drop=True)

        train = df_temp[df_temp["year"] < 2024].copy()

        val = df_temp[
            (df_temp["year"] == 2024)
        ].copy()

        test = df_temp[
            (df_temp["year"] >= 2025)
        ].copy()

        feature_cols = [
            col for col in df_temp.columns
            if col not in ["rr", "rr_log_raw", "week_canton", "year", "week"]
        ]

        X_train = train[feature_cols].reset_index(drop=True)
        X_val = val[feature_cols].reset_index(drop=True)
        X_test = test[feature_cols].reset_index(drop=True)

        y_train = train["rr_log"].values

        y_val = val["rr_log"].values

        y_test = test["rr_log"].values

        scaler_X = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_val_scaled = scaler_X.transform(X_val)
        X_test_scaled = scaler_X.transform(X_test)

        scaler_y = MinMaxScaler()

        y_train_scaled = scaler_y.fit_transform(
            y_train.reshape(-1, 1)
        ).ravel()

        y_val_scaled = scaler_y.transform(
            y_val.reshape(-1, 1)
        ).ravel()

        y_test_scaled = scaler_y.transform(
            y_test.reshape(-1, 1)
        ).ravel()

        X_train_seq, y_train_seq = self.create_sequences_lstm(
            X_train_scaled,
            y_train_scaled,
            sequence_length
        )

        train_dates = train["week_canton"].iloc[
        sequence_length:
    ].reset_index(drop=True)

        X_val_context = np.vstack([
            X_train_scaled[-sequence_length:],
            X_val_scaled
        ])

        y_val_context = np.concatenate([
            y_train_scaled[-sequence_length:],
            y_val_scaled
        ])

        X_val_seq, y_val_seq = self.create_sequences_lstm(
            X_val_context,
            y_val_context,
            sequence_length
        )

        val_dates = val["week_canton"].reset_index(drop=True)

        X_test_context = np.vstack([
            X_val_scaled[-sequence_length:],
            X_test_scaled
        ])

        y_test_context = np.concatenate([
            y_val_scaled[-sequence_length:],
            y_test_scaled
        ])

        X_test_seq, y_test_seq = self.create_sequences_lstm(
            X_test_context,
            y_test_context,
            sequence_length
        )

        test_dates = test["week_canton"].reset_index(drop=True)

        if dif:
            train_anchor = train["rr_log_raw"].iloc[sequence_length - 1]
            val_anchor = train["rr_log_raw"].iloc[-1]
            test_anchor = val["rr_log_raw"].iloc[-1]
        else:
            train_anchor, val_anchor, test_anchor = None, None, None

        anchors = (train_anchor, val_anchor, test_anchor)

        return (X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y, train_dates, val_dates, test_dates, test, epsilon, anchors)

    def partition_lstm_full(self, dif = False):
        
        df_temp = self.df.copy()
    
        df_temp = self.drop_lags(i = 5, f = 9, df_temp = df_temp)
        
        for i in range(5, 9):
            df_temp.drop(columns = [f"rr_lag_{i}"], inplace = True)
        
        df_temp.drop(columns=['clasi_rr'], inplace=True)
        df_temp.drop(columns = ["casos"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0"], inplace = True)

        df_temp.drop(columns = ["casos_lag_1"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_1"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_1"], inplace = True)
        df_temp.drop(columns = ["casos_lag_2"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_2"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_2"], inplace = True)
        df_temp.drop(columns = ["casos_lag_3"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_3"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_3"], inplace = True)
        df_temp.drop(columns = ["casos_lag_4"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_4"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_4"], inplace = True)
        df_temp.drop(columns = ["casos_lag_5"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_5"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_5"], inplace = True)
        df_temp.drop(columns = ["casos_lag_6"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_6"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_6"], inplace = True)
        df_temp.drop(columns = ["casos_lag_7"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_7"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_7"], inplace = True)
        df_temp.drop(columns = ["casos_lag_8"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_lag_8"], inplace = True)
        df_temp.drop(columns = ["clasi_rr_no_0_lag_8"], inplace = True)

        df_temp.drop(columns = ["nino34ssta"], inplace = True)
        df_temp.drop(columns = ["tmax_mean"], inplace = True)
        df_temp.drop(columns = ["tmin_mean"], inplace = True)
        df_temp.drop(columns = ["temp_prom"], inplace = True)
        df_temp.drop(columns = ["precip_total"], inplace = True)
        df_temp.drop(columns = ["precip_max"], inplace = True)
        df_temp.drop(columns = ["precip_min"], inplace = True)
        df_temp.drop(columns = ["precip_median"], inplace = True)
        df_temp.drop(columns = ["precip_mean"], inplace = True)
        df_temp.drop(columns = ["ndvi"], inplace = True)
        df_temp.drop(columns = ["aedes_aegypti"], inplace = True)
        df_temp.drop(columns = ["criaderos_de_mosquito"], inplace = True)
        df_temp.drop(columns = ["fiebre"], inplace = True)
        df_temp.drop(columns = ["plaquetas"], inplace = True)
        df_temp.drop(columns = ["buscar_trabajo"], inplace = True)
        df_temp.drop(columns = ["desempleo"], inplace = True)
        df_temp.drop(columns = ["empleo"], inplace = True)
        df_temp.drop(columns = ["comprar_carro"], inplace = True)
        df_temp.drop(columns = ["préstamo"], inplace = True)
        df_temp.drop(columns = ["tiquetes_avion"], inplace = True)
        df_temp.drop(columns = ["vacaciones"], inplace = True)
        df_temp.drop(columns = ["corte_agua"], inplace = True)
        df_temp.drop(columns = ["corte_de_agua"], inplace = True)
        df_temp.drop(columns = ["suspension_de_agua"], inplace = True)
        df_temp.drop(columns = ["dengue"], inplace = True)

        df_temp = pd.get_dummies(df_temp, columns = ["urb"])

        parts = df_temp["week_canton"].str.split("-", expand=True)
        df_temp["year"] = parts[0].astype(int)
        df_temp["week"] = parts[1].astype(int)
        df_temp["canton"] = parts[2].astype(int)
    
        iso_date_str = parts[0] + "-W" + parts[1].str.zfill(2) + "-7"

        df_temp["date"] = pd.to_datetime(iso_date_str, format="%G-W%V-%u")

        df_temp.loc[df_temp["year"] >= 2021, "date"] -= pd.Timedelta(weeks=1)

        df_temp["rr_log_raw"] = np.log(df_temp["rr"] + 0.2)

        df_temp = (
            df_temp
            .sort_values(["canton", "date"])
            .reset_index(drop=True)
        )
        
        if dif:
            df_temp["rr_log"] = (
                df_temp
                .groupby("canton")["rr_log_raw"]
                .diff()
            )

            df_temp = (
                df_temp
                .dropna(subset=["rr_log"])
                .reset_index(drop=True)
            )
        else:
            df_temp["rr_log"] = df_temp["rr_log_raw"]

        df_temp = (
            df_temp
            .sort_values(["canton", "date"])
            .reset_index(drop=True)
        )

        canton_ids = sorted(df_temp["canton"].unique())

        cov_exclude = [
            "rr",
            "week_canton",
            "year",
            "week",
            "canton",
            "date",
            "rr_log",
            "rr_log_raw"
        ]

        cov_cols = [
            c for c in df_temp.columns
            if c not in cov_exclude
        ]

        canton_ids = sorted(df_temp["canton"].unique())

        target_series = []
        raw_log_series = []
        cov_series = []

        for canton in canton_ids:

            group = df_temp[df_temp["canton"] == canton].copy()

            target_series.append(
                TimeSeries.from_dataframe(
                    group,
                    time_col="date",
                    value_cols="rr_log",
                    freq="W-SUN"
                )
            )

            raw_log_series.append(
                TimeSeries.from_dataframe(
                    group,
                    time_col="date",
                    value_cols="rr_log_raw",
                    freq="W-SUN"
                    )
                )

            if cov_cols:
                cov_series.append(
                    TimeSeries.from_dataframe(
                        group,
                        time_col="date",
                        value_cols=cov_cols,
                        freq="W-SUN"
                    )
                )

        train_end = pd.Timestamp("2023-12-31")
        val_start = pd.Timestamp("2024-01-01")
        val_end = pd.Timestamp("2024-12-31")
        test_start = pd.Timestamp("2025-01-01")
        test_end = pd.Timestamp("2025-12-31")

        train_y = [
            ts.drop_after(train_end)
            for ts in target_series
        ]

        val_y = [
            ts.slice(val_start, val_end)
            for ts in target_series
        ]

        val_log_raw = [
            ts.slice(val_start, val_end)
            for ts in raw_log_series
        ]

        test_y = [
            ts.slice(test_start, test_end)
            for ts in target_series
        ]

        test_log_raw = [
            ts.slice(test_start, test_end)
            for ts in raw_log_series
        ]

        train_log_last = {
            canton: raw_log_series[i]
            .drop_after(train_end)
            .values()[-1, 0]
            for i, canton in enumerate(canton_ids)
        }

        val_log_last = {
            canton: raw_log_series[i]
            .drop_after(val_end)
            .values()[-1, 0]
            for i, canton in enumerate(canton_ids)
        }

        for canton in canton_ids:

            group = (
                df_temp[df_temp["canton"] == canton]
                .sort_values("date")
                .copy()
            )

            expected_dates = pd.date_range(
                start=group["date"].min(),
                end=group["date"].max(),
                freq="W-SUN"
            )

        scaler_y = Scaler(global_fit=True)
        train_y_scaled = scaler_y.fit_transform(train_y)
        full_y_scaled = scaler_y.transform(target_series)
        val_y_scaled = scaler_y.transform(val_y)
        test_y_scaled = scaler_y.transform(test_y)

        if cov_series:
            train_X = [
            ts.drop_after(train_end)
            for ts in cov_series
            ]

            scaler_X = Scaler(global_fit=True)

            train_X_scaled = scaler_X.fit_transform(train_X)
            full_X_scaled = scaler_X.transform(cov_series)
        else:
            train_X = None

            train_X_scaled = None

            scaler_X = None

        for i, ts in enumerate(train_X_scaled):
            values = ts.values()

            print(
                canton_ids[i],
                "covariates:",
                np.min(values),
                np.max(values),
                "finite:",
                np.all(np.isfinite(values))
            )

        return full_y_scaled, val_y_scaled, test_y_scaled, val_log_raw, test_log_raw, train_log_last, val_log_last, full_X_scaled, scaler_X, scaler_y, canton_ids, val_end, test_end
    
    def fit_lstm(self, dif = False):

        start = time.perf_counter()

        self.set_seed(42)

        param_distributions = {
            "sequence_length": [4, 8, 12, 26, 52],
            "units": [8, 16, 32, 64],
            "dropout": [0.0, 0.1, 0.2, 0.3],
            "batch_size": [16, 32, 64],          
            "epochs": [100],                     
            "learning_rate": [0.001, 0.005, 0.01]
        }

        n_iter = 10

        nrmse_list = []
        models = []
        seq_lens = []
        param_list = []
        stopped_epochs = []

        for i in range(n_iter):

            params = {key: random.choice(values) for key, values in param_distributions.items()}

            X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y, train_dates, val_dates, test_dates, test, epsilon, anchors  = self.partition_lstm(sequence_length=params["sequence_length"], dif = dif)
            
            model_temp = keras.Sequential([
            keras.layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            keras.layers.LSTM(units=params["units"]),
            keras.layers.Dropout(rate=params["dropout"]),
            keras.layers.Dense(1)
        ])

            model_temp.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]), metrics=["mse"])

            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True
            )

            history = model_temp.fit(
                X_train_seq, 
                y_train_seq, 
                validation_data=(X_val_seq, y_val_seq), # Explicitly pass validation data
                batch_size=params["batch_size"],          
                callbacks=[early_stopping],  
                epochs = params["epochs"],                 # Apply the callback
                verbose=0
            )

            y_pred_temp, nrmse_temp = self.lstm_predict_nrmse(X_val_seq, y_val_seq, epsilon, scaler_y = scaler_y, model = model_temp, anchor_y = anchors[1], dif = dif)

            best_epoch = early_stopping.stopped_epoch - early_stopping.patience + 1
            if best_epoch <= 0: 
                best_epoch = params["epochs"] # Default if early stopping didn't trigger
                
            stopped_epochs.append(best_epoch)

            nrmse_list.append(nrmse_temp)
            models.append(model_temp)
            seq_lens.append(params["sequence_length"])
            param_list.append(params)

            keras.backend.clear_session()

        best_idx = np.argmin(nrmse_list)
        best_params = param_list[best_idx]
        best_seq_len = seq_lens[best_idx]
        best_refit_epochs = stopped_epochs[best_idx]

        (
            X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, 
            X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, 
            scaler_X, scaler_y, train_dates, val_dates, test_dates, test, epsilon, anchors
        ) = self.partition_lstm(sequence_length=best_seq_len, dif=dif)

        refit_model = keras.Sequential([
            keras.layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            keras.layers.LSTM(units=best_params["units"]),
            keras.layers.Dropout(rate=best_params["dropout"]),
            keras.layers.Dense(1)
        ])

        refit_model.compile(
            loss="mse", 
            optimizer=keras.optimizers.Adam(learning_rate=best_params["learning_rate"])
        )

        X_seq_combined = np.vstack([X_train_seq, X_val_seq])
        y_seq_combined = np.concatenate([y_train_seq, y_val_seq])

        history = refit_model.fit(
            X_seq_combined, 
            y_seq_combined, 
            batch_size=best_params["batch_size"],    
            epochs=best_refit_epochs,                 
            verbose=0
        )

        end = time.perf_counter()
        comp_time = end - start

        print(f"LSTM model ready for {self.canton}. Elapsed time: {comp_time:.2f} seconds")

        file_suffix = "_dif" if dif else ""
        model_artifacts = {
            'model': refit_model, 
            'scaler_X': scaler_X, 
            'scaler_y': scaler_y, 
            'history': history, 
            'seq_len': best_seq_len, 
            'params': best_params,
            'epochs': best_refit_epochs
        }

        joblib.dump(model_artifacts, f"../saved_models/lstm/lstm_{self.canton}{file_suffix}.joblib")

        return comp_time

    def create_sequences_lstm(self, X, y, sequence_length):
    
            X_seq = []
            y_seq = []
    
            for i in range(sequence_length, len(X)):
    
                X_seq.append(
                    X[i-sequence_length:i]
                )
    
                y_seq.append(
                    y[i]
                )
    
            return np.array(X_seq), np.array(y_seq)

    def fit_lstm_full(self, dif=False):

        self.set_seed_full(seed=42)

        start = time.perf_counter()

        full_y_scaled, val_y_scaled, test_y_scaled, val_log_raw, test_log_raw, train_log_last, val_log_last, full_X_scaled, scaler_X, scaler_y, canton_ids, val_end, test_end = self.partition_lstm_full(dif=dif)

        param_distributions = {
            "sequence_length": [4, 8, 12, 26, 52],
            "units": [8, 16, 32, 64],
            "dropout": [0.0, 0.1, 0.2, 0.3],
            "batch_size": [16, 32, 64],
            "epochs": [100],
            "learning_rate": [0.001]
        }

        val_scores, param_list, stopped_epochs = [], [], []

        n_iter = 1

        seed = 42

        internal_train_end = pd.Timestamp("2023-12-31")
        internal_val_start = pd.Timestamp("2024-01-01")
        internal_val_end = pd.Timestamp("2024-12-31")

        internal_train_y = [
            ts.slice(ts.start_time(), internal_train_end)
            for ts in full_y_scaled
        ]

        internal_val_y = [
            ts.slice(internal_val_start, internal_val_end)
            for ts in full_y_scaled
        ]

        internal_train_X = [
            ts.slice(ts.start_time(), internal_train_end)
            for ts in full_X_scaled
        ]

        internal_val_X = [
            ts.slice(internal_val_start, internal_val_end)
            for ts in full_X_scaled
        ]

        for iter_idx in range(n_iter):

            params = {k: random.choice(v) for k, v in param_distributions.items()}

            sequence_length = params["sequence_length"] 

            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                patience=15,
                min_delta=1e-4,
                mode="min"
            )

            model_temp = RNNModel(
                model="LSTM",
                input_chunk_length=sequence_length,
                training_length=sequence_length + 1,
                output_chunk_length=1,
                hidden_dim=params["units"],
                dropout=params["dropout"],
                batch_size=params["batch_size"],
                n_epochs=params["epochs"],
                optimizer_kwargs={"lr": params["learning_rate"]},
                random_state=seed + iter_idx, 
                pl_trainer_kwargs={
                    "accelerator": "cpu",
                    "enable_progress_bar": False,
                    "deterministic": True,         
                    "callbacks": [early_stop_callback],
                    "enable_model_summary": False
                }
            )

            model_temp.fit(
                series=internal_train_y,
                future_covariates=internal_train_X,
                val_series=internal_val_y,
                val_future_covariates=internal_val_X
            )

            print(model_temp.trainer.callback_metrics)

            stopped_epoch = model_temp.trainer.current_epoch + 1

            stopped_epoch = min(
                stopped_epoch,
                params["epochs"]
            )

            stopped_epochs.append(stopped_epoch)

            _, nrmse_temp = self.lstm_predict_nrmse_full(
                model=model_temp,

                full_y_scaled=full_y_scaled,

                val_y_scaled=val_y_scaled,

                val_log_raw=val_log_raw,

                train_log_last=train_log_last,

                full_X_scaled=full_X_scaled,

                scaler_y=scaler_y,

                canton_ids=canton_ids,

                dif=dif,

                epsilon=0.2
            )

            val_scores.append(nrmse_temp)
            param_list.append(params)

        best_idx = np.argmin(val_scores)  
        best_params = param_list[best_idx]
        best_refit_epochs = stopped_epochs[best_idx]

        train_val_y = [
                ts.slice(
                    ts.start_time(),
                    val_end
                )
                for ts in full_y_scaled
            ]

        if full_X_scaled is not None:

            train_val_X = [
                ts.slice(
                    ts.start_time(),
                    val_end
                )
                for ts in full_X_scaled
            ]

        else:

            train_val_X = None

        self.set_seed_full(seed=42)

        refit_model = RNNModel(
            model="LSTM",

            input_chunk_length=(
                best_params["sequence_length"]
            ),

            training_length=(
                best_params["sequence_length"] + 1
            ),

            output_chunk_length=1,

            hidden_dim=best_params["units"],
            dropout=best_params["dropout"],
            batch_size=best_params["batch_size"],

            n_epochs=best_refit_epochs,

            optimizer_kwargs={
                "lr": best_params["learning_rate"]
            },

            random_state=42,

            pl_trainer_kwargs={
                "accelerator": "cpu",
                "enable_progress_bar": False,
                "deterministic": True,
                "enable_model_summary": False
            }
        )

        refit_model.fit(
            series=train_val_y,
            future_covariates=train_val_X
        )

        comp_time = time.perf_counter() - start

        file_suffix = "_dif" if dif else ""

        artifacts = {
            "model": refit_model,
            "scaler_y": scaler_y,
            "scaler_X": scaler_X,
            "best_params": best_params,
            "epochs": best_refit_epochs,
            "seq_len": best_params["sequence_length"],
            "canton_ids": canton_ids
        }

        joblib.dump(
            artifacts,
            f"../saved_models/lstm/lstm_full{file_suffix}.joblib"
        )

        print(
            f"Global LSTM ready for {file_suffix}. "
            f"Elapsed time: {comp_time:.2f}s"
        )

        return comp_time

    def lstm_predict_nrmse(self, X_test_seq, y_test_seq, epsilon, anchor_y, scaler_y = None, model = None, dif = False):

        if model is None:
            file_suffix = "_dif" if dif else ""
            artifact = joblib.load(
                f'../saved_models/lstm/lstm_{self.canton}{file_suffix}.joblib'
            )

            model = artifact["model"]
            scaler_y = artifact["scaler_y"]

        pred_scaled = model.predict(
            X_test_seq,
            verbose=0
        ).ravel()

        pred_diff_or_log = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        y_true_diff_or_log = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()

        if dif:
            if anchor_y is None:
                raise ValueError("anchor_y must be provided when dif=True for inverse cumulative sum.")
            y_test_log = anchor_y + np.cumsum(y_true_diff_or_log)
            rolling_anchors = np.roll(y_test_log, 1)
            rolling_anchors[0] = anchor_y
            pred_log = rolling_anchors + pred_diff_or_log
        else:
            pred_log = pred_diff_or_log
            y_test_log = y_true_diff_or_log

        y_pred = np.exp(pred_log) - epsilon
        y_true = np.exp(y_test_log) - epsilon

        y_pred = np.maximum(y_pred, 0)

        rmse = root_mean_squared_error(
            y_true,
            y_pred
        )

        nrmse = rmse / (np.mean(y_true) + epsilon)

        return y_pred, nrmse

    def lstm_predict_nrmse_full(self,
    model,
    full_y_scaled,
    val_y_scaled,
    val_log_raw,
    train_log_last,
    scaler_y = None,
    full_X_scaled=None,
    canton_ids=None,
    dif=False,
    epsilon=0.2):

        if model is None:
            file_suffix = "_dif" if dif else ""
            artifact = joblib.load(
                f'../saved_models/lstm/lstm_{self.canton}{file_suffix}.joblib'
            )

            model = artifact["model"]
            scaler_y = artifact["scaler_y"]

        val_start = val_y_scaled[0].start_time()
        val_end = val_y_scaled[0].end_time()

        historical_preds_scaled = model.historical_forecasts(
            series=full_y_scaled,
            future_covariates=full_X_scaled,
            start=val_start,
            start_format="value",
            forecast_horizon=1,
            stride=1,
            retrain=False,
            last_points_only=True,
            verbose=False,
            apply_globally=True
        )

        if isinstance(
            historical_preds_scaled,
            TimeSeries
        ):

            historical_preds_scaled = [
                historical_preds_scaled
            ]

        historical_preds_unscaled = [
            scaler_y.inverse_transform(pred)
            for pred in historical_preds_scaled
        ]

        for i, pred in enumerate(historical_preds_unscaled):
            values = pred.values().ravel()

            print(
                f"Canton {canton_ids[i]}: "
                f"min={np.min(values):.4f}, "
                f"max={np.max(values):.4f}, "
                f"finite={np.all(np.isfinite(values))}"
            )

        
        if canton_ids is None:

            canton_ids = list(
                range(len(historical_preds_unscaled))
            )

        y_true_all = []
        y_pred_all = []

        
        prediction_records = []

        for i, canton_id in enumerate(canton_ids):

            pred_series = historical_preds_unscaled[i]

            true_series = val_log_raw[i]

            pred_series = pred_series.slice(
                val_start,
                val_end
            )

            true_series = true_series.slice(
                val_start,
                val_end
            )

            common_times = (
                pred_series.time_index
                .intersection(
                    true_series.time_index
                )
            )

            # ========================================================
            # 8. LOOP THROUGH VALIDATION WEEKS
            # ========================================================

            for timestamp in common_times:

                # Model output.
                pred_transformed = (
                    pred_series
                    .slice(timestamp, timestamp)
                    .values()[0, 0]
                )

                # Actual log(RR + epsilon).
                true_log = (
                    true_series
                    .slice(timestamp, timestamp)
                    .values()[0, 0]
                )

                # ====================================================
                # DIFFERENCED MODEL
                # ====================================================

                if dif:

                    # ------------------------------------------------
                    # CHANGED:
                    #
                    # Reconstruct using the ACTUAL previous week's
                    # value.
                    #
                    # We do NOT use one fixed anchor for all of 2024.
                    # ------------------------------------------------

                    previous_timestamp = (
                        timestamp - true_series.freq
                    )

                    # First validation observation.
                    if previous_timestamp < val_start:

                        # Last observed 2023 value.
                        previous_log = (
                            train_log_last[canton_id]
                        )

                    else:

                        # Actual previous validation observation.
                        previous_log = (
                            true_series
                            .slice(
                                previous_timestamp,
                                previous_timestamp
                            )
                            .values()[0, 0]
                        )

                    # Δlog(RR + epsilon)
                    # ->
                    # log(RR + epsilon)
                    pred_log = (
                        previous_log
                        + pred_transformed
                    )

                # ====================================================
                # NON-DIFFERENCED MODEL
                # ====================================================

                else:

                    # Model directly predicts log(RR + epsilon).
                    pred_log = pred_transformed

                # ====================================================
                # INVERSE LOG TRANSFORMATION
                # ====================================================

                pred_rr = (
                    np.exp(pred_log)
                    - epsilon
                )

                true_rr = (
                    np.exp(true_log)
                    - epsilon
                )

                y_pred_all.append(pred_rr)
                y_true_all.append(true_rr)

                prediction_records.append({
                    "canton": canton_id,
                    "date": timestamp,
                    "y_true": true_rr,
                    "y_pred": pred_rr
                })

        # ============================================================
        # 9. CONVERT TO NUMPY ARRAYS
        # ============================================================

        y_true_all = np.asarray(y_true_all)
        y_pred_all = np.asarray(y_pred_all)

        # ============================================================
        # 10. SANITY CHECK
        # ============================================================

        print(
            f"Validation predictions: {len(y_pred_all)}"
        )

        print(
            f"Validation observations: {len(y_true_all)}"
        )

        # For 2 cantons and approximately 52 weeks:
        #
        #     ~104 observations
        #
        # For your complete 82-canton dataset:
        #
        #     ~82 × 52 = ~4264 observations
        #
        # depending on the exact number of ISO weeks represented.

        # ============================================================
        # 11. CHECK FOR INVALID VALUES
        # ============================================================

        if not np.all(np.isfinite(y_true_all)):

            raise ValueError(
                "Non-finite values found in y_true."
            )

        if not np.all(np.isfinite(y_pred_all)):

            raise ValueError(
                "Non-finite values found in y_pred."
            )

        # ============================================================
        # 12. CALCULATE RMSE
        # ============================================================

        rmse = np.sqrt(
            np.mean(
                (y_true_all - y_pred_all) ** 2
            )
        )

        # ============================================================
        # 13. CALCULATE YOUR NRMSE
        # ============================================================

        nrmse = (
            rmse
            / (
                np.mean(y_true_all)
                + epsilon
            )
        )

        # ============================================================
        # 14. RETURN
        # ============================================================

        return y_pred_all, nrmse

    def prediction_intervals_aci_lstm(
        self,
        X_train_seq,
        y_train_seq,
        X_test_seq,
        y_test_seq,
        epsilon,
        scaler_y,
        anchor_test=None,
        alpha=0.05,
        gamma=0.005,
        n_resamplings=30,
        dif=False
    ):
        """
        ACI prediction intervals for LSTM using BlockBootstrap via SciKeras wrapper.
        """
        start = time.perf_counter()

        file_suffix = "_dif" if dif else ""
        artifacts = joblib.load(f"../saved_models/lstm/lstm_{self.canton}{file_suffix}.joblib")
        best_params = artifacts["params"]

        def build_lstm_model():
            model = keras.Sequential([
                keras.layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                keras.layers.LSTM(units=best_params["units"]),
                keras.layers.Dropout(rate=best_params["dropout"]),
                keras.layers.Dense(1)
            ])
            model.compile(
                loss="mse",
                optimizer=keras.optimizers.Adam(learning_rate=best_params["learning_rate"])
            )
            return model

        base_estimator = KerasRegressor(
            model=build_lstm_model,
            epochs=35,
            batch_size=best_params["batch_size"],
            verbose=0
        )

        cv = BlockBootstrap(
            n_resamplings=n_resamplings,
            length=1,
            overlapping=False,
            random_state=42,
        )

        mapie = TimeSeriesRegressor(
            estimator=base_estimator,
            method="aci",
            cv=cv,
            conformity_score=AbsoluteConformityScore(sym=True),
            agg_function="mean",
            random_state=42,
            n_jobs=1  
        )

        mapie.fit(X_train_seq, y_train_seq)

        lower_scaled, upper_scaled = [], []

        pred_scaled = []

        for i in range(len(X_test_seq)):
            x_i = X_test_seq[i:i+1]
            y_i = y_test_seq[i:i+1]

            y_pred_i, interval = mapie.predict(
                x_i,
                confidence_level=1 - alpha,
                ensemble=True,
                optimize_beta=False
            )

            pred_scaled.append(y_pred_i[0])

            lower_scaled.append(interval[0, 0, 0])
            upper_scaled.append(interval[0, 1, 0])

            mapie.adapt_conformal_inference(
                x_i,
                y_i,
                gamma=gamma,
                ensemble=True,
                optimize_beta=False
            )

        lower = scaler_y.inverse_transform(np.array(lower_scaled).reshape(-1, 1)).ravel()
        upper = scaler_y.inverse_transform(np.array(upper_scaled).reshape(-1, 1)).ravel()

        if dif:
            if anchor_test is None:
                raise ValueError("anchor_test must be provided when dif=True.")

            pred_diff_or_log = scaler_y.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).ravel()

            y_test_diff_or_log = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()
            y_test_log = anchor_test + np.cumsum(y_test_diff_or_log)

            rolling_anchors = np.roll(y_test_log, 1)
            rolling_anchors[0] = anchor_test

            pred_log = rolling_anchors + pred_diff_or_log

            lower_offset = pred_diff_or_log - lower
            upper_offset = upper - pred_diff_or_log

            lower = pred_log - lower_offset
            upper = pred_log + upper_offset

        lower = np.exp(lower) - epsilon
        upper = np.exp(upper) - epsilon
        
        end = time.perf_counter()
        print(f"BlockBootstrap ACI LSTM execution time: {end - start:.2f} seconds")

        return lower, upper

    def lstm_forecast_plot(
        self, dif = False
    ):
        file_suffix = "_dif" if dif else ""
        artifact = joblib.load(f'../saved_models/lstm/lstm_{self.canton}{file_suffix}.joblib')

        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y, train_dates, val_dates, test_dates, test, epsilon, anchors  = self.partition_lstm(sequence_length=artifact["seq_len"], dif = dif)

        train_anchor, val_anchor, test_anchor = anchors

        y_pred_train, nrmse_train = self.lstm_predict_nrmse(
            X_test_seq=X_train_seq,
            y_test_seq=y_train_seq,
            epsilon=epsilon,
            scaler_y=scaler_y,
            model=artifact["model"],
            anchor_y=train_anchor,
            dif=dif
        )

        y_pred_val, nrmse_val = self.lstm_predict_nrmse(
            X_test_seq=X_val_seq,
            y_test_seq=y_val_seq,
            epsilon=epsilon,
            scaler_y=scaler_y,
            model=artifact["model"],
            anchor_y=val_anchor,
            dif=dif
        )

        y_pred_test, nrmse_test = self.lstm_predict_nrmse(
            X_test_seq=X_test_seq,
            y_test_seq=y_test_seq,
            epsilon=epsilon,
            scaler_y=scaler_y,
            model=artifact["model"],
            anchor_y=test_anchor,
            dif=dif
        )

        if dif:
            y_train_log = train_anchor + np.cumsum(scaler_y.inverse_transform(y_train_seq.reshape(-1, 1)).ravel())
            y_val_log = val_anchor + np.cumsum(scaler_y.inverse_transform(y_val_seq.reshape(-1, 1)).ravel())
            y_test_log = test_anchor + np.cumsum(scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel())

            y_train_true = np.exp(y_train_log) - epsilon
            y_val_true = np.exp(y_val_log) - epsilon
            y_test_true = np.exp(y_test_log) - epsilon
        else:
            y_train_log = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1)).ravel()
            y_val_log = scaler_y.inverse_transform(y_val_seq.reshape(-1, 1)).ravel()
            y_test_log = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()

            y_train_true = np.exp(y_train_log) - epsilon
            y_val_true = np.exp(y_val_log) - epsilon
            y_test_true = np.exp(y_test_log) - epsilon

        # lower_test, upper_test = self.prediction_intervals_aci_lstm(
        #     X_train_seq=X_train_seq,
        #     y_train_seq=y_train_seq,
        #     X_test_seq=X_test_seq,
        #     y_test_seq=y_test_seq,
        #     epsilon=epsilon,
        #     scaler_y=scaler_y,
        #     anchor_test=test_anchor,
        #     dif=dif
        # )

        fig, ax = plt.subplots(figsize=(16, 5))

        ax.plot(
            train_dates,
            y_train_true,
            label="Real",
            marker="o",
            linestyle="dashed",
            color="#0A0909",
            linewidth=4
        )

        ax.plot(
            val_dates,
            y_val_true,
            marker="o",
            linestyle="dashed",
            color="#000000",
            linewidth=4
        )

        ax.plot(
            test_dates,
            y_test_true,
            marker="o",
            linestyle="dashed",
            color="#000000",
            linewidth=4
        )

        ax.plot(
            test_dates,
            y_pred_test,
            label="Predicted (test)",
            marker="o",
            linestyle="solid",
            color="#009E73",
            linewidth=4
        )

        ax.plot(
            train_dates,
            y_pred_train,
            label="Predicted (train)",
            marker="o",
            linestyle="solid",
            color="#9E5C00",
            linewidth=4
        )

        ax.plot(
            val_dates,
            y_pred_val,
            label="Predicted (validation)",
            marker="o",
            linestyle="solid",
            color="#170291",
            linewidth=4
        )

        # ax.fill_between(x = test_dates, label = "CI (95%)", y1 = lower_test, y2 = upper_test, alpha = 0.2, color = "#009E73")

        ax.legend(
            loc="upper left",
            fontsize=18
        )

        ax.set_xlabel(
            "Week",
            fontweight="bold",
            fontsize=20
        )

        self.ticks_years_top(ax, 25)

        ax.set_ylabel(
            "Relative risk",
            fontweight="bold",
            fontsize=20
        )

        ax.set_title(
            f"LSTM for {self.canton} (NRMSE train = {nrmse_train:.3f}, NRMSE val = {nrmse_val:.3f}, NRMSE test = {nrmse_test:.3f})"
        )

        plt.show()

    def lstm_eval_plots(self):

        artifact = joblib.load(f'../saved_models/lstm/lstm_{self.canton}.joblib')

        history = artifact["history"]

        plt.figure(figsize=(12, 4))

        # Gráfico de Pérdida (Loss)
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Entrenamiento', color='blue', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validación', color='orange', linewidth=2)
        plt.title('Curva de Pérdida (Loss)')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()

    def split_sequence(self, sequence, n_steps):
        X, y = [], []
        for i in range(len(sequence) - n_steps):
            X.append(sequence[i : i + n_steps])
            y.append(sequence[i + n_steps])
        return np.array(X), np.array(y)

    def generate_mbb(self, X_1d, y_1d, block_length, seed):

        np.random.seed(seed)

        N = len(y_1d)
        n_ob = N - block_length + 1  
        num_blocks = int(np.ceil(N / block_length))  

        start_indices = np.random.randint(0, n_ob, size=num_blocks)
        
        boot_X_blocks = [X_1d[idx : idx + block_length] for idx in start_indices]
        boot_y_blocks = [y_1d[idx : idx + block_length] for idx in start_indices]
        
        boot_X = np.vstack(boot_X_blocks)[:N]
        boot_y = np.concatenate(boot_y_blocks)[:N]
    
        return boot_X, boot_y


    def mbb_lstm_ensemble_pipeline(self,
        X_train_scaled, 
        y_train_scaled,
        X_eval_seq, 
        # y_eval_scaled, 
        block_length = 8,
        n_bootstrap=50, 
        epsilon=0.2
    ):

        artifact = joblib.load(f'../saved_models/lstm/lstm_{self.canton}.joblib')
        model = artifact["model"]
        params = artifact["params"]
        scaler_y = artifact["scaler_y"]
        
        sequence_length = params["sequence_length"]
        n_features = X_train_scaled.shape[1]
        
        all_predictions_raw = []
        base_seed = 42
        
        for i in range(n_bootstrap):

            iteration_seed = base_seed + i

            self.set_seed(iteration_seed)

            boot_X_scaled, boot_y_scaled = self.generate_mbb(
            X_train_scaled, 
            y_train_scaled, 
            block_length=block_length,
            seed=iteration_seed
            )
            
            X_boot_seq, y_boot_seq = self.create_sequences_lstm(
                boot_X_scaled, 
                boot_y_scaled, 
                sequence_length=sequence_length
            )

            history = model.fit(X_boot_seq, 
            y_boot_seq, 
            batch_size=params["batch_size"], 
            epochs=params["epochs"],               
            verbose=0
            )

            preds = model.predict(X_eval_seq, verbose=0).flatten()
            all_predictions_raw.append(preds)
            
            tf.keras.backend.clear_session()

        all_predictions_raw = np.array(all_predictions_raw)

        all_preds_original_scale = []
        for i in range(all_predictions_raw.shape[0]):
            preds_i = all_predictions_raw[i, :]
            unscaled = scaler_y.inverse_transform(preds_i.reshape(-1, 1)).ravel()
            original_scale = np.exp(unscaled) - epsilon
            all_preds_original_scale.append(original_scale)

        all_preds_original_scale = np.array(all_preds_original_scale)

        # y_eval_unscaled = scaler_y.inverse_transform(y_eval_unscaled.reshape(-1, 1)).ravel()
        # y_eval_true = np.exp(y_eval_unscaled) - epsilon

        #nrmse_list = [root_mean_squared_error(y_eval_true, y_pred) / (np.mean(y_eval_true) + epsilon)for y_pred in all_preds_original_scale]

        point_forecast = np.mean(all_preds_original_scale, axis=0)
        lower_bound_forecast = np.percentile(all_preds_original_scale, 2.5, axis=0)
        upper_bound_forecast = np.percentile(all_preds_original_scale, 97.5, axis=0)

        # point_nrmse = np.mean(nrmse_list)
        # lower_bound_nrmse = np.percentile(nrmse_list, 2.5, axis = 0)
        # upper_bound_nrmse = np.percentile(nrmse_list, 97.5, axis = 0)
        
        return point_forecast, lower_bound_forecast, upper_bound_forecast
        
        # point_nrmse, lower_bound_nrmse, upper_bound_nrmse
                
    def ticks_years_top(self, ax, n):

        """
        Method that adjusts plot ticks in the time series plot so that it looks more organized. 

        Parameters
        ----
        ax: Axes
            canvas were the data will be plotted

        n: int
            show every n-th tick

        Returns
        ----
            Nothing, but adds neat ticks to the plot
        """
        ax.tick_params(axis='x', labelsize=12)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')

        ax.xaxis.set_major_locator(ticker.MultipleLocator(n))

  
    def confint_nrmse_lstm(self, which_set, n_bootstrap = 500, base_seed = 42, epsilon = 0.2):

        start = time.perf_counter()
        # Check if there's a classifier model if it's a hybrid model type

        artifact = joblib.load(f'../saved_models/lstm/lstm_{self.canton}.joblib')
        params = artifact["params"]
        scaler_y = artifact["scaler_y"]
        model = artifact["model"]

        nrmse = []
        

        for i in range(n_bootstrap):

            iteration_seed = base_seed + i
            
            self.set_seed(iteration_seed)

        if which_set == "test":
            
            bs = self.generate_mbb(
                X_train_scaled,           
                y_train_scaled, 
                block_length = 8,
                seed = 42
            )
            n = len(X_test)
        elif which_set == "train":
            bs = IIDBootstrap(
                X_combined,
                y_combined,
                seed = 42
            )
            n = len(X_combined)
        elif which_set == "val":
            bs = IIDBootstrap(
                X_combined,
                y_combined,
                seed = 42
            )
            n = len(X_combined)
        else:
            print("Choose valid set")


            preds_scaled = model.predict(X_boot)
            preds_unscaled = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
            preds = np.exp(preds_unscaled) - epsilon

            y_unscaled = scaler_y.inverse_transform(y_boot.reshape(-1, 1)).ravel()
            y_true = np.exp(y_boot) - epsilon
                
            nrmse.append(root_mean_squared_error(y_true = y_true, y_pred = preds) / (np.mean(np.exp(y_boot) - epsilon) + epsilon))

        point = np.mean(nrmse)
        
        lower_bound = np.percentile(nrmse, 2.5)
        upper_bound = np.percentile(nrmse, 97.5)

        end = time.perf_counter()
        
        print(f"Elapsed time: {end - start:.2f} seconds")

        return self.canton, point, lower_bound, upper_bound
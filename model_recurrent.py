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

    def partition_lstm_full(
        self,
        sequence_length=8,
        epsilon=0.2,
        feature_columns=None,
        scaler_X=None,
        scaler_y=None,
        medians=None
    ):
        """
        Prepare pooled LSTM data.

        Splits are determined by the target observation's year:
            Train:      2020-2023
            Validation: 2024
            Test:       2025

        Returns a dictionary containing sequences, scalers, and metadata.
        """

        df = self.df.copy()

        if "week_canton" not in df.columns:
            raise ValueError("The dataframe must contain 'week_canton'.")

        if "rr" not in df.columns:
            raise ValueError("The dataframe must contain the target 'rr'.")

        # ---------------------------------------------------------
        # 1. Parse identifiers: year-week-canton
        # ---------------------------------------------------------
        pattern = r"^(\d{4})-(\d{1,2})-(\d+)$"

        parsed = df["week_canton"].astype(str).str.extract(pattern)

        if parsed.isna().any().any():
            invalid = df.loc[
                parsed.isna().any(axis=1), "week_canton"
            ].head().tolist()

            raise ValueError(
                f"Invalid week_canton identifiers. Examples: {invalid}"
            )

        df["year"] = parsed[0].astype(int)
        df["week"] = parsed[1].astype(int)
        df["canton_id"] = parsed[2].astype(str)

        if not df["week"].between(1, 52).all():
            raise ValueError(
                "Weeks must be numbered 1-52 after removing week 53."
            )

        df["rr"] = pd.to_numeric(df["rr"], errors="coerce")

        # Keep the period relevant to this model.
        df = df.loc[
            df["year"].between(2020, 2025) & df["rr"].notna()
        ].copy()

        if df.empty:
            raise ValueError("No usable observations from 2020-2025.")

        if (df["rr"] + epsilon <= 0).any():
            raise ValueError(
                "log(rr + epsilon) requires rr + epsilon > 0."
            )

        df = df.sort_values(
            ["canton_id", "year", "week"]
        ).reset_index(drop=True)

        # The index treats every year as having 52 weeks.
        # Thus 2020-week 52 and 2021-week 1 are consecutive.
        df["week_index"] = df["year"] * 52 + df["week"]

        # ---------------------------------------------------------
        # 2. Select predictor columns
        # ---------------------------------------------------------
        excluded = {
            "week_canton",
            "year",
            "week",
            "canton_id",
            "week_index",
            "rr",
            "rr_log",
            "rr_log_raw",
            "target_log",
            "date",
            "ds",
        }

        if feature_columns is None:
            feature_columns = [
                col for col in df.columns
                if col not in excluded
                and pd.api.types.is_numeric_dtype(df[col])
            ]
        else:
            missing = [
                col for col in feature_columns
                if col not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"Predictor columns missing from dataframe: {missing}"
                )

            feature_columns = list(feature_columns)

        if not feature_columns:
            raise ValueError("No numeric predictor columns were found.")

        # Convert any infinite values to missing values.
        X_raw = df[feature_columns].copy()
        X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

        train_rows = df["year"].between(2020, 2023).to_numpy()

        if not train_rows.any():
            raise ValueError("No training observations from 2020-2023.")

        # ---------------------------------------------------------
        # 3. Impute predictors using training data only
        # ---------------------------------------------------------
        if medians is None:
            medians = X_raw.loc[train_rows].median().fillna(0.0)

        X_raw = X_raw.fillna(medians).fillna(0.0)

        # ---------------------------------------------------------
        # 4. Scale predictors using training data only
        # ---------------------------------------------------------
        if scaler_X is None:
            scaler_X = MinMaxScaler()
            scaler_X.fit(X_raw.loc[train_rows])

        X_scaled = scaler_X.transform(X_raw).astype(np.float32)

        # ---------------------------------------------------------
        # 5. Transform and scale target
        # ---------------------------------------------------------
        target_log = np.log(df["rr"].to_numpy() + epsilon).reshape(-1, 1)

        if scaler_y is None:
            scaler_y = MinMaxScaler()
            scaler_y.fit(target_log[train_rows])

        y_scaled = scaler_y.transform(target_log).astype(np.float32)

        # ---------------------------------------------------------
        # 6. Build windows independently for each municipality
        # ---------------------------------------------------------
        split_data = {
            "train": {"X": [], "y": [], "meta": []},
            "val": {"X": [], "y": [], "meta": []},
            "test": {"X": [], "y": [], "meta": []},
        }

        for canton, group in df.groupby("canton_id", sort=True):
            positions = group.index.to_numpy()

            years = df.loc[positions, "year"].to_numpy()
            weeks = df.loc[positions, "week"].to_numpy()
            week_indices = df.loc[positions, "week_index"].to_numpy()

            # Each window ends at the target observation's week.
            for end in range(sequence_length - 1, len(positions)):
                start = end - sequence_length + 1

                window_positions = positions[start:end + 1]
                window_indices = week_indices[start:end + 1]

                # Reject windows that contain missing internal weeks.
                if not np.all(np.diff(window_indices) == 1):
                    continue

                target_position = positions[end]
                target_year = years[end]

                if 2020 <= target_year <= 2023:
                    split = "train"
                elif target_year == 2024:
                    split = "val"
                elif target_year == 2025:
                    split = "test"
                else:
                    continue

                split_data[split]["X"].append(
                    X_scaled[window_positions]
                )

                split_data[split]["y"].append(
                    y_scaled[target_position]
                )

                split_data[split]["meta"].append({
                    "week_canton": df.loc[target_position, "week_canton"],
                    "canton_id": str(canton),
                    "year": int(df.loc[target_position, "year"]),
                    "week": int(df.loc[target_position, "week"]),
                    "rr_true": float(df.loc[target_position, "rr"]),
                    "target_log": float(target_log[target_position, 0]),
                })

        # ---------------------------------------------------------
        # 7. Convert lists to arrays and dataframes
        # ---------------------------------------------------------
        for split in split_data:
            if not split_data[split]["X"]:
                raise ValueError(
                    f"No sequences were created for the '{split}' split. "
                    "Check sequence_length and data coverage."
                )

            split_data[split]["X"] = np.asarray(
                split_data[split]["X"], dtype=np.float32
            )

            split_data[split]["y"] = np.asarray(
                split_data[split]["y"], dtype=np.float32
            ).reshape(-1, 1)

            split_data[split]["meta"] = pd.DataFrame(
                split_data[split]["meta"]
            )

        return {
            "sets": split_data,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "medians": medians,
            "feature_columns": feature_columns,
            "epsilon": epsilon,
            "sequence_length": sequence_length,
            "canton_ids": sorted(df["canton_id"].unique()),
            "df": df,
        }

    def fit_lstm_full(
        self,
        n_iter=12,
        epochs=50,
        patience=15,
        seed=42,
        model_dir="../saved_models/lstm"
    ):

        self.set_seed(seed)

        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "lstm_full.keras")
        metadata_path = os.path.join(model_dir, "lstm_full.joblib")

        start_time = time.perf_counter()

        epsilon = 0.2

        search_space = {
            "sequence_length": [4, 8, 12, 26, 52],
            "units": [8, 16, 32, 64],
            "dropout": [0.0, 0.1, 0.2, 0.3],
            "batch_size": [16, 32, 64],          
            "epochs": [100],                     
            "learning_rate": [0.001, 0.005, 0.01]
        }

        search_rng = random.Random(seed)

        data_by_length = {
            seq_len: self.partition_lstm_full(
                sequence_length=seq_len,
                epsilon=epsilon
            )
            for seq_len in search_space["sequence_length"]
        }

        best_nrmse = np.inf
        best_params = None
        best_data = None
        trial_results = []

        for trial in range(n_iter):
            params = {
                key: search_rng.choice(values)
                for key, values in search_space.items()
            }

            trial_seed = seed + trial

            tf.keras.backend.clear_session()
            self.set_seed(trial_seed)

            data = data_by_length[params["sequence_length"]]
            train = data["sets"]["train"]
            val = data["sets"]["val"]

            model = tf.keras.Sequential([
                tf.keras.layers.Input(
                    shape=(
                        params["sequence_length"],
                        train["X"].shape[2]
                    )
                ),
                tf.keras.layers.LSTM(
                    params["units"],
                    dropout=params["dropout"]
                ),
            keras.layers.Dropout(rate=params["dropout"]),
                tf.keras.layers.Dense(1),
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=params["learning_rate"]
                ),
                loss="mse",
            )

            early_stopping = keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            patience=15,
                            restore_best_weights=True
                        )

            history = model.fit(
                train["X"],
                train["y"],
                validation_data=(val["X"], val["y"]),
                epochs=epochs,
                batch_size=params["batch_size"],
                shuffle=False,
                callbacks=[early_stopping],
                verbose=0,
            )

            # Validation predictions, returned to the original RR scale.
            val_pred_scaled = model.predict(
                val["X"], verbose=0
            )

            val_pred_log = data["scaler_y"].inverse_transform(
                val_pred_scaled
            ).ravel()

            val_pred_rr = np.maximum(
                np.exp(val_pred_log) - epsilon, 0.0
            )

            val_true_rr = val["meta"]["rr_true"].to_numpy()

            val_rmse = root_mean_squared_error(val_true_rr, val_pred_rr)

            denominator = val_true_rr.mean() + epsilon
            val_nrmse = val_rmse / denominator

            trial_results.append({
                "trial": trial + 1,
                **params,
                "epochs_trained": len(history.history["loss"]),
                "validation_rmse": val_rmse,
                "validation_nrmse": val_nrmse,
            })

            if np.isfinite(val_nrmse) and val_nrmse < best_nrmse:
                best_nrmse = val_nrmse
                best_params = params.copy()
                best_data = data
                best_epochs = len(history.history["loss"])

        if best_params is None:
            raise RuntimeError(
                "No successful trial produced a finite validation NRMSE."
            )

        # ---------------------------------------------------------
        # Refit final model using train + validation data (2020-2024)
        # ---------------------------------------------------------
        tf.keras.backend.clear_session()
        self.set_seed(seed + n_iter)

        sequence_length = best_params["sequence_length"]
        data = data_by_length[sequence_length]

        train = data["sets"]["train"]
        val = data["sets"]["val"]

        X_train_val = np.concatenate(
            [train["X"], val["X"]], axis=0
        )

        y_train_val = np.concatenate(
            [train["y"], val["y"]], axis=0
        )

        final_model = tf.keras.Sequential([
            tf.keras.layers.Input(
                shape=(sequence_length, X_train_val.shape[2])
            ),
            tf.keras.layers.LSTM(
                best_params["units"],
                dropout=best_params["dropout"]
            ),
            tf.keras.layers.Dense(1),
        ])

        final_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=best_params["learning_rate"]
            ),
            loss="mse",
        )

        history_final = final_model.fit(
            X_train_val,
            y_train_val,
            epochs=best_epochs,
            batch_size=best_params["batch_size"],
            shuffle=False,
            verbose=0,
        )
        
        elapsed_time = time.perf_counter() - start_time

        # Save the final refitted model, replacing the search model.
        final_model.save(model_path)

        # Save preprocessing information and selected hyperparameters.
        metadata = {
            "scaler_X": data["scaler_X"],
            "scaler_y": data["scaler_y"],
            "medians": data["medians"],
            "feature_columns": data["feature_columns"],
            "epsilon": data["epsilon"],
            "sequence_length": sequence_length,
            "canton_ids": data["canton_ids"],
            "best_params": best_params,
            "best_epochs": best_epochs,
            "validation_nrmse": float(best_nrmse),
            "seed": seed,
            "trial_results": pd.DataFrame(trial_results),
            "history": history_final
        }

        joblib.dump(metadata, metadata_path)

        return {
            "best_params": best_params,
            "validation_nrmse": float(best_nrmse),
            "elapsed_seconds": elapsed_time,
            "model_path": model_path,
            "metadata_path": metadata_path,
            "trials": pd.DataFrame(trial_results),
        }

    def predict_nrmse_lstm_full(
        self,
        model_dir="../saved_models/lstm"
    ):
        """
        Load the saved pooled LSTM and predict all available sequences.

        Returns predictions for train, validation, and test, plus:
            pooled: pooled 2025 predictions and NRMSE
            by_canton: predictions and NRMSE for each municipality
            canton_nrmse: municipality-level summary dataframe
        """

        model_path = os.path.join(model_dir, "lstm_full.keras")
        metadata_path = os.path.join(model_dir, "lstm_full.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Model metadata not found: {metadata_path}"
            )

        metadata = joblib.load(metadata_path)

        model = tf.keras.models.load_model(model_path)

        # Reuse the exact preprocessing objects from training.
        data = self.partition_lstm_full(
            sequence_length=metadata["sequence_length"],
            epsilon=metadata["epsilon"],
            feature_columns=metadata["feature_columns"],
            scaler_X=metadata["scaler_X"],
            scaler_y=metadata["scaler_y"],
            medians=metadata["medians"],
        )

        predictions = {}
        split_nrmse = {}

        # ---------------------------------------------------------
        # 1. Predict each split and reverse the transformations
        # ---------------------------------------------------------
        for split in ["train", "val", "test"]:
            split_data = data["sets"][split]

            pred_scaled = model.predict(
                split_data["X"], verbose=0
            )

            pred_log = metadata["scaler_y"].inverse_transform(
                pred_scaled
            ).ravel()

            pred_rr = np.maximum(
                np.exp(pred_log) - metadata["epsilon"], 0.0
            )

            result = split_data["meta"].copy()
            result["predicted_rr"] = pred_rr

            result = result.sort_values(
                ["canton_id", "year", "week"]
            ).reset_index(drop=True)

            predictions[split] = result

            y_true = result["rr_true"].to_numpy()
            y_pred = result["predicted_rr"].to_numpy()

            rmse = root_mean_squared_error(y_true, y_pred)
            denominator = y_true.mean() + metadata["epsilon"]

            split_nrmse[split] = rmse / denominator

        # ---------------------------------------------------------
        # 2. Pooled test predictions and NRMSE
        # ---------------------------------------------------------
        test_predictions = predictions["test"].copy()

        pooled_true = test_predictions["rr_true"].to_numpy()
        pooled_pred = test_predictions["predicted_rr"].to_numpy()

        pooled_rmse = root_mean_squared_error(pooled_true, pooled_pred)

        pooled_nrmse = pooled_rmse / (
            pooled_true.mean() + metadata["epsilon"]
        )

        pooled_result = {
            "predictions": test_predictions,
            "rmse": float(pooled_rmse),
            "nrmse": float(pooled_nrmse),
            "n_observations": len(test_predictions),
        }

        # ---------------------------------------------------------
        # 3. Municipality-specific test predictions and NRMSE
        # ---------------------------------------------------------
        by_canton = {}
        canton_scores = []

        for canton, group in test_predictions.groupby(
            "canton_id", sort=True
        ):
            group = group.copy().reset_index(drop=True)

            y_true = group["rr_true"].to_numpy()
            y_pred = group["predicted_rr"].to_numpy()

            rmse = root_mean_squared_error(y_true, y_pred)

            # Each municipality uses its own 2025 mean RR.
            denominator = y_true.mean() + metadata["epsilon"]
            nrmse = rmse / denominator

            by_canton[str(canton)] = {
                "predictions": group,
                "rmse": float(rmse),
                "nrmse": float(nrmse),
                "n_observations": len(group),
            }

            canton_scores.append({
                "canton_id": str(canton),
                "n_observations": len(group),
                "mean_rr": float(y_true.mean()),
                "rmse": float(rmse),
                "nrmse": float(nrmse),
            })

        canton_nrmse = pd.DataFrame(canton_scores).sort_values(
            "canton_id"
        ).reset_index(drop=True)

        return {
            "pooled": pooled_result,
            "by_canton": by_canton,
            "canton_nrmse": canton_nrmse,
            "predictions": predictions,
            "nrmse": split_nrmse,
            "metadata": metadata,
        }

    def plot_forecast_lstm_full(
        self,
        model_dir="../saved_models/lstm",
        results=None,
        figsize=(14, 6)
    ):
        """
        Plot actual RR and pooled-LSTM predictions for one municipality.

        Parameters
        ----------
        canton : str or int
            Municipality identifier, e.g. 101.
        model_dir : str
            Directory containing the saved model.
        results : dict or None
            Optional output from predict_nrmse_lstm_full().
            If omitted, predictions are calculated automatically.
        figsize : tuple
            Figure size.

        Returns
        -------
        fig, ax
        """

        if results is None:
            results = self.predict_nrmse_lstm_full(
                model_dir=model_dir
            )

        # ---------------------------------------------------------
        # 1. Prepare actual observations
        # ---------------------------------------------------------
        df = self.df.copy()

        parsed = df["week_canton"].astype(str).str.extract(
            r"^(\d{4})-(\d{1,2})-(\d+)$"
        )

        if parsed.isna().any().any():
            raise ValueError(
                "Could not parse all week_canton identifiers."
            )

        df["year"] = parsed[0].astype(int)
        df["week"] = parsed[1].astype(int)
        df["canton_id"] = parsed[2].astype(str)
        df["rr"] = pd.to_numeric(df["rr"], errors="coerce")

        actual = df.loc[
            (df["canton_id"] == self.canton)
            & df["year"].between(2020, 2025)
            & df["rr"].notna()
        ].copy()

        if actual.empty:
            raise ValueError(
                f"No observations found for municipality {self.canton}."
            )

        actual["time"] = (
            actual["year"] + (actual["week"] - 1) / 52
        )

        actual = actual.sort_values(
            ["year", "week"]
        )

        # ---------------------------------------------------------
        # 2. Plot actual observations and split predictions
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(
            actual["time"],
            actual["rr"],
            color="black",
            marker="o",
            linewidth=4,
            label="Real",
            zorder=1,
        )

        split_styles = {
            "train": {
                "color": "#9E5C00",
                "label": "Train predictions",
                "marker": "o", 
                "linestyle":"solid",
                "linewidth": "4"
            },
            "val": {
                "color": "#170291",
                "label": "Val predictions",
                "marker": "o", 
                "linestyle":"solid",
                "linewidth": "4"
            },
            "test": {
                "color": "#009E73",
                "label": "Test predictions",
                "marker": "o", 
                "linestyle":"solid",
                "linewidth": "4"
            },
        }

        nrmse_split = []
        for split, style in split_styles.items():
            pred_df = results["predictions"][split]
            nrmse_split.append(results["nrmse"][split])

            pred_df = pred_df.loc[
                pred_df["canton_id"] == self.canton
            ].copy()

            if pred_df.empty:
                continue

            pred_df["time"] = (
                pred_df["year"] + (pred_df["week"] - 1) / 52
            )

            pred_df = pred_df.sort_values(
                ["year", "week"]
            )

            ax.plot(
                pred_df["time"],
                pred_df["predicted_rr"],
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                label=style["label"],
                zorder=2,
            )

        # ---------------------------------------------------------
        # 3. Format plot
        # ---------------------------------------------------------
        ax.set_title(
            f"Pooled LSTM: Dengue Relative Risk — Municipality {self.canton} (NRMSE train = {nrmse_split[0]:.3f}, NRMSE val = {nrmse_split[1]:.3f}, NRMSE test = {nrmse_split[2]:.3f})"
        )

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
        
        ax.set_xlabel("Year")
        ax.set_ylabel("Relative risk (RR)")

        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        plt.show()

        return fig, ax

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
         
        artifact = joblib.load(f'../saved_models/lstm/lstm_{self.canton}.joblib')
  
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
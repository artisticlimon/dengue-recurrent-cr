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
            self.df = df[df["week_canton"].str.contains(canton)] # Filter for the desired canton
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

    def serie_temp_canton(self, model_type, X_combined, y_combined, y_test, lower_bound, upper_bound, epsilon, lims = [0, 7]):

        """
        Method that plots the time series of actual and predicted relative risk values for the test set, including the confidence intervals for the predictions. It reads the results from the csv file where they were saved after making predictions with the trained model.

        Parameters
        ----
        model_type: str
            the type of model for which to plot the results (options: "rf", "xgb", "hrf", "hxgb")
        X_combined, y_combined: dataframe, Pandas Series
            training and validation data
        y_test: pandas series
            test relative risk data
        lower_bound: array-like
            array-like with the lower bound of the confidence intervals for the predictions, to be used as the lower limit shaded area around the predicted values in the plot.
        upper_bound: array-like
            array-like with the upper bound of the confidence intervals for the predictions, to be used as the upper limit shaded area around the predicted values in the plot.
        epsilon: float
            value used before log-transform
        lims: list
            list with the limits for the y-axis of the plot, to be used as the ylim parameter in the plot. 
        Returns
        ----
            Nothing, just shows the plot with the time series of actual and predicted relative risk values for the test set, including the confidence intervals for the predictions.
        """
        preds = pd.read_csv(f'../data/model_results/results_{model_type}_full.csv')
        results = preds[preds["week_canton"].str.contains(self.canton)] 

        df_copy = self.df.copy()

        df_copy['week_canton'] = df_copy['week_canton'].str.extract(r'(\d{4}-\d+)').iloc[:, 0].tolist()

        y_pred_train = self.calculate_train_predictions(model_type, X_combined, epsilon)

        y_combined = np.exp(y_combined) - epsilon
        y_test = np.exp(y_test) - epsilon
        
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(df_copy['week_canton'].iloc[0:len(y_combined)], y_combined, label='Real', marker='o', linestyle = "dashed", color = "#000000", linewidth=4)
        ax.plot(df_copy['week_canton'].iloc[len(y_combined):], y_test, marker='o', linestyle = "dashed", color = "#000000", linewidth=4)
        ax.plot(df_copy['week_canton'].iloc[len(y_combined):], results["pred"], label='Predicted', marker='o',linestyle = "solid", color = "#009E73", linewidth=4)
        ax.plot(df_copy['week_canton'].iloc[0:len(y_combined)], y_pred_train, marker='o', linestyle = "solid", color = "#009E73", linewidth=4)
        ax.fill_between(x = df_copy['week_canton'].iloc[len(y_combined):], label = "CI (95%)", y1 = lower_bound, y2 = upper_bound, alpha = 0.2, color = "#009E73")
        ax.legend(loc = "upper right", fontsize = 18)
        y_ticks = list(range(0, int(np.ceil(lims[1]))))
        ax.set_yticks(y_ticks)
        y_labels = [str(tick) if i % 2 == 0 else "" for i, tick in enumerate(y_ticks)]
        ax.set_yticklabels(y_labels, fontsize=14, fontweight="bold")
        self.ticks_years_top(ax, 25)
        ax.set_xlabel('Week', fontweight = "bold", fontsize = 20)
        ax.set_ylabel(f'Relative risk', fontweight = "bold", fontsize = 20)
        ax.set_ylim(lims)
        ax.set_xlim([0, 304])

        # ax.set_title(f"{model_type} for {self.canton}")
        plt.show()

    def custom_threshold_score(self, y_true, y_pred_proba, threshold):

        """
        Scorer used to calculate the permutation importance for the classification part of the hybrid models. It uses a custom threshold to convert predicted probabilities into binary predictions, and then calculates the negative root mean squared error between the true labels and the predicted labels.
        
        Parameters
        ----
        y_true: Pandas series
            True labels for the classification task.
        y_pred_proba: array-like
            Predicted probabilities for the positive class.
        threshold: float
            Threshold to convert predicted probabilities into binary predictions.
        """
    
        if y_pred_proba.ndim == 2:
            y_scores = y_pred_proba[:, 1]
        else:
            # If it is already 1-dimensional, use it directly
            y_scores = y_pred_proba
        
        y_pred= (y_scores >= threshold).astype(int)
        
        return -1 * (root_mean_squared_error(y_true, y_pred)) ** 2 # The neg root mean squared error is used as the scoring metric for the permutation importance, to be consistent with the scoring used in the regression part of the hybrid models.
 
    
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

    def prediction_intervals_aci(self,
        model_type,
        X_train,
        y_train,
        X_test,
        y_test,
        epsilon,
        alpha=0.05,
        gamma=0.005,
        n_resamplings=100
    ):
        """
        This method uses Adaptive Conformal Inference (ACI) to construct prediction intervals.

        Parameters
        ----------
        model_type: str
            the type of model, which can be "rf", "xgb", "hrf", or "hxgb"
        X_train, y_train: dataframe, Pandas series
            training data
        X_test, y_test: dataframe, Pandas series
            test data (y_test is only used for online updating)
        epsilon: float
            value used before log-transform
        alpha: float
            miscoverage level (0.05 -> 95% PI)
        gamma: float 
            adaptation rate
        n_resamplings: 
            number of bootstrap resamples

        Returns
        -------
        y_pred: array-like
            point predictions
        lower: array-like
            lower prediction interval
        upper: array-like
            upper prediction interval
        """

        start = time.perf_counter()

        try:
            grid = joblib.load(f'../models/saved_models/{model_type}_reg_full.joblib')
        except: 
            grid = joblib.load(f'../models/saved_models/{model_type}_full.joblib')
        
        best_reg = grid.best_estimator_

        X_train = X_train.to_numpy(dtype=np.float32)
        X_test = X_test.to_numpy(dtype=np.float32)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        cv = BlockBootstrap(
            n_resamplings=n_resamplings,
            length=1,
            overlapping=False,
            random_state=42,
        )

        mapie = TimeSeriesRegressor(
            estimator=best_reg,
            method="aci",
            cv=cv,
            conformity_score=AbsoluteConformityScore(sym=True),
            agg_function="mean",
            random_state=42,
            n_jobs=-1,
        )

        mapie.fit(X_train, y_train)

        lower = []
        upper = []

        for x, y in zip(X_test, y_test):

            pred, interval = mapie.predict(
                x.reshape(1, -1),
                confidence_level=1 - alpha,
                ensemble=True,
                optimize_beta=False,
            )

            lower.append(interval[0, 0, 0])
            upper.append(interval[0, 1, 0])

            mapie.adapt_conformal_inference(
                x.reshape(1, -1),
                np.array([y]),
                gamma=gamma,
                ensemble=True,
                optimize_beta=False,
            )

        lower = np.exp(lower) - epsilon
        upper = np.exp(upper) - epsilon

        end = time.perf_counter()
        
        print(f"Elapsed time: {end - start:.2f} seconds")

        return (
            np.asarray(lower),
            np.asarray(upper),
        )
    
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
        # Exog contains the variables for the test set
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

    def partition_lstm(self, sequence_length, epsilon=0.2):

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
            if col not in ["rr", "week_canton", "year", "week"]
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
            y_train.to_numpy().reshape(-1, 1)
        ).ravel()

        y_val_scaled = scaler_y.transform(
            y_val.to_numpy().reshape(-1, 1)
        ).ravel()

        y_test_scaled = scaler_y.transform(
            y_test.to_numpy().reshape(-1, 1)
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

        return (X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, scaler_X, scaler_y, train_dates, val_dates, test_dates, test, epsilon)
    
    def fit_lstm(self):

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

        n_iter = 50

        nrmse_list = []
        models = []
        seq_lens = []
        param_list = []

        for i in range(n_iter):

            params = {key: random.choice(values) for key, values in param_distributions.items()}

            X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, scaler_X, scaler_y, train_dates, val_dates, test_dates, test, epsilon = self.partition_lstm(sequence_length=params["sequence_length"])
            
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

            y_pred_temp, nrmse_temp = self.lstm_predict_nrmse(X_val_seq, y_val_seq, epsilon, scaler_y = scaler_y, model = model_temp)

            nrmse_list.append(nrmse_temp)
            models.append(model_temp)
            seq_lens.append(params["sequence_length"])
            param_list.append(params)

            keras.backend.clear_session()

        best_nrmse = np.argmin(nrmse_list)
        model = models[best_nrmse]
        seq_len = seq_lens[best_nrmse]
        best_params = param_list[best_nrmse]

        end = time.perf_counter()
        
        comp_time = end - start

        print(f"LSTM model ready for {self.canton}. Elapsed time: {comp_time:.2f} seconds")

        model_artifacts = {'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y, 'history': history, 'seq_len': seq_len}

        joblib.dump(model_artifacts, f"../saved_models/lstm/lstm_{self.canton}.joblib")

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

    def lstm_predict_nrmse(self, X_test_seq, y_test_seq, epsilon, scaler_y = None, model = None):

        if model is None:
            artifact = joblib.load(
                f'../saved_models/lstm/lstm_{self.canton}.joblib'
            )

            model = artifact["model"]
            scaler_y = artifact["scaler_y"]

        # Predictions are in scaled log space
        pred_scaled = model.predict(
            X_test_seq,
            verbose=0
        ).ravel()

        # Back to log(rr + epsilon)
        pred_log = scaler_y.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).ravel()

        y_test_log = scaler_y.inverse_transform(
            y_test_seq.reshape(-1, 1)
        ).ravel()

        # Back to original relative-risk scale
        y_pred = np.exp(pred_log) - epsilon
        y_true = np.exp(y_test_log) - epsilon

        # Avoid tiny negative numerical values
        y_pred = np.maximum(y_pred, 0)

        rmse = root_mean_squared_error(
            y_true,
            y_pred
        )

        nrmse = rmse / (np.mean(y_true) + epsilon)

        return y_pred, nrmse

    def lstm_forecast_plot(
        self,
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq,
        X_test_seq,
        y_test_seq,
        scaler_y,
        train_dates,
        val_dates,
        test_dates,
        epsilon
    ):

        y_pred_test, nrmse_test = self.lstm_predict_nrmse(
            X_test_seq,
            y_test_seq,
            epsilon
        )

        y_pred_train, nrmse_train = self.lstm_predict_nrmse(
            X_train_seq,
            y_train_seq,
            epsilon
        )

        y_pred_val, nrmse_val = self.lstm_predict_nrmse(
                X_val_seq,
                y_val_seq,
                epsilon
        )

        y_train_log = scaler_y.inverse_transform(
            y_train_seq.reshape(-1, 1)
        ).ravel()

        y_val_log = scaler_y.inverse_transform(
            y_val_seq.reshape(-1, 1)
        ).ravel()

        y_test_log = scaler_y.inverse_transform(
            y_test_seq.reshape(-1, 1)
        ).ravel()

        y_train_true = np.exp(y_train_log) - epsilon
        y_val_true = np.exp(y_val_log) - epsilon
        y_test_true = np.exp(y_test_log) - epsilon

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


    def generate_mbb_series(self, series_1d, block_length):

        np.random.seed(42)

        N = len(series_1d)
        n_ob = N - block_length + 1  
        num_blocks = int(np.ceil(N / block_length))  

        start_indices = np.random.randint(0, n_ob, size=num_blocks)
        
        boot_blocks = [series_1d[idx : idx + block_length] for idx in start_indices]
        boot_series = np.concatenate(boot_blocks)[:N]  
        
        return boot_series


    def mbb_lstm_ensemble_pipeline(self,
        series, 
        n_steps, 
        block_length, 
        n_features, 
        n_bootstrap=50, 
        test_ratio=1/3
    ):

        series_values = np.array(series).flatten()
        total_len = len(series_values)
        train_size = int(total_len * (1 - test_ratio))
        
        train_series = series_values[:train_size]
        test_series = series_values[train_size - n_steps:]  
        
        X_test, Y_test = self.split_sequence(test_series, n_steps)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        all_predictions = []

        artifact = joblib.load(f'../saved_models/lstm/lstm_{self.canton}.joblib')
        params = artifact["best_params"]
        
        for i in range(n_bootstrap):
            boot_train_series = self.generate_mbb_series(train_series, block_length)
            
            # Split bootstrap series into training inputs (X_train) and outputs (Y_train)
            X_train_boot, Y_train_boot = self.split_sequence(boot_train_series, n_steps)
            X_train_boot = X_train_boot.reshape((X_train_boot.shape[0], X_train_boot.shape[1], 1))
            
            # Fit independent LSTM instance
            model_temp = keras.Sequential([
                keras.layers.Input(shape=(n_steps, n_features)),
                keras.layers.LSTM(units=params["units"]),
                keras.layers.Dropout(rate=params["dropout"]),
                keras.layers.Dense(1)
            ])

            model_temp.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]), metrics=["mse"])

            history = model_temp.fit(
                X_train_boot, 
                Y_train_boot, 
                batch_size=params["batch_size"], 
                epochs = params["epochs"],                
                verbose=0
            )

            
            # Predict on the exact same original Test Set
            preds = model_temp.predict(X_test, verbose=0).flatten()
            all_predictions.append(preds)
            
            tf.keras.backend.clear_session()
            
            if (i + 1) % 5 == 0 or (i + 1) == n_bootstrap:
                print(f"Completed Bootstrap Model {i + 1}/{n_bootstrap}")

        all_predictions = np.array(all_predictions)

        point = np.mean(all_predictions, axis=0)
        lower_bound = np.percentile(all_predictions, 2.5, axis=0)
        upper_bound = np.percentile(all_predictions, 97.5, axis=0)

        return (point, lower_bound, upper_bound)
                
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


    def calculate_train_nrmse(self, model_type, X_combined, y_combined, epsilon):
        """
        Method that calculates the NRMSE for the predictions from a model of a certain type, using the training set.

        Parameters
        ----
        model_type: str
            model for which the NRMSE will be calculated (options: "rf", "xgb", "hrf", "hxgb")
        X_combined, y_combined: dataframe, pandas Series
            training and validation data (y_combined is log-transformed and with epsilon added)
        epsilon: float
            the small value that was added to the relative risk before log-transforming.
        
        Returns
        ----
            self.canton: str
                the canton included in the class instance

            nrmse: float
                the calculated NRMSE
        """

        predictions = self.calculate_train_predictions(self, model_type, X_combined, epsilon)

        y_true = np.exp(y_combined) - epsilon

        rmse = root_mean_squared_error(y_true, predictions)
        
        nrmse = rmse / (np.mean(y_true) + epsilon)

        return self.canton, nrmse

    def calculate_test_nrmse(self, model_type, epsilon):

        """
        Method that calculates the NRMSE for the test set predictions from a model of a certain type.

        Parameters
        ----
        model_type: str
            model for which the NRMSE will be calculated (options: "rf", "xgb", "hrf", "hxgb")

        epsilon: float
            the small value that was added to the relative risk before log-transforming.

        Returns
        ----
            self.canton: str
                the canton included in the class instance

            nrmse: float
                the calculated NRMSE
        """
        
        results = pd.read_csv(f'../data/model_results/results_{model_type}_{self.canton}.csv')

        rmse = root_mean_squared_error(results["actual"], results["pred"])
        
        nrmse = rmse / (np.mean(results["actual"]) + epsilon)

        return self.canton, nrmse
  
    def confint_nrmse(self, model_type, X_combined, y_combined, X_test, y_test, epsilon, which_set, n_bootstraps = 500):

        """
        Method that calculates the point estimate NRMSE for the predictions from an individual model of a certain type, using either the training or test set, and calculates a 95% confidence interval for the NRMSE using bootstrapping.

        Parameters
        ----
        model_type: str
            model for which the NRMSE will be calculated (options: "rf", "xgb", "hrf", "hxgb")
        
        X_combined, y_combined: dataframe, pandas Series
            training and validation data

        X_test, y_test: dataframe, pandas Series
            test data
        
        epsilon: float
            the small value that was added to the relative risk before log-transforming.

        which_set: str
            indicates which set to use for the NRMSE calculation and bootstrapping ("train" or "test")
        
        n_bootstraps: int
            number of bootstrap resamples to use for the confidence interval calculation
        
        Returns
        ----
            self.canton: str
                the canton included in the class instance

            point: float    
                the point estimate for the NRMSE, calculated by averaging the NRMSEs from the bootstrap resamples

            lower_bound: float
                the lower bound of the 95% confidence interval for the NRMSE, calculated by taking the 2.5th percentile of the NRMSEs from the bootstrap resamples

            upper_bound: float
                the upper bound of the 95% confidence interval for the NRMSE, calculated by taking the 97.5th percentile of the NRMSEs from the bootstrap resamples
        """

        start = time.perf_counter()
        # Check if there's a classifier model if it's a hybrid model type
        try:
            grid_classi = joblib.load(f'../models/saved_models/{model_type}_classi_{self.canton}.joblib')
            modelo_classi = grid_classi.best_estimator_
        except FileNotFoundError: 
            grid_classi = None

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
        try:
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_reg_{self.canton}.joblib')
        except FileNotFoundError: 
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_{self.canton}.joblib')
        
        modelo_reg = grid_reg.best_estimator_

        if which_set == "test":
            bs = IIDBootstrap(
                X_test,
                y_test,
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
        else:
            print("Choose valid set")

        nrmse = []

        if grid_classi == None and model_type == "hrf":
            print("Only one class present in train set for classification. Skipping classification step.")
        elif grid_classi == None and model_type == "hxgb":
            print("Only one class present in train set for classification. Skipping classification step.")

        if model_type == "hrf":
            if self.canton == "full":
                thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hrf.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
            else: 
                thresholds = pd.read_csv("../data/model_results/thresholds/thresholds/thresholds_hrf.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == self.canton]["best_th"].values[0]
        elif model_type == "hxgb":
            if self.canton == "full":
                thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hxgb.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
            else: 
                thresholds = pd.read_csv("../data/model_results/thresholds/thresholds_hxgb.csv")
                thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == self.canton]["best_th"].values[0]

        for data in bs.bootstrap(n_bootstraps):

            X_boot = data[0][0]
            y_boot = data[0][1]

            if model_type == "rf" or model_type == "xgb":
                preds = modelo_reg.predict(X_boot)
                preds = np.exp(preds) - epsilon
            elif model_type == "hrf" or model_type == "hxgb":
                pred_reg = modelo_reg.predict(X_boot)
                pred_reg = np.exp(pred_reg) - epsilon
                prob_classi = modelo_classi.predict_proba(X_boot)[:, 1]
                preds = np.where(prob_classi < best_threshold, 0, pred_reg)

            y_true = np.exp(y_boot) - epsilon
                
            nrmse.append(root_mean_squared_error(y_true = y_true, y_pred = preds) / (np.mean(np.exp(y_boot) - epsilon) + epsilon))

        point = np.mean(nrmse)
        
        lower_bound = np.percentile(nrmse, 2.5)
        upper_bound = np.percentile(nrmse, 97.5)

        end = time.perf_counter()
        
        print(f"Elapsed time: {end - start:.2f} seconds")

        return self.canton, point, lower_bound, upper_bound
    
    def calculate_nrmse_full(self, model_type, X_combined, y_combined, X_test, y_test, epsilon, which_set):

        """
        Method that calculates the NRMSE for the predictions from the global model of a certain type, using either the training or test set

        Parameters
        ----
        model_type: str
            model for which the NRMSE will be calculated (options: "rf", "xgb", "hrf", "hxgb")
        
        X_combined, y_combined: dataframe, pandas Series
            training and validation data

        X_test, y_test: dataframe, pandas Series
            test data
        
        epsilon: float
            the small value that was added to the relative risk before log-transforming.

        which_set: str
            indicates which set to use for the NRMSE calculation and bootstrapping ("train" or "test")

        Returns
        ----
            self.canton: str
                the canton included in the class instance

            nrmse: float    
                the point estimate for the NRMSE, using the predictions from our data
        """

        # Check if there's a classifier model if it's a hybrid model type
        try:
            grid_classi = joblib.load(f'../models/saved_models/{model_type}_classi_full.joblib')
            modelo_classi = grid_classi.best_estimator_
        except FileNotFoundError: 
            grid_classi = None

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
        try:
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_reg_full.joblib')
            modelo_reg = grid_reg.best_estimator_
        except FileNotFoundError: 
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_full.joblib')
            modelo_reg = grid_reg.best_estimator_

        if model_type == "hrf":
            thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hrf.csv")
            thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == "full"]["best_th"].values[0]
        elif model_type == "hxgb":
            thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hxgb.csv")
            thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == "full"]["best_th"].values[0]

        expected_features = list(modelo_reg.feature_names_in_)
    
        urb_features = [col for col in expected_features if col.startswith("urb_")]
            
        for X in [X_combined, X_test]:
            for col in urb_features:
                if col not in X.columns:
                    X[col] = 0

        X_combined = X_combined[expected_features]
        X_test = X_test[expected_features]
        
        if which_set == "test":
            X_eval = X_test
            y_eval = y_test
        elif which_set == "train":
            X_eval = X_combined
            y_eval = y_combined
        else:
            raise ValueError("which_set must be 'test' or 'train'")
        
        if grid_classi is None:
            predictions = modelo_reg.predict(X_eval)
            predictions = np.exp(predictions) - epsilon
        else:
            y_pred_reg = modelo_reg.predict(X_eval)
            y_pred_reg = np.exp(y_pred_reg) - epsilon
            prob_classi = modelo_classi.predict_proba(X_eval)[:, 1]
            predictions = np.where(prob_classi < best_threshold, 0, y_pred_reg)

        y_true = np.exp(y_eval) - epsilon
        
        rmse = root_mean_squared_error(y_true, predictions)
        nrmse = rmse / (np.mean(y_true) + epsilon)

        return self.canton, nrmse
        
    def confint_nrmse_full(self, model_type, X_combined, y_combined, X_test, y_test, epsilon, which_set, n_bootstraps = 500):

        """
        Method that calculates the point estimate NRMSE for the predictions from a global model of a certain type, using either the training or test set, and calculates a 95% confidence interval for the NRMSE using bootstrapping.

        Parameters
        ----
        model_type: str
            model for which the NRMSE will be calculated (options: "rf", "xgb", "hrf", "hxgb")
        
        X_combined, y_combined: dataframe, pandas Series
            training and validation data

        X_test, y_test: dataframe, pandas Series
            test data
        
        epsilon: float
            the small value that was added to the relative risk before log-transforming.

        which_set: str
            indicates which set to use for the NRMSE calculation and bootstrapping ("train" or "test")
        
        n_bootstraps: int
            number of bootstrap resamples to use for the confidence interval calculation
        
        Returns
        ----
            self.canton: str
                the canton included in the class instance

            point: float    
                the point estimate for the NRMSE, calculated by averaging the NRMSEs from the bootstrap resamples

            lower_bound: float
                the lower bound of the 95% confidence interval for the NRMSE, calculated by taking the 2.5th percentile of the NRMSEs from the bootstrap resamples

            upper_bound: float
                the upper bound of the 95% confidence interval for the NRMSE, calculated by taking the 97.5th percentile of the NRMSEs from the bootstrap resamples
        """

        start = time.perf_counter()

        try:
            grid_classi = joblib.load(f'../models/saved_models/{model_type}_classi_full.joblib')
            modelo_classi = grid_classi.best_estimator_
        except FileNotFoundError: 
            grid_classi = None
            modelo_classi = None

        # The file names are different for the regression models (with "_reg" in the name) and the hybrid models (without "_reg" in the name), so we need to try both options when loading the model.
        try:
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_reg_full.joblib')
        except FileNotFoundError: 
            grid_reg = joblib.load(f'../models/saved_models/{model_type}_full.joblib')
            
        modelo_reg = grid_reg.best_estimator_

        expected_features = list(modelo_reg.feature_names_in_)
    
        urb_features = [col for col in expected_features if col.startswith("urb_")]
            
        for X in [X_combined, X_test]:
            for col in urb_features:
                if col not in X.columns:
                    X[col] = 0
     
        X_combined = X_combined[expected_features]
        X_test = X_test[expected_features]
    
        if which_set == "test":
            bs = IIDBootstrap(
                X_test,
                y_test,
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
        else:
            print("Choose valid set")
    
        nrmse = []
    
        if grid_classi == None and model_type == "hrf":
            print("Only one class present in train set for classification. Skipping classification step.")
        elif grid_classi == None and model_type == "hxgb":
            print("Only one class present in train set for classification. Skipping classification step.")

        if model_type == "hrf":
            thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hrf.csv")
            thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == "full"]["best_th"].values[0]
        elif model_type == "hxgb":
            thresholds = pd.read_csv("../data/model_results/thresholds/full_threshold_hxgb.csv")
            thresholds["canton"] = thresholds["canton"].astype(str)
            best_threshold = thresholds[thresholds["canton"] == "full"]["best_th"].values[0]
    
        for data in bs.bootstrap(n_bootstraps):
    
            X_boot = data[0][0]
            y_boot = data[0][1]
    
            if model_type == "rf" or model_type == "xgb" or grid_classi == None:
                preds = modelo_reg.predict(X_boot)
                preds = np.exp(preds) - epsilon
            elif model_type == "hrf" or model_type == "hxgb":
                pred_reg = modelo_reg.predict(X_boot)
                pred_reg = np.exp(pred_reg) - epsilon
                prob_classi = modelo_classi.predict_proba(X_boot)[:, 1]
                preds = np.where(prob_classi < best_threshold, 0, pred_reg)

            y_true = np.exp(y_boot) - epsilon
            nrmse.append(root_mean_squared_error(y_true = y_true, y_pred = preds) / (np.mean(y_true) + epsilon))

        point = np.mean(nrmse)
        lower_bound = np.percentile(nrmse, 2.5)
        upper_bound = np.percentile(nrmse, 97.5)

        end = time.perf_counter()
        
        print(f"Elapsed time: {end - start:.2f} seconds")
    
        return self.canton, point, lower_bound, upper_bound

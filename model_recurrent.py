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
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.decomposition import PCA
import random
from scikeras.wrappers import KerasRegressor
import re

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

    def set_seed(self, seed=42):
        random.seed(seed)
        
        np.random.seed(seed)
        
        tf.random.set_seed(seed)
        
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'

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
    
## ARIMA

    def partition_arima_new(self):
        df = self.df.copy()

        # Parse strings such as "2025-1-101".
        parsed = df["week_canton"].astype(str).str.extract(
            r"^(\d{4})-(\d{1,2})-(.+)$"
        )

        if parsed.isna().any().any():
            raise ValueError(
                "Could not parse every week_canton value."
            )

        df["year"] = parsed[0].astype(int)
        df["week"] = parsed[1].astype(int)
        df["canton_id"] = parsed[2].astype(str)

        # This project uses 52 weeks per year.
        if not df["week"].between(1, 52).all():
            raise ValueError(
                "Found weeks outside 1-52. Resolve week 53 "
                "before constructing week_index."
            )

        df["week_index"] = df["year"] * 52 + df["week"]

        df = (
            df.sort_values("week_index")
            .reset_index(drop=True)
        )

        if df["week_index"].duplicated().any():
            raise ValueError(
                "Duplicate week_index values. Make sure self.df "
                "contains only one canton."
            )

        # Check that week 52 of one year is followed by week 1
        # of the next year under the 52-week convention.
        differences = df["week_index"].diff().dropna()

        if not differences.eq(1).all():
            bad_positions = differences.index[
                differences.ne(1)
            ].tolist()

            raise ValueError(
                "Missing or discontinuous weeks detected. "
                f"Check rows around positions {bad_positions[:10]}. "
                "Do not silently fill missing target observations."
            )

        # Preserve the original exclusions.
        drop_base = [
            "clasi_rr", "casos", "clasi_rr_no_0",
            "nino34ssta", "tmax_mean", "tmin_mean",
            "temp_prom", "precip_total", "precip_max",
            "precip_min", "precip_median", "precip_mean",
            "ndvi", "aedes_aegypti", "criaderos_de_mosquito",
            "fiebre", "plaquetas", "buscar_trabajo",
            "desempleo", "empleo", "comprar_carro",
            "préstamo", "tiquetes_avion", "vacaciones",
            "corte_agua", "corte_de_agua",
            "suspension_de_agua", "dengue"
        ]

        df = df.drop(
            columns=[c for c in drop_base if c in df.columns],
            errors="ignore"
        )

        # Remove lags 5-8 for all predictors.
        lag_5_to_8 = re.compile(r"(?:^|_)lag_[5-8](?:_|$)")

        df = df.drop(
            columns=[
                c for c in df.columns
                if lag_5_to_8.search(c)
            ],
            errors="ignore"
        )

        # Remove these target/case-related lagged variables.
        for base in ["rr", "casos", "clasi_rr", "clasi_rr_no_0"]:
            for lag in range(1, 9):
                col = f"{base}_lag_{lag}"
                if col in df.columns:
                    df = df.drop(columns=col)

        # One-hot encode urbanization if it is categorical.
        if "urb" in df.columns:
            df = pd.get_dummies(
                df,
                columns=["urb"],
                drop_first=True,
                dtype=int
            )

        # Retain the requested period.
        df = (
            df[df["year"].between(2020, 2025)]
            .copy()
            .reset_index(drop=True)
        )

        if df.empty:
            raise ValueError("No observations remain for 2020-2025.")

        excluded = {
            "rr", "year", "week", "week_index",
            "week_canton", "canton_id"
        }

        feature_cols = [
            c for c in df.columns
            if c not in excluded
        ]

        if "rr" not in df.columns:
            raise ValueError("The target column 'rr' is missing.")

        # Convert predictor columns to numeric, failing rather
        # than silently converting invalid values into missing data.
        df[feature_cols] = df[feature_cols].apply(
            pd.to_numeric, errors="raise"
        )

        if df["rr"].isna().any():
            raise ValueError("Missing target values found in rr.")

        if not np.isfinite(df["rr"].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite values found in rr.")

        if df[feature_cols].isna().any().any():
            raise ValueError(
                "Missing predictor values found. Resolve or "
                "impute them using training-only information."
            )

        if not np.isfinite(
            df[feature_cols].to_numpy(dtype=float)
        ).all():
            raise ValueError("Non-finite predictor values found.")

        train = df[df["year"].between(2020, 2023)].copy()
        val = df[df["year"] == 2024].copy()
        test = df[df["year"] == 2025].copy()

        if train.empty or val.empty or test.empty:
            raise ValueError(
                "Train, validation, and test must all contain data."
            )

        return {
            "df": df,
            "feature_cols": feature_cols,
            "train": train,
            "val": val,
            "test": test,
        }

    
    def _select_feature_columns(self, feature_cols, lag=None):
        """
        Select exogenous predictors.

        lag=None: all eligible predictors.
        lag=k: only columns containing the exact lag_k token.
        """
        if lag is None:
            return list(feature_cols)

        pattern = re.compile(
            rf"(?:^|_)lag_{int(lag)}(?:_|$)"
        )

        selected = [
            c for c in feature_cols
            if pattern.search(c)
        ]

        if not selected:
            raise ValueError(
                f"No feature columns found for lag_{lag}."
            )

        return selected


    def _fit_preprocessors(
        self,
        rr,
        scaler_X,
        scaler_y,
        X=None,
        epsilon=0.2
    ):
        """
        Fit target scaler, predictor scaler, and PCA using
        only the data passed to this call.
        """
        rr = np.asarray(rr, dtype=float).reshape(-1)

        if np.any(rr + epsilon <= 0):
            raise ValueError("rr + epsilon must be positive.")

        y_log = np.log(rr).reshape(-1, 1)
        # Correct target transformation is log(rr + epsilon).
        y_log = np.log(rr + epsilon).reshape(-1, 1)

        if scaler_y is None: 
            scaler_y = MinMaxScaler()
            y_scaled = scaler_y.fit_transform(y_log).ravel()
        else:
            y_scaled = scaler_y.transform(y_log).ravel()
        pca = None
        X_scaled = None

        if X is not None:
            X = np.asarray(X, dtype=float)

            if scaler_X is None:
                scaler_X = MinMaxScaler()
                X_minmax = scaler_X.fit_transform(X)
            else:
                X_minmax = scaler_X.transform(X)

            pca = PCA(n_components=0.95)
            X_scaled = pca.fit_transform(X_minmax)

        return {
            "y_scaled": y_scaled,
            "X_scaled": X_scaled,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            'pca': pca
        }


    def _transform_X(self, X, scaler_X, pca):
        """Transform exogenous predictors with already-fitted objects."""
        if scaler_X is None or pca is None:
            return None

        X = np.asarray(X, dtype=float)
        X_minmax = scaler_X.transform(X)

        return pca.transform(X_minmax)


    def _transform_y(self, rr, scaler_y, epsilon=0.2):
        """Log-transform and scale rr using an existing target scaler."""
        rr = np.asarray(rr, dtype=float).reshape(-1)

        if np.any(rr + epsilon <= 0):
            raise ValueError("rr + epsilon must be positive.")

        y_log = np.log(rr + epsilon).reshape(-1, 1)

        return scaler_y.transform(y_log).ravel()


    def _inverse_transform_y(
        self,
        y_scaled,
        scaler_y,
        epsilon=0.2
    ):
        """Convert scaled predictions back to the original RR scale."""
        y_scaled = np.asarray(
            y_scaled, dtype=float
        ).reshape(-1, 1)

        y_log = scaler_y.inverse_transform(y_scaled).ravel()

        return np.exp(y_log) - epsilon


    def calculate_nrmse(self, y_true, y_pred, epsilon=0.2):
        """Calculate NRMSE on the original RR scale."""
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]

        if len(y_true) == 0:
            return np.nan

        denominator = np.mean(y_true) + epsilon

        if denominator <= 0:
            raise ValueError("NRMSE denominator must be positive.")

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        return rmse / denominator

    def _refit_selected_arima_model(
        self,
        parts,
        selected,
        scaler_X,
        scaler_y,
        epsilon=0.2,
        maxiter=100,
        save_path=None
    ):
        """
        Refit a selected ARIMA or ARIMAX candidate on 2020-2024.

        Refit target scaling, feature scaling, and PCA using only
        the combined training and validation periods.
        """
        train = parts["train"]
        val = parts["val"]
        feature_cols = parts["feature_cols"]

        train_val = (
            pd.concat([train, val])
            .sort_values("week_index")
            .reset_index(drop=True)
        )

        lag = selected["lag"]
        selected_cols = selected["selected_cols"]

        rr = train_val["rr"].to_numpy(dtype=float)

        X_raw = (
            train_val[selected_cols].to_numpy(dtype=float)
            if lag is not None else None
        )

        prep = self._fit_preprocessors(
            rr=rr,
            X=X_raw,
            epsilon=epsilon,
            scaler_X = scaler_X,
            scaler_y = scaler_y
        )

        y_scaled = pd.Series(
            prep["y_scaled"],
            index=pd.RangeIndex(len(train_val))
        )

        if lag is not None:
            pc_names = [
                f"PC{i + 1}"
                for i in range(prep["X_scaled"].shape[1])
            ]

            X_scaled = pd.DataFrame(
                prep["X_scaled"],
                index=pd.RangeIndex(len(train_val)),
                columns=pc_names
            )
        else:
            X_scaled = None
            pc_names = None

        final_model = SARIMAX(
            y_scaled,
            exog=X_scaled,
            order=selected["order"],
            seasonal_order=selected["seasonal_order"],
            trend=None,
            enforce_stationarity=True,
            enforce_invertibility=True
        ).fit(disp=False, maxiter=maxiter)

        artifacts = {
            "model": final_model,
            "model_type": "ARIMA" if lag is None else "ARIMAX",
            "canton": self.canton,
            "lag": lag,
            "selected_cols": selected_cols,
            "feature_cols": feature_cols,
            "order": selected["order"],
            "seasonal_order": selected["seasonal_order"],
            "epsilon": epsilon,
            "scaler_y": prep["scaler_y"],
            "scaler_X": prep["scaler_X"],
            "pca": prep["pca"],
            "pc_names": pc_names,
            "selection": selected,
            "validation_nrmse": selected["validation_nrmse"],
            "train_val_end_week_index": int(
                train_val["week_index"].iloc[-1]
            ),
            "seed": 42,
        }

        if save_path is not None:
            os.makedirs(
                os.path.dirname(save_path) or ".",
                exist_ok=True
            )
            joblib.dump(artifacts, save_path)

        return artifacts

    def _fit_arima_candidate(
        self,
        train,
        val,
        feature_cols,
        lag=None,
        epsilon=0.2,
        maxiter=100
    ):
        """
        Fit ARIMA (lag=None) or ARIMAX (lag=1,...,4) on 2020-2023
        and evaluate rolling one-step-ahead forecasts on 2024.

        All preprocessing is fitted on the training period only.
        """
        selected_cols = self._select_feature_columns(
            feature_cols, lag=lag
        )

        rr_train = train["rr"].to_numpy(dtype=float)
        rr_val = val["rr"].to_numpy(dtype=float)

        X_train_raw = (
            train[selected_cols].to_numpy(dtype=float)
            if lag is not None else None
        )

        X_val_raw = (
            val[selected_cols].to_numpy(dtype=float)
            if lag is not None else None
        )

        prep = self._fit_preprocessors(
            rr=rr_train,
            X=X_train_raw,
            scaler_X= None, 
            scaler_y = None,
            epsilon=epsilon
        )

        y_train = pd.Series(
            prep["y_scaled"],
            index=pd.RangeIndex(len(train))
        )

        if lag is not None:
            n_pc = prep["X_scaled"].shape[1]
            pc_names = [f"PC{i + 1}" for i in range(n_pc)]

            X_train = pd.DataFrame(
                prep["X_scaled"],
                index=pd.RangeIndex(len(train)),
                columns=pc_names
            )

            X_val_transformed = self._transform_X(
                X_val_raw,
                prep["scaler_X"],
                prep["pca"]
            )

            X_val = pd.DataFrame(
                X_val_transformed,
                index=pd.RangeIndex(
                    len(train), len(train) + len(val)
                ),
                columns=pc_names
            )
        else:
            X_train = None
            X_val = None

        auto_model = pm.auto_arima(
            y_train,
            X=X_train,
            seasonal=True,
            m=52,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            random_state=42
        )

        order = auto_model.order
        seasonal_order = auto_model.seasonal_order

        model = SARIMAX(
            y_train,
            exog=X_train,
            order=order,
            seasonal_order=seasonal_order,
            trend=None,
            enforce_stationarity=True,
            enforce_invertibility=True
        ).fit(disp=False, maxiter=maxiter)

        # Transform validation targets using the TRAIN target scaler.
        y_val_scaled = self._transform_y(
            rr_val,
            prep["scaler_y"],
            epsilon=epsilon
        )

        # Rolling one-step-ahead validation predictions.
        results = model
        pred_scaled = []

        for j in range(len(val)):
            next_index = len(train) + j

            if lag is not None:
                X_next = X_val.iloc[[j]]
                forecast = results.get_forecast(
                    steps=1,
                    exog=X_next
                )
            else:
                X_next = None
                forecast = results.get_forecast(steps=1)

            pred_scaled.append(
                float(np.asarray(forecast.predicted_mean).ravel()[0])
            )

            # Update with the ACTUAL observation after forecasting.
            y_actual = pd.Series(
                [y_val_scaled[j]],
                index=pd.Index([next_index])
            )

            if lag is not None:
                results = results.extend(
                    endog=y_actual,
                    exog=X_next
                )
            else:
                results = results.extend(endog=y_actual)

        pred_rr = self._inverse_transform_y(
            pred_scaled,
            prep["scaler_y"],
            epsilon=epsilon
        )

        validation_nrmse = self.calculate_nrmse(
            rr_val,
            pred_rr,
            epsilon=epsilon
        )

        return {
            "lag": lag,
            "selected_cols": selected_cols,
            "order": order,
            "seasonal_order": seasonal_order,
            "validation_nrmse": validation_nrmse,
            "validation_predictions": pred_rr,
            "model": model,
            "scaler_y": prep["scaler_y"],
            "scaler_X": prep["scaler_X"],
            "pca": prep["pca"],
            "pc_names": (
                [f"PC{i + 1}" for i in range(prep["X_scaled"].shape[1])]
                if lag is not None else None
            ),
        }

    def fit_and_save_arima_and_best_arimax(
        self,
        epsilon=0.2,
        maxiter=100,
        save_dir="../saved_models/arima"
    ):
        """
        Fit and save:
        1. A standalone ARIMA model.
        2. The best ARIMAX model among lags 1-4.

        Candidate selection uses validation NRMSE on 2024 only.
        Both final models are refitted on 2020-2024.
        """
        parts = self.partition_arima_new()
        feature_cols = parts["feature_cols"]

        os.makedirs(save_dir, exist_ok=True)

        # --------------------------------------------------
        # 1. Fit ARIMA separately (no exogenous predictors).
        # --------------------------------------------------
        arima_candidate = self._fit_arima_candidate(
            train=parts["train"],
            val=parts["val"],
            feature_cols=feature_cols,
            lag=None,
            epsilon=epsilon,
            maxiter=maxiter
        )

        arima_path = os.path.join(
            save_dir,
            f"arima_{self.canton}.joblib"
        )

        arima_artifacts = self._refit_selected_arima_model(
            parts=parts,
            selected=arima_candidate,
            epsilon=epsilon,
            maxiter=maxiter,
            scaler_X = None,
            scaler_y = arima_candidate["scaler_y"],
            save_path=arima_path
        )

        print(
            f"ARIMA saved to {arima_path} | "
            f"2024 validation NRMSE: "
            f"{arima_candidate['validation_nrmse']:.4f}"
        )

        # --------------------------------------------------
        # 2. Compare ARIMAX candidates using lags 1-4 only.
        # --------------------------------------------------
        arimax_candidates = []

        for lag in [1, 2, 3, 4]:
            try:
                candidate = self._fit_arima_candidate(
                    train=parts["train"],
                    val=parts["val"],
                    feature_cols=feature_cols,
                    lag=lag,
                    epsilon=epsilon,
                    maxiter=maxiter
                )

                arimax_candidates.append(candidate)

                print(
                    f"ARIMAX lag_{lag}: "
                    f"validation NRMSE = "
                    f"{candidate['validation_nrmse']:.4f}; "
                    f"order={candidate['order']}; "
                    f"seasonal_order={candidate['seasonal_order']}"
                )

            except Exception as exc:
                print(f"ARIMAX lag_{lag} failed: {exc}")

        if not arimax_candidates:
            raise RuntimeError(
                "All ARIMAX candidates failed. "
                "The standalone ARIMA model was still saved."
            )

        best_arimax = min(
            arimax_candidates,
            key=lambda result: result["validation_nrmse"]
        )

        best_lag = best_arimax["lag"]

        arimax_path = os.path.join(
            save_dir,
            f"arimax_{self.canton}.joblib"
        )

        arimax_artifacts = self._refit_selected_arima_model(
            parts=parts,
            selected=best_arimax,
            epsilon=epsilon,
            maxiter=maxiter,
            scaler_X = arima_candidate["scaler_X"],
            scaler_y = arima_candidate["scaler_y"],
            save_path=arimax_path
        )

        print(
            f"\nBest ARIMAX candidate: lag_{best_lag}"
        )
        print(
            f"2024 validation NRMSE: "
            f"{best_arimax['validation_nrmse']:.4f}"
        )
        print(f"Best ARIMAX saved to {arimax_path}")

        # Return the two models separately, along with the
        # candidate comparison results.
        comparison = pd.DataFrame([
            {
                "model": "ARIMA",
                "lag": None,
                "validation_nrmse":
                    arima_candidate["validation_nrmse"],
                "order": str(arima_candidate["order"]),
                "seasonal_order":
                    str(arima_candidate["seasonal_order"])
            },
            *[
                {
                    "model": "ARIMAX",
                    "lag": c["lag"],
                    "validation_nrmse": c["validation_nrmse"],
                    "order": str(c["order"]),
                    "seasonal_order": str(c["seasonal_order"])
                }
                for c in arimax_candidates
            ]
        ])

        comparison_path = os.path.join(
            save_dir,
            f"candidate_comparison_{self.canton}.csv"
        )
        comparison.to_csv(comparison_path, index=False)

        return {
            "arima": arima_artifacts,
            "arimax": arimax_artifacts,
            "best_arimax_lag": best_lag,
            "comparison": comparison,
            "arima_path": arima_path,
            "arimax_path": arimax_path,
        }

    def _rolling_training_predictions(
        self,
        train,
        artifacts,
        epsilon=0.2
    ):
        """
        Return in-sample one-step predictions from the saved model.
        No refitting or state extension is performed.
        """
        model = artifacts["model"]
        scaler_y = artifacts["scaler_y"]

        pred = model.get_prediction(
            start=0,
            end=len(train) - 1,
            dynamic=False
        )

        pred_scaled = np.asarray(
            pred.predicted_mean
        ).reshape(-1, 1)

        pred_rr = self._inverse_transform_y(
            pred_scaled,
            scaler_y,
            epsilon=epsilon
        ).ravel()

        if len(pred_rr) != len(train):
            raise ValueError(
                f"Expected {len(train)} predictions, "
                f"but got {len(pred_rr)}."
            )

        return pred_rr


    def _rolling_forecasts_with_fitted_model(
        self,
        model,
        frame,
        selected_cols,
        lag,
        scaler_y,
        scaler_X,
        pca,
        pc_names,
        start_index,
        epsilon=0.2
    ):
        """
        Rolling one-step forecasts using an already-fitted model
        and fixed preprocessing.

        Forecast a week, then extend the model with its actual target.
        """
        rr = frame["rr"].to_numpy(dtype=float)

        if lag is not None:
            X_raw = frame[selected_cols].to_numpy(dtype=float)

            X_transformed = self._transform_X(
                X_raw,
                scaler_X,
                pca
            )

            X_all = pd.DataFrame(
                X_transformed,
                index=pd.RangeIndex(
                    start_index, start_index + len(frame)
                ),
                columns=pc_names
            )
        else:
            X_all = None

        # Scale actual targets with the SAME scaler used for this model.
        y_actual_scaled = self._transform_y(
            rr,
            scaler_y,
            epsilon=epsilon
        )

        results = model
        predictions = []

        for j in range(len(frame)):
            current_index = start_index + j

            if lag is not None:
                X_next = X_all.iloc[[j]]
                forecast = results.get_forecast(
                    steps=1,
                    exog=X_next
                )
            else:
                X_next = None
                forecast = results.get_forecast(steps=1)

            pred_scaled = float(
                np.asarray(forecast.predicted_mean).ravel()[0]
            )

            predictions.append(
                self._inverse_transform_y(
                    [pred_scaled],
                    scaler_y,
                    epsilon=epsilon
                )[0]
            )

            # Update only after forecasting this week.
            y_new = pd.Series(
                [y_actual_scaled[j]],
                index=pd.Index([current_index])
            )

            if lag is not None:
                results = results.extend(
                    endog=y_new,
                    exog=X_next
                )
            else:
                results = results.extend(endog=y_new)

        return np.asarray(predictions)


    def predict_arima_whole_dataset(
        self,
        artifacts,
        maxiter=50
    ):
        """
        Return observed and rolling one-step-ahead predicted RR
        across training, validation, and test periods.
        """
        parts = self.partition_arima_new()

        df = parts["df"].copy()
        train = parts["train"]
        val = parts["val"]
        test = parts["test"]

        epsilon = artifacts["epsilon"]
        lag = artifacts["lag"]
        selected_cols = artifacts["selected_cols"]

        df["rr_real"] = df["rr"].astype(float)
        df["rr_pred"] = np.nan
        df["period"] = np.select(
            [
                df["year"].between(2020, 2023),
                df["year"] == 2024,
                df["year"] == 2025
            ],
            ["train", "validation", "test"],
            default="other"
        )

        # 1. Training: expanding-window forecasts.
        train_val = pd.concat([train, val]).sort_index()

        train_val_pred = self._rolling_training_predictions(
            train=train_val,
            artifacts=artifacts,
            epsilon=epsilon
        )

        df.loc[train_val.index, "rr_pred"] = train_val_pred

        # 3. Test: final model fitted on 2020-2024, with
        # preprocessing refitted on that same combined period.
        test_pred = self._rolling_forecasts_with_fitted_model(
            model=artifacts["model"],
            frame=test,
            selected_cols=selected_cols,
            lag=lag,
            scaler_y=artifacts["scaler_y"],
            scaler_X=artifacts["scaler_X"],
            pca=artifacts["pca"],
            pc_names=artifacts["pc_names"],
            start_index=len(train) + len(val),
            epsilon=epsilon
        )

        df.loc[test.index, "rr_pred"] = test_pred

        return df.sort_values("week_index").reset_index(drop=True)


    def plot_predictions_whole_dataset(
        self,
        predictions_df,
        save_path=None,
        title=None
    ):
        df = predictions_df.sort_values("week_index").copy()

        required = {
            "week_index", "year", "rr_real", "rr_pred"
        }
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        fig, ax = plt.subplots(figsize=(16, 6))

        # Shade the periods.
        period_specs = [
            (2020, 2024, "Train + val"),
            (2025, 2025, "Test")
        ]

        for start_year, end_year, label in period_specs:
            subset = df[df["year"].between(start_year, end_year)]

            if not subset.empty:
                ax.axvspan(
                    subset["week_index"].min(),
                    subset["week_index"].max(),
                    alpha=0.08,
                    label=label
                )

        ax.plot(
            df["week_index"],
            df["rr_real"],
            color="black",
            linewidth=1.4,
            label="Observed RR",
            zorder=3
        )

        ax.plot(
            df["week_index"],
            df["rr_pred"],
            linewidth=1.1,
            label="Predicted RR",
            zorder=4
        )

        # Boundary markers.
        for year in [2024, 2025]:
            boundary = year * 52 + 1
            ax.axvline(
                boundary,
                linestyle="--",
                linewidth=1
            )

        year_ticks = df.groupby("year")["week_index"].min()

        ax.set_xticks(year_ticks.to_numpy())
        ax.set_xticklabels(year_ticks.index.astype(str))

        ax.set_xlabel("Year")
        ax.set_ylabel("Relative risk (RR)")
        ax.set_title(
            title or f"Observed vs. predicted RR — {self.canton}"
        )

        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()

        if save_path is not None:
            os.makedirs(
                os.path.dirname(save_path) or ".",
                exist_ok=True
            )
            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

        return fig, ax
    
    def bootstrap_nrmse(
        self,
        predictions_df,
        n_bootstrap=2000,
        block_size=4,
        epsilon=0.2,
        seed=42,
        output_dir=None
    ):
        """
        Calculate point NRMSE and moving-block-bootstrap 95% intervals.

        Resample paired (actual, prediction) blocks within each period.
        Does not retrain the forecasting model in each bootstrap replicate.
        """
        rng = np.random.default_rng(seed)

        summary_rows = []
        distribution_rows = []

        periods = {
            "train_validation": lambda d: d["year"].between(2020, 2024),
            "test": lambda d: d["year"].eq(2025),
        }

        for period_name, mask_func in periods.items():
            period_df = predictions_df.loc[
                mask_func(predictions_df)
            ].sort_values("week_index").copy()

            period_df = period_df.dropna(
                subset=["rr_real", "rr_pred"]
            )

            y_true = period_df["rr_real"].to_numpy(dtype=float)
            y_pred = period_df["rr_pred"].to_numpy(dtype=float)
            n = len(period_df)

            if n == 0:
                print(f"No valid predictions for {period_name}; skipping.")
                continue

            point_nrmse = self.calculate_nrmse(
                y_true, y_pred, epsilon=epsilon
            )

            # Make sure the bootstrap block size is valid.
            b = min(int(block_size), n)

            if b < 1:
                raise ValueError("block_size must be at least 1.")

            # Starts of all overlapping moving blocks.
            possible_starts = np.arange(n - b + 1)

            boot_scores = np.empty(n_bootstrap, dtype=float)

            for k in range(n_bootstrap):
                sampled_indices = []

                # Sample blocks until we have n observations.
                while len(sampled_indices) < n:
                    start = int(rng.choice(possible_starts))
                    sampled_indices.extend(
                        range(start, start + b)
                    )

                sampled_indices = np.asarray(
                    sampled_indices[:n], dtype=int
                )

                # Actual and predicted values remain paired.
                yt = y_true[sampled_indices]
                yp = y_pred[sampled_indices]

                boot_scores[k] = self.calculate_nrmse(
                    yt, yp, epsilon=epsilon
                )

                distribution_rows.append({
                    "period": period_name,
                    "bootstrap_iteration": k,
                    "nrmse": boot_scores[k]
                })

            lower_95, upper_95 = np.percentile(
                boot_scores, [2.5, 97.5]
            )

            summary_rows.append({
                "period": period_name,
                "n_weeks": n,
                "point_nrmse": point_nrmse,
                "lower_95": lower_95,
                "upper_95": upper_95,
                "block_size": b,
                "n_bootstrap": n_bootstrap
            })

        summary = pd.DataFrame(summary_rows)
        distributions = pd.DataFrame(distribution_rows)

        return summary, distributions

    def prediction_intervals_aci_arima(
        self,
        artifacts,
        alpha=0.05,
        gamma=0.005,
        epsilon=None,
    ):
        """
        Adaptive Conformal Inference (ACI) prediction intervals
        for the 2025 test period of an ARIMA/ARIMAX model.

        Calibration:
            2020-2024

        Test:
            2025

        The interval is generated sequentially. After each test
        observation is observed, ACI adapts the effective miscoverage
        level for the next prediction.
        """

        start = time.perf_counter()

        parts = self.partition_arima_new()

        train = parts["train"]
        val = parts["val"]
        test = parts["test"]

        model = artifacts["model"]

        if epsilon is None:
            epsilon = artifacts["epsilon"]

        lag = artifacts["lag"]
        selected_cols = artifacts["selected_cols"]

        scaler_y = artifacts["scaler_y"]
        scaler_X = artifacts["scaler_X"]
        pca = artifacts["pca"]
        pc_names = artifacts["pc_names"]

        # ------------------------------------------------------------
        # 1. Calibration predictions: 2020-2024
        # ------------------------------------------------------------
        train_val = (
            pd.concat([train, val])
            .sort_values("week_index")
            .reset_index(drop=True)
        )

        # Use the same prediction procedure as the whole-dataset
        # prediction function.
        calibration_predictions = self._rolling_training_predictions(
            train=train_val,
            artifacts=artifacts,
            epsilon=epsilon
        )

        calibration_actual = (
            train_val["rr"]
            .to_numpy(dtype=float)
        )

        # Absolute conformity scores
        conformity_scores = np.abs(
            calibration_actual - calibration_predictions
        )

        conformity_scores = conformity_scores[
            np.isfinite(conformity_scores)
        ]

        if len(conformity_scores) == 0:
            raise ValueError(
                "No valid conformity scores available for ACI calibration."
            )

        # ------------------------------------------------------------
        # 2. Prepare test predictors
        # ------------------------------------------------------------
        if lag is not None:

            X_test_raw = test[selected_cols].to_numpy(dtype=float)

            X_test_transformed = self._transform_X(
                X_test_raw,
                scaler_X,
                pca
            )

            X_test = pd.DataFrame(
                X_test_transformed,
                index=pd.RangeIndex(
                    len(train) + len(val),
                    len(train) + len(val) + len(test)
                ),
                columns=pc_names
            )

        else:
            X_test = None

        # Actual test target on the original scale
        y_test = test["rr"].to_numpy(dtype=float)

        # ------------------------------------------------------------
        # 3. Sequential ACI
        # ------------------------------------------------------------
        results = model

        lower_test = []
        upper_test = []
        predictions_test = []

        # Current adaptive miscoverage level.
        alpha_t = alpha

        for j in range(len(test)):

            # --------------------------------------------------------
            # Forecast current test observation
            # --------------------------------------------------------
            if lag is not None:

                X_next = X_test.iloc[[j]]

                forecast = results.get_forecast(
                    steps=1,
                    exog=X_next
                )

            else:

                X_next = None

                forecast = results.get_forecast(
                    steps=1
                )

            pred_scaled = float(
                np.asarray(
                    forecast.predicted_mean
                ).ravel()[0]
            )

            pred_rr = self._inverse_transform_y(
                [pred_scaled],
                scaler_y,
                epsilon=epsilon
            )[0]

            predictions_test.append(pred_rr)

            # --------------------------------------------------------
            # Calculate adaptive conformal quantile
            # --------------------------------------------------------
            # Higher alpha_t -> narrower interval
            #
            # np.quantile uses the corresponding upper quantile
            # of the conformity scores.
            q = np.quantile(
                conformity_scores,
                1.0 - alpha_t
            )

            lower_test.append(
                pred_rr - q
            )

            upper_test.append(
                pred_rr + q
            )

            # --------------------------------------------------------
            # Observe the current test value
            # and update ACI for the NEXT prediction.
            # --------------------------------------------------------
            y_actual = y_test[j]

            covered = (
                y_actual >= lower_test[-1]
                and y_actual <= upper_test[-1]
            )

            error_t = 0 if covered else 1

            # ACI update
            alpha_t = alpha_t + gamma * (
                alpha - error_t
            )

            # Keep alpha in a sensible range.
            alpha_t = np.clip(
                alpha_t,
                0.001,
                0.999
            )

            # --------------------------------------------------------
            # Add the observed test residual to the calibration pool.
            # --------------------------------------------------------
            conformity_scores = np.append(
                conformity_scores,
                abs(y_actual - pred_rr)
            )

            # --------------------------------------------------------
            # Update ARIMA state using the ACTUAL observation.
            # --------------------------------------------------------
            y_actual_scaled = self._transform_y(
                [y_actual],
                scaler_y,
                epsilon=epsilon
            )[0]

            y_new = pd.Series(
                [y_actual_scaled],
                index=pd.Index(
                    [len(train) + len(val) + j]
                )
            )

            if lag is not None:

                results = results.extend(
                    endog=y_new,
                    exog=X_next
                )

            else:

                results = results.extend(
                    endog=y_new
                )

        lower_test = np.asarray(lower_test)
        upper_test = np.asarray(upper_test)
        predictions_test = np.asarray(predictions_test)

        # RR cannot be negative.
        lower_test = np.maximum(
            lower_test,
            0
        )

        end = time.perf_counter()

        print(
            f"ACI ARIMA execution time: "
            f"{end - start:.2f} seconds"
        )

        return (
            predictions_test,
            lower_test,
            upper_test
        )

    def arima_forecast_plot(
        self,
        model_type="arima",
        alpha=0.05,
        gamma=0.005,
        save_path=None
    ):
        """
        Plot ARIMA/ARIMAX predictions for train, validation and test.

        All predictions are shown in green.

        ACI prediction intervals are shown only for the test period.
        """

        # ------------------------------------------------------------
        # 1. Load the requested model
        # ------------------------------------------------------------
        if model_type not in ["arima", "arimax"]:
            raise ValueError(
                "model_type must be either 'arima' or 'arimax'."
            )

        artifact_path = (
            f"../saved_models/arima/"
            f"{model_type}_{self.canton}.joblib"
        )

        artifacts = joblib.load(artifact_path)

        # ------------------------------------------------------------
        # 2. Generate predictions over the whole dataset
        # ------------------------------------------------------------
        predictions_df = self.predict_arima_whole_dataset(
            artifacts=artifacts
        )

        parts = self.partition_arima_new()

        train = parts["train"]
        val = parts["val"]
        test = parts["test"]

        # ------------------------------------------------------------
        # 3. Extract predictions
        # ------------------------------------------------------------
        train_pred = predictions_df.loc[
            predictions_df["year"].between(2020, 2023),
            "rr_pred"
        ].to_numpy()

        val_pred = predictions_df.loc[
            predictions_df["year"] == 2024,
            "rr_pred"
        ].to_numpy()

        test_pred = predictions_df.loc[
            predictions_df["year"] == 2025,
            "rr_pred"
        ].to_numpy()

        # ------------------------------------------------------------
        # 4. ACI intervals ONLY for test
        # ------------------------------------------------------------
        (
            aci_test_pred,
            lower_test,
            upper_test
        ) = self.prediction_intervals_aci_arima(
            artifacts=artifacts,
            alpha=alpha,
            gamma=gamma
        )

        # ------------------------------------------------------------
        # 5. NRMSE
        # ------------------------------------------------------------
        y_train = train["rr"].to_numpy(dtype=float)
        y_val = val["rr"].to_numpy(dtype=float)
        y_test = test["rr"].to_numpy(dtype=float)

        nrmse_train = self.calculate_nrmse(
            y_train,
            train_pred,
            epsilon=artifacts["epsilon"]
        )

        nrmse_val = self.calculate_nrmse(
            y_val,
            val_pred,
            epsilon=artifacts["epsilon"]
        )

        nrmse_test = self.calculate_nrmse(
            y_test,
            test_pred,
            epsilon=artifacts["epsilon"]
        )

        # ------------------------------------------------------------
        # 6. Plot
        # ------------------------------------------------------------
        fig, ax = plt.subplots(
            figsize=(16, 5)
        )

        # ------------------------------------------------------------
        # Real observations
        # ------------------------------------------------------------
        ax.plot(
            train["week_index"],
            y_train,
            label="Real",
            marker="o",
            linestyle="dashed",
            color="#0A0909",
            linewidth=4
        )

        ax.plot(
            val["week_index"],
            y_val,
            marker="o",
            linestyle="dashed",
            color="#0A0909",
            linewidth=4
        )

        ax.plot(
            test["week_index"],
            y_test,
            marker="o",
            linestyle="dashed",
            color="#0A0909",
            linewidth=4
        )

        # ------------------------------------------------------------
        # Predictions
        # ------------------------------------------------------------
        prediction_color = "#009E73"

        ax.plot(
            train["week_index"],
            train_pred,
            label="Predicted (train)",
            marker="o",
            linestyle="solid",
            color=prediction_color,
            linewidth=4
        )

        ax.plot(
            val["week_index"],
            val_pred,
            label="Predicted (validation)",
            marker="o",
            linestyle="solid",
            color=prediction_color,
            linewidth=4
        )

        ax.plot(
            test["week_index"],
            test_pred,
            label="Predicted (test)",
            marker="o",
            linestyle="solid",
            color=prediction_color,
            linewidth=4
        )

        # ------------------------------------------------------------
        # ACI interval ONLY on test
        # ------------------------------------------------------------
        ax.fill_between(
            test["week_index"],
            lower_test,
            upper_test,
            label=f"ACI ({int((1 - alpha) * 100)}%)",
            alpha=0.2,
            color=prediction_color
        )

        # ------------------------------------------------------------
        # Labels
        # ------------------------------------------------------------
        ax.legend(
            loc="upper left",
            fontsize=18
        )

        ax.set_xlabel(
            "Week",
            fontweight="bold",
            fontsize=20
        )

        ax.set_ylabel(
            "Relative risk",
            fontweight="bold",
            fontsize=20
        )

        self.ticks_years_top(
            ax,
            25
        )

        model_label = (
            "ARIMA"
            if model_type == "arima"
            else "ARIMAX"
        )

        ax.set_title(
            f"{model_label} for {self.canton} "
            f"(NRMSE train = {nrmse_train:.3f}, "
            f"NRMSE val = {nrmse_val:.3f}, "
            f"NRMSE test = {nrmse_test:.3f})"
        )

        ax.grid(
            alpha=0.2
        )

        fig.tight_layout()

        if save_path is not None:
            os.makedirs(
                os.path.dirname(save_path) or ".",
                exist_ok=True
            )

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

        return (
            fig,
            ax,
            predictions_df,
            lower_test,
            upper_test
        )

## LSTM

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

        n_iter = 50

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
        
        X_seq_combined = np.vstack([X_train_seq, X_val_seq])
        y_seq_combined = np.concatenate([y_train_seq, y_val_seq])

        refit_model = keras.Sequential([
            keras.layers.Input(shape=(X_seq_combined.shape[1], X_seq_combined.shape[2])),
            keras.layers.LSTM(units=best_params["units"]),
            keras.layers.Dropout(rate=best_params["dropout"]),
            keras.layers.Dense(1)
        ])

        refit_model.compile(
            loss="mse", 
            optimizer=keras.optimizers.Adam(learning_rate=best_params["learning_rate"])
        )

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

        lower_test, upper_test = self.prediction_intervals_aci_lstm(
            X_train_seq=X_train_seq,
            y_train_seq=y_train_seq,
            X_test_seq=X_test_seq,
            X_val_seq = X_val_seq,
            y_val_seq = y_val_seq,
            y_test_seq=y_test_seq,
            epsilon=epsilon,
            scaler_y=scaler_y,
            anchor_test=test_anchor,
            dif=dif
        )

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

        ax.fill_between(x = test_dates, label = "CI (95%)", y1 = lower_test, y2 = upper_test, alpha = 0.2, color = "#009E73")

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

    def shap_lstm_individual(
        self,
        dif=False,
        split="test",
        max_background=100,
        max_explain=200,
        seed=42,
        figsize=(10, 7)
    ):
        """
        Calculate and plot SHAP variable importance for the
        individual LSTM model corresponding to self.canton.

        SHAP values are calculated for the complete LSTM sequences
        and then aggregated over the temporal dimension so that
        each predictor receives one global importance value.

        Parameters
        ----------
        dif : bool
            Whether to use the differenced LSTM model.

        split : {"train", "val", "test"}
            Dataset split on which to calculate SHAP values.

        max_background : int
            Maximum number of observations used as SHAP background.

        max_explain : int
            Maximum number of observations explained.

        seed : int
            Random seed used when subsampling observations.

        figsize : tuple
            Figure size.

        Returns
        -------
        shap_values : np.ndarray
            SHAP values with shape:
            (n_observations, sequence_length, n_features)

        importance : pd.DataFrame
            Variable importance sorted from most to least important.

        fig, ax : matplotlib objects
            SHAP variable-importance plot.
        """

        import shap

        rng = np.random.default_rng(seed)

        # ---------------------------------------------------------
        # 1. Load individual canton model
        # ---------------------------------------------------------
        file_suffix = "_dif" if dif else ""

        artifact = joblib.load(
            f"../saved_models/lstm/lstm_{self.canton}{file_suffix}.joblib"
        )

        model = artifact["model"]
        scaler_X = artifact["scaler_X"]
        sequence_length = artifact["seq_len"]

        # ---------------------------------------------------------
        # 2. Recreate the exact data used by the model
        # ---------------------------------------------------------
        (
            X_train_seq,
            y_train_seq,
            X_val_seq,
            y_val_seq,
            X_test_seq,
            y_test_seq,
            X_train_scaled,
            y_train_scaled,
            X_val_scaled,
            y_val_scaled,
            X_test_scaled,
            y_test_scaled,
            scaler_X_partition,
            scaler_y,
            train_dates,
            val_dates,
            test_dates,
            test,
            epsilon,
            anchors
        ) = self.partition_lstm(
            sequence_length=sequence_length,
            dif=dif
        )

        split_data = {
            "train": X_train_seq,
            "val": X_val_seq,
            "test": X_test_seq
        }

        if split not in split_data:
            raise ValueError(
                "split must be one of: 'train', 'val', or 'test'."
            )

        X = split_data[split]

        if len(X) == 0:
            raise ValueError(
                f"No sequences available for split='{split}'."
            )

        # ---------------------------------------------------------
        # 3. Recover feature names
        # ---------------------------------------------------------
        df_temp = self.df.copy()

        df_temp = self.drop_lags(
            i=1,
            f=9,
            df_temp=df_temp
        )

        for i in range(1, 9):
            df_temp.drop(
                columns=[f"rr_lag_{i}"],
                inplace=True
            )

        columns_to_drop = [
            "clasi_rr",
            "casos",
            "clasi_rr_no_0"
        ]

        for i in range(1, 9):
            columns_to_drop.extend([
                f"casos_lag_{i}",
                f"clasi_rr_lag_{i}",
                f"clasi_rr_no_0_lag_{i}"
            ])

        columns_to_drop = [
            col for col in columns_to_drop
            if col in df_temp.columns
        ]

        df_temp.drop(
            columns=columns_to_drop,
            inplace=True
        )

        df_temp = pd.get_dummies(
            df_temp,
            columns=["urb"]
        )

        df_temp["rr_log"] = np.log(
            df_temp["rr"] + epsilon
        )

        df_temp["rr_log_raw"] = df_temp["rr_log"]

        if dif:
            df_temp["rr_log"] = (
                df_temp["rr_log"]
                .diff()
                .fillna(0)
            )

            df_temp = (
                df_temp
                .iloc[1:]
                .reset_index(drop=True)
            )

        parts = df_temp["week_canton"].str.split(
            "-",
            expand=True
        )

        df_temp["year"] = parts[0].astype(int)
        df_temp["week"] = parts[1].astype(int)

        feature_columns = [
            col
            for col in df_temp.columns
            if col not in [
                "rr",
                "rr_log_raw",
                "week_canton",
                "year",
                "week"
            ]
        ] 

        # Make sure the feature count agrees with the model.
        if len(feature_columns) != X.shape[2]:
            raise ValueError(
                f"Feature mismatch for canton {self.canton}: "
                f"model expects {X.shape[2]} features, "
                f"but {len(feature_columns)} feature names were recovered."
            )

        # ---------------------------------------------------------
        # 4. Subsample background and observations to explain
        # ---------------------------------------------------------
        n_background = min(max_background, len(X))
        n_explain = min(max_explain, len(X))

        background_idx = rng.choice(
            len(X),
            size=n_background,
            replace=False
        )

        explain_idx = rng.choice(
            len(X),
            size=n_explain,
            replace=False
        )

        background = X[background_idx]
        X_explain = X[explain_idx]

        # ---------------------------------------------------------
        # 5. Calculate SHAP values
        # ---------------------------------------------------------
        explainer = shap.GradientExplainer(
            model,
            background
        )

        shap_values = explainer.shap_values(
            X_explain
        )

        # SHAP versions differ in whether a single-output model
        # returns an array or a list containing an array.
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)

        # Some SHAP versions may return an extra output dimension.
        if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]

        if shap_values.ndim != 3:
            raise ValueError(
                f"Unexpected SHAP shape: {shap_values.shape}. "
                "Expected (samples, sequence_length, features)."
            )

        # ---------------------------------------------------------
        # 6. Aggregate SHAP values over time
        # ---------------------------------------------------------
        # First take absolute SHAP values, then average over:
        #   1. observations
        #   2. sequence/time steps
        #
        # This produces one importance score per variable.
        mean_abs_shap = np.mean(
            np.abs(shap_values),
            axis=(0, 1)
        )

        importance = pd.DataFrame({
            "variable": feature_columns,
            "mean_abs_shap": mean_abs_shap
        })

        importance = (
            importance
            .sort_values(
                "mean_abs_shap",
                ascending=False
            )
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------
        # 7. Plot variable importance
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)

        plot_data = importance.iloc[::-1]

        ax.barh(
            plot_data["variable"],
            plot_data["mean_abs_shap"]
        )

        ax.set_xlabel(
            "Mean absolute SHAP value",
            fontweight="bold",
            fontsize=14
        )

        ax.set_ylabel(
            "Variable",
            fontweight="bold",
            fontsize=14
        )

        ax.set_title(
            f"SHAP variable importance — LSTM "
            f"canton {self.canton} ({split})",
            fontsize=16,
            fontweight="bold"
        )

        ax.tick_params(
            axis="y",
            labelsize=10
        )

        ax.grid(
            axis="x",
            alpha=0.25
        )

        fig.tight_layout()

        plt.show()

        return shap_values, importance, fig, ax

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

        df = self.drop_lags(i = 1, f = 9, df_temp = df)
        
        for i in range(1, 9):
            df.drop(columns = [f"rr_lag_{i}"], inplace = True)
        
        df.drop(columns=['clasi_rr'], inplace=True)
        df.drop(columns = ["casos"], inplace = True)
        df.drop(columns = ["clasi_rr_no_0"], inplace = True)

        df.drop(columns = ["casos_lag_1"], inplace = True)
        df.drop(columns = ["clasi_rr_lag_1"], inplace = True)
        df.drop(columns = ["clasi_rr_no_0_lag_1"], inplace = True)
        df.drop(columns = ["casos_lag_2"], inplace = True)
        df.drop(columns = ["clasi_rr_lag_2"], inplace = True)
        df.drop(columns = ["clasi_rr_no_0_lag_2"], inplace = True)
        df.drop(columns = ["casos_lag_3"], inplace = True)
        df.drop(columns = ["clasi_rr_lag_3"], inplace = True)
        df.drop(columns = ["clasi_rr_no_0_lag_3"], inplace = True)
        df.drop(columns = ["casos_lag_4"], inplace = True)
        df.drop(columns = ["clasi_rr_lag_4"], inplace = True)
        df.drop(columns = ["clasi_rr_no_0_lag_4"], inplace = True)
        df.drop(columns = ["casos_lag_5"], inplace = True)
        df.drop(columns = ["clasi_rr_lag_5"], inplace = True)
        df.drop(columns = ["clasi_rr_no_0_lag_5"], inplace = True)
        df.drop(columns = ["casos_lag_6"], inplace = True)
        df.drop(columns = ["clasi_rr_lag_6"], inplace = True)
        df.drop(columns = ["clasi_rr_no_0_lag_6"], inplace = True)
        df.drop(columns = ["casos_lag_7"], inplace = True)
        df.drop(columns = ["clasi_rr_lag_7"], inplace = True)
        df.drop(columns = ["clasi_rr_no_0_lag_7"], inplace = True)
        df.drop(columns = ["casos_lag_8"], inplace = True)
        df.drop(columns = ["clasi_rr_lag_8"], inplace = True)
        df.drop(columns = ["clasi_rr_no_0_lag_8"], inplace = True)

        df = pd.get_dummies(df, columns = ["urb"])

        if self.canton != 'full':
            for level in [0, 1, 2]:
                col = f"urb_{level}"
                if col not in df.columns:
                    df[col] = 0

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

        df["rr_log"] = np.log(df["rr"] + epsilon)

        excluded = {
            "week_canton",
            "year",
            "week",
            "canton_id",
            "week_index",
            "rr",
            "rr_log_raw",
            "target_log",
            "date",
            "ds",
            "casos",
            "clasi_rr_no_0",
            "clasi_rr"
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

            # end identifies the target observation.
            # The input window contains only the preceding observations.
            for end in range(sequence_length, len(positions)):

                start = end - sequence_length

                window_positions = positions[start:end]
                target_position = positions[end]

                window_indices = week_indices[start:end]

                # Require consecutive weeks within the input window.
                if not np.all(np.diff(window_indices) == 1):
                    continue

                # Require the target to immediately follow the input window.
                if week_indices[end] - window_indices[-1] != 1:
                    continue

                target_year = years[end]

                # Assign split according to the target observation's year.
                if 2020 <= target_year <= 2023:
                    split = "train"
                elif target_year == 2024:
                    split = "val"
                elif target_year == 2025:
                    split = "test"
                else:
                    continue

                # Input: historical predictors only, excluding target week.
                split_data[split]["X"].append(
                    X_scaled[window_positions]
                )

                # Target: log-transformed relative risk for the target week.
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
        n_iter=50,
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
            keras.layers.LSTM(units=best_params["units"]),
            keras.layers.Dropout(rate=best_params["dropout"]),
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
    
            lower_test, upper_test = self.prediction_intervals_aci_lstm_full(seed = 42)

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

                if split == "test":
                    ax.fill_between(x = pred_df["time"], label = "CI (95%)", y1 = lower_test, y2 = upper_test, alpha = 0.2, color = "#009E73")
    
            # ---------------------------------------------------------
            # 3. Format plot
            # ---------------------------------------------------------
            ax.set_title(
                f"Pooled LSTM: Dengue Relative Risk ΓÇö Municipality {self.canton} (NRMSE train = {nrmse_split[0]:.3f}, NRMSE val = {nrmse_split[1]:.3f}, NRMSE test = {nrmse_split[2]:.3f})"
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

    def lstm_eval_plots(self):

        artifact = joblib.load(f'../saved_models/lstm/lstm_{self.canton}.joblib')

        history = artifact["history"]

        plt.figure(figsize=(12, 4))

        # Gr├ífico de P├⌐rdida (Loss)
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Entrenamiento', color='blue', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validaci├│n', color='orange', linewidth=2)
        plt.title('Curva de P├⌐rdida (Loss)')
        plt.xlabel('├ëpocas')
        plt.ylabel('P├⌐rdida')
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

    def shap_lstm_full(
        self,
        split="test",
        max_background=100,
        max_explain=500,
        seed=42,
        figsize=(10, 7)
    ):
        """
        Calculate and plot SHAP variable importance for the
        global/pooled LSTM model.

        IMPORTANT:
            This calculates SHAP values for the global model as a whole.
            It does NOT calculate a separate SHAP analysis for each canton.

        Parameters
        ----------
        split : {"train", "val", "test"}
            Dataset split on which to calculate SHAP values.

        max_background : int
            Maximum number of observations used as SHAP background.

        max_explain : int
            Maximum number of observations explained.

        seed : int
            Random seed used when subsampling observations.

        figsize : tuple
            Figure size.

        Returns
        -------
        shap_values : np.ndarray
            SHAP values with shape:
            (n_observations, sequence_length, n_features)

        importance : pd.DataFrame
            Variable importance sorted from most to least important.

        fig, ax : matplotlib objects
            SHAP variable-importance plot.
        """

        import shap

        rng = np.random.default_rng(seed)

        # ---------------------------------------------------------
        # 1. Load global model and metadata
        # ---------------------------------------------------------
        model_dir = "../saved_models/lstm"

        model_path = os.path.join(
            model_dir,
            "lstm_full.keras"
        )

        metadata_path = os.path.join(
            model_dir,
            "lstm_full.joblib"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Global model not found: {model_path}"
            )

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Global model metadata not found: {metadata_path}"
            )

        model = tf.keras.models.load_model(
            model_path
        )

        metadata = joblib.load(
            metadata_path
        )

        sequence_length = metadata["sequence_length"]
        epsilon = metadata["epsilon"]
        scaler_X = metadata["scaler_X"]
        scaler_y = metadata["scaler_y"]
        feature_columns = metadata["feature_columns"]
        medians = metadata["medians"]

        # ---------------------------------------------------------
        # 2. Recreate the exact global data
        # ---------------------------------------------------------
        data = self.partition_lstm_full(
            sequence_length=sequence_length,
            epsilon=epsilon,
            feature_columns=feature_columns,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            medians=medians
        )

        if split not in data["sets"]:
            raise ValueError(
                "split must be one of: 'train', 'val', or 'test'."
            )

        X = data["sets"][split]["X"]

        if len(X) == 0:
            raise ValueError(
                f"No sequences available for split='{split}'."
            )

        # ---------------------------------------------------------
        # 3. Verify feature names
        # ---------------------------------------------------------
        if len(feature_columns) != X.shape[2]:
            raise ValueError(
                f"Feature mismatch in global model: "
                f"model has {X.shape[2]} features, "
                f"but metadata contains {len(feature_columns)}."
            )

        # ---------------------------------------------------------
        # 4. Subsample observations
        # ---------------------------------------------------------
        n_background = min(
            max_background,
            len(X)
        )

        n_explain = min(
            max_explain,
            len(X)
        )

        background_idx = rng.choice(
            len(X),
            size=n_background,
            replace=False
        )

        explain_idx = rng.choice(
            len(X),
            size=n_explain,
            replace=False
        )

        background = X[background_idx]
        X_explain = X[explain_idx]

        # ---------------------------------------------------------
        # 5. Calculate SHAP values
        # ---------------------------------------------------------
        explainer = shap.GradientExplainer(
            model,
            background
        )

        shap_values = explainer.shap_values(
            X_explain
        )

        # Handle SHAP versions where single-output models
        # return [array] rather than array.
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)

        # Handle possible extra output dimension.
        if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]

        if shap_values.ndim != 3:
            raise ValueError(
                f"Unexpected SHAP shape: {shap_values.shape}. "
                "Expected (samples, sequence_length, features)."
            )

        # ---------------------------------------------------------
        # 6. Aggregate over observations and sequence positions
        # ---------------------------------------------------------
        mean_abs_shap = np.mean(
            np.abs(shap_values),
            axis=(0, 1)
        )

        importance = pd.DataFrame({
            "variable": feature_columns,
            "mean_abs_shap": mean_abs_shap
        })

        importance = (
            importance
            .sort_values(
                "mean_abs_shap",
                ascending=False
            )
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------
        # 7. Plot global variable importance
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)

        plot_data = importance.iloc[::-1]

        ax.barh(
            plot_data["variable"],
            plot_data["mean_abs_shap"]
        )

        ax.set_xlabel(
            "Mean absolute SHAP value",
            fontweight="bold",
            fontsize=14
        )

        ax.set_ylabel(
            "Variable",
            fontweight="bold",
            fontsize=14
        )

        ax.set_title(
            f"SHAP variable importance — Global LSTM ({split})",
            fontsize=16,
            fontweight="bold"
        )

        ax.tick_params(
            axis="y",
            labelsize=10
        )

        ax.grid(
            axis="x",
            alpha=0.25
        )

        fig.tight_layout()

        plt.show()

        return shap_values, importance, fig, ax

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

    def prediction_intervals_aci_lstm(
        self,
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq,
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

        self.set_seed(42)

        file_suffix = "_dif" if dif else ""
        artifacts = joblib.load(f"../saved_models/lstm/lstm_{self.canton}{file_suffix}.joblib")
        best_params = artifacts["params"]

        X_calib_seq = np.concatenate(
            [X_train_seq, X_val_seq],
            axis=0
        )

        y_calib_seq = np.concatenate(
            [y_train_seq, y_val_seq],
            axis=0
        )

        def build_lstm_model():
            model = keras.Sequential([
                keras.layers.Input(shape=(X_calib_seq.shape[1], X_calib_seq.shape[2])),
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

        mapie.fit(X_calib_seq, y_calib_seq)

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

    def prediction_intervals_aci_lstm_full(
            self,
            alpha=0.05,
            gamma=0.005,
            n_resamplings=30,
            seed=42
        ):
        """
        ACI prediction intervals for the pooled/global LSTM.

        The global model is calibrated using:
            train + validation = 2020-2024

        The test period is:
            2025

        Test predictions are processed sequentially so that
        adapt_conformal_inference updates the ACI state after
        each observed test point.
        """
        start = time.perf_counter()

        # ---------------------------------------------------------
        # 1. Reproducibility
        # ---------------------------------------------------------
        self.set_seed(seed)

        # ---------------------------------------------------------
        # 2. Load global model metadata
        # ---------------------------------------------------------
        artifacts = joblib.load(
            "../saved_models/lstm/lstm_full.joblib"
        )

        best_params = artifacts["best_params"]

        epsilon = artifacts["epsilon"]
        sequence_length = artifacts["sequence_length"]

        scaler_y = artifacts["scaler_y"]
        scaler_X = artifacts["scaler_X"]
        feature_columns = artifacts["feature_columns"]
        medians = artifacts["medians"]

        # ---------------------------------------------------------
        # 3. Recreate the global sequences
        #
        # partition_lstm_full creates:
        #   train = 2020-2023
        #   val   = 2024
        #   test  = 2025
        # ---------------------------------------------------------
        data = self.partition_lstm_full(
            sequence_length=sequence_length,
            epsilon=epsilon,
            feature_columns=feature_columns,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            medians=medians
        )

        train = data["sets"]["train"]
        val = data["sets"]["val"]
        test = data["sets"]["test"]

        # ---------------------------------------------------------
        # 4. Combine train + validation
        #
        # This mirrors the final global model training:
        # 2020-2023 + 2024 = 2020-2024
        # ---------------------------------------------------------
        X_calib_seq = np.concatenate(
            [train["X"], val["X"]],
            axis=0
        )

        y_calib_seq = np.concatenate(
            [train["y"], val["y"]],
            axis=0
        )

        X_test_seq = test["X"]
        y_test_seq = test["y"]

        # ---------------------------------------------------------
        # 5. Build a fresh global LSTM estimator
        #
        # IMPORTANT:
        # This matches the architecture of the FINAL model
        # created in fit_lstm_full:
        #
        # Input -> LSTM(dropout=...) -> Dense(1)
        #
        # There is NO separate Dropout layer here because the
        # final refit in fit_lstm_full does not have one.
        # ---------------------------------------------------------
        def build_lstm_model():

            model = keras.Sequential([
                keras.layers.Input(
                    shape=(
                        X_calib_seq.shape[1],
                        X_calib_seq.shape[2]
                    )
                ),
                keras.layers.LSTM(
                    units=best_params["units"],
                    dropout=best_params["dropout"]
                ),
                keras.layers.Dense(1),
            ])

            model.compile(
                loss="mse",
                optimizer=keras.optimizers.Adam(
                    learning_rate=best_params["learning_rate"]
                )
            )

            return model

        base_estimator = KerasRegressor(
            model=build_lstm_model,
            epochs=artifacts["best_epochs"],
            batch_size=best_params["batch_size"],
            verbose=0,
            random_state=seed
        )

        # ---------------------------------------------------------
        # 6. Block bootstrap
        # ---------------------------------------------------------
        cv = BlockBootstrap(
            n_resamplings=n_resamplings,
            length=1,
            overlapping=False,
            random_state=seed,
        )

        # ---------------------------------------------------------
        # 7. ACI
        # ---------------------------------------------------------
        mapie = TimeSeriesRegressor(
            estimator=base_estimator,
            method="aci",
            cv=cv,
            conformity_score=AbsoluteConformityScore(sym=True),
            agg_function="mean",
            random_state=seed,
            n_jobs=1
        )

        # ---------------------------------------------------------
        # 8. Fit ACI on 2020-2024
        # ---------------------------------------------------------
        mapie.fit(
            X_calib_seq,
            y_calib_seq
        )

        # ---------------------------------------------------------
        # 9. Sequential 2025 predictions
        # ---------------------------------------------------------
        lower_scaled = []
        upper_scaled = []
        pred_scaled = []


        for i in range(len(X_test_seq)):

            x_i = X_test_seq[i:i + 1]
            y_i = y_test_seq[i:i + 1]

            y_pred_i, interval = mapie.predict(
                x_i,
                confidence_level=1 - alpha,
                ensemble=True,
                optimize_beta=False
            )

            pred_scaled.append(
                y_pred_i[0]
            )

            lower_scaled.append(
                interval[0, 0, 0]
            )

            upper_scaled.append(
                interval[0, 1, 0]
            )

            # Sequential ACI update after observing y_i
            mapie.adapt_conformal_inference(
                x_i,
                y_i,
                gamma=gamma,
                ensemble=True,
                optimize_beta=False
            )

        # ---------------------------------------------------------
        # 10. Convert scaled values back to log(rr + epsilon)
        # ---------------------------------------------------------
        lower_log = scaler_y.inverse_transform(
            np.asarray(lower_scaled).reshape(-1, 1)
        ).ravel()

        upper_log = scaler_y.inverse_transform(
            np.asarray(upper_scaled).reshape(-1, 1)
        ).ravel()

        # ---------------------------------------------------------
        # 11. Back-transform to original RR scale
        # ---------------------------------------------------------
        lower = np.exp(lower_log) - epsilon
        upper = np.exp(upper_log) - epsilon

        end = time.perf_counter()

        print(
            "BlockBootstrap ACI global LSTM execution time: "
            f"{end - start:.2f} seconds"
        )

        return lower, upper
    
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

  
    def moving_block_bootstrap_nrmse(
        self,
        predictions,
        block_size=4,
        n_bootstrap=500,
        epsilon=0.2,
        seed=42
    ):
        """
        Calculate a point NRMSE and 95% moving block bootstrap interval.

        Required columns:
            year, week, rr_true, predicted_rr

        For pooled data, all municipality observations from the same
        week are sampled together.
        """

        df = predictions.copy()

        required = {"year", "week", "rr_true", "predicted_rr"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if df[list(required)].isna().any().any():
            raise ValueError("Predictions contain missing values.")

        df["week_index"] = df["year"].astype(int) * 52 + df["week"].astype(int)

        # Sort chronologically.
        df = df.sort_values(["week_index", "canton_id"] if
                            "canton_id" in df.columns else ["week_index"])

        weeks = np.sort(df["week_index"].unique())
        n_weeks = len(weeks)

        if not 1 <= block_size <= n_weeks:
            raise ValueError(
                f"block_size must be between 1 and {n_weeks} weeks."
            )

        def nrmse(y_true, y_pred):
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            return rmse / (np.mean(y_true) + epsilon)

        y_true = df["rr_true"].to_numpy()
        y_pred = df["predicted_rr"].to_numpy()
        point_estimate = nrmse(y_true, y_pred)

        # Each block contains consecutive weeks.
        blocks = [
            weeks[i:i + block_size]
            for i in range(n_weeks - block_size + 1)
        ]

        rng = np.random.default_rng(seed)
        n_blocks = int(np.ceil(n_weeks / block_size))
        bootstrap_scores = np.empty(n_bootstrap)

        for b in range(n_bootstrap):
            chosen = rng.integers(0, len(blocks), size=n_blocks)
            sampled_weeks = np.concatenate([blocks[i] for i in chosen])
            sampled_weeks = sampled_weeks[:n_weeks]

            # Include every municipality row belonging to each sampled week.
            # If a week is sampled multiple times, include its rows repeatedly.
            sampled_parts = [
                df[df["week_index"] == week]
                for week in sampled_weeks
            ]
            sample = pd.concat(sampled_parts, ignore_index=True)

            bootstrap_scores[b] = nrmse(
                sample["rr_true"].to_numpy(),
                sample["predicted_rr"].to_numpy()
            )

        return {
            "n_weeks": n_weeks,
            "block_size": block_size,
            "point_nrmse": point_estimate,
            "lower_95": np.quantile(bootstrap_scores, 0.025),
            "upper_95": np.quantile(bootstrap_scores, 0.975),
            "bootstrap_scores": bootstrap_scores
        }

    def bootstrap_lstm_individual(
        self,
        epsilon=0.2,
        dif=False,
        block_size=4,
        n_bootstrap=500,
        seed=42
    ):

        artifacts = joblib.load(f"../saved_models/lstm/lstm_{self.canton}.joblib")
        model = artifacts["model"]
        sequence_length = artifacts["seq_len"]

        parts = self.partition_lstm_individual(
            sequence_length=sequence_length,
            epsilon=epsilon,
            dif=dif
        )

        (
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            X_test_seq, y_test_seq,
            X_train_scaled, y_train_scaled,
            X_val_scaled, y_val_scaled,
            X_test_scaled, y_test_scaled,
            scaler_X, scaler_y,
            train_dates, val_dates, test_dates,
            test_df, epsilon, anchors
        ) = parts

        train_anchor, val_anchor, test_anchor = anchors

        # Predict once. Keep the fitted model fixed throughout bootstrapping.
        predictions_scaled = {
            "train": model.predict(X_train_seq, verbose=0).reshape(-1),
            "val": model.predict(X_val_seq, verbose=0).reshape(-1),
            "test": model.predict(X_test_seq, verbose=0).reshape(-1)
        }

        actual_scaled = {
            "train": np.asarray(y_train_seq).reshape(-1),
            "val": np.asarray(y_val_seq).reshape(-1),
            "test": np.asarray(y_test_seq).reshape(-1)
        }

        dates = {
            "train": pd.Series(train_dates).reset_index(drop=True),
            "val": pd.Series(val_dates).reset_index(drop=True),
            "test": pd.Series(test_dates).reset_index(drop=True)
        }

        split_anchors = {
            "train": train_anchor,
            "val": val_anchor,
            "test": test_anchor
        }

        def to_rr(values_scaled, anchor):
            values = scaler_y.inverse_transform(
                np.asarray(values_scaled).reshape(-1, 1)
            ).ravel()

            if dif:
                values = anchor + np.cumsum(values)

            return np.maximum(np.exp(values) - epsilon, 0)

        frames = {}

        for split in ("train", "val", "test"):
            anchor = split_anchors[split]

            y_true = to_rr(actual_scaled[split], anchor)
            y_pred = to_rr(predictions_scaled[split], anchor)

            date_parts = dates[split].str.split("-", expand=True)

            if len(y_true) != len(date_parts):
                raise ValueError(
                    f"{split}: dates and predictions have different lengths."
                )

            frames[split] = pd.DataFrame({
                "year": date_parts[0].astype(int),
                "week": date_parts[1].astype(int),
                "rr_true": y_true,
                "predicted_rr": y_pred
            })

        # Combine train and validation chronologically.
        train_val = pd.concat(
            [frames["train"], frames["val"]],
            ignore_index=True
        ).sort_values(["year", "week"]).reset_index(drop=True)

        results = {}

        results["train_val"] = self.moving_block_bootstrap_nrmse(
            train_val,
            block_size=block_size,
            n_bootstrap=n_bootstrap,
            epsilon=epsilon,
            seed=seed
        )

        results["test"] = self.moving_block_bootstrap_nrmse(
            frames["test"],
            block_size=block_size,
            n_bootstrap=n_bootstrap,
            epsilon=epsilon,
            seed=seed + 1
        )

        summary = pd.DataFrame({
            split: {
                key: value
                for key, value in result.items()
                if key != "bootstrap_scores"
            }
            for split, result in results.items()
        }).T

        return summary, results

    
    def bootstrap_lstm_full(
        self,
        block_size=4,
        n_bootstrap=500,
        epsilon=0.2,
        seed=42
    ):
        """
        Calculate 95% moving block bootstrap intervals for each
        canton's NRMSE on:
            1. Combined train + validation
            2. Test

        The model predictions are kept fixed. Blocks are sampled
        independently within each canton.
        """

        prediction_results = self.predict_nrmse_lstm_full()

        predictions = prediction_results["predictions"]

        train = predictions["train"].copy()
        val = predictions["val"].copy()
        test = predictions["test"].copy()

        train_val = pd.concat(
            [train, val],
            ignore_index=True
        )

        required = {
            "canton_id", "year", "week",
            "rr_true", "predicted_rr"
        }

        for name, data in [
            ("train_val", train_val),
            ("test", test)
        ]:
            missing = required - set(data.columns)
            if missing:
                raise ValueError(
                    f"{name} predictions are missing columns: {missing}"
                )

            if data[list(required)].isna().any().any():
                raise ValueError(
                    f"{name} predictions contain missing values."
                )

        def calculate_nrmse(y_true, y_pred):
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            return rmse / (np.mean(y_true) + epsilon)

        def bootstrap_one_canton(data, rng):
            data = data.sort_values(
                ["year", "week"]
            ).reset_index(drop=True)

            # Require exactly one observation per week for this canton.
            if data.duplicated(["year", "week"]).any():
                raise ValueError(
                    "Duplicate canton-week observations found."
                )

            # The existing data convention uses 52 weeks per year.
            data["week_index"] = (
                data["year"].astype(int) * 52
                + data["week"].astype(int)
            )

            # Do not silently bridge missing weeks.
            week_index = data["week_index"].to_numpy()

            if len(week_index) > 1 and not np.all(
                np.diff(week_index) == 1
            ):
                raise ValueError(
                    "Missing or nonconsecutive weeks detected for "
                    f"canton {data['canton_id'].iloc[0]}. "
                    "Resolve gaps before bootstrapping."
                )

            y_true = data["rr_true"].to_numpy(dtype=float)
            y_pred = data["predicted_rr"].to_numpy(dtype=float)
            n = len(data)

            if block_size < 1 or block_size > n:
                raise ValueError(
                    f"block_size must be between 1 and {n} "
                    f"for canton {data['canton_id'].iloc[0]}."
                )

            point = calculate_nrmse(y_true, y_pred)

            # Overlapping, non-circular moving blocks.
            blocks = [
                np.arange(start, start + block_size)
                for start in range(n - block_size + 1)
            ]

            n_blocks = int(np.ceil(n / block_size))
            scores = np.empty(n_bootstrap)

            for b in range(n_bootstrap):
                chosen = rng.integers(
                    0, len(blocks), size=n_blocks
                )

                indices = np.concatenate(
                    [blocks[j] for j in chosen]
                )[:n]

                scores[b] = calculate_nrmse(
                    y_true[indices],
                    y_pred[indices]
                )

            return {
                "n_weeks": n,
                "point_nrmse": point,
                "lower_95": np.quantile(scores, 0.025),
                "upper_95": np.quantile(scores, 0.975),
                "bootstrap_scores": scores
            }

        rng = np.random.default_rng(seed)

        results = {}

        for period, data in [
            ("train_val", train_val),
            ("test", test)
        ]:
            period_results = {}

            for canton, canton_data in data.groupby(
                "canton_id", sort=True
            ):
                period_results[canton] = bootstrap_one_canton(
                    canton_data,
                    rng
                )

            results[period] = period_results

        # Convenient summary DataFrames, one row per canton.
        summaries = {}

        for period, canton_results in results.items():
            summaries[period] = pd.DataFrame.from_dict(
                {
                    canton: {
                        key: value
                        for key, value in result.items()
                        if key != "bootstrap_scores"
                    }
                    for canton, result in canton_results.items()
                },
                orient="index"
            )
            summaries[period].index.name = "canton_id"

        return summaries, results

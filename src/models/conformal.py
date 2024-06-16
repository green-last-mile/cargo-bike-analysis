from typing import Any, Dict, Union
import multiprocessing as mp

from crepes import ConformalPredictiveSystem
from crepes.extras import margin, DifficultyEstimator, binning
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost
import polars as pl

from src.models.baseline import BaseModelDatasetAPI, wrap_feature_cols
from src.models.dataset import ServiceTimeDataset
from sklearn.model_selection import train_test_split

from src.models.xgboostlss import N_BOOSTS, N_CPU


class XGBoostWrapper:
    def __init__(self, params: Dict = {}) -> None:
        # self._model = xgboost.XGBModel()
        self.booster: xgboost.Booster = None
        self.params = params
        # self.params['objective'] = 'reg:squarederror'

    def fit(self, X, y):
        dtrain = xgboost.DMatrix(data=X, label=y, nthread=N_CPU)
        self.booster = xgboost.train(
            params=self.params, dtrain=dtrain, num_boost_round=N_BOOSTS
        )

    def predict(self, X):
        dtest = xgboost.DMatrix(X)
        return self.booster.predict(dtest)


CONFORMAL_MODEL = {
    "objective": "reg:squarederror",
    # "tweedie_variance_power": 1.9,
    "max_depth": 8,
    "n_estimators": 1600,
    "eta": 0.0244317757684199,
    "subsample": 0.30000000000000004,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.8,
    "min_child_weight": 175.26004564476892,
    "reg_lambda": 1320.385431944243,
    "reg_alpha": 1.0193124665081958,
    "gamma": 0.48854670937122824,
}


class ConformalModel(BaseModelDatasetAPI):
    def __init__(
        self,
        model_params: Union[Dict[str, Any], None] = None,
        name: str = "ConformalXGBoost",
        cps: bool = True,
        feature_cols: list[str] = None,
    ):
        super().__init__(name=name)
        model_params = (
            model_params
            or {
                # "n_estimators": 1000,
                # "max_depth": 2,
                # 'learning_rate': 0.01,
            }
        )
        # model_params['n_estimators'] = model_params.get('n_estimators', N_BOOSTS)

        self._model_prop: xgboost.XGBRegressor = xgboost.XGBRegressor(
            **CONFORMAL_MODEL,
            # n_estimators=N_BOOSTS,
        )
        # self._model_prop  = RandomForestRegressor(**model_params)

        self._model: ConformalPredictiveSystem = ConformalPredictiveSystem()

        self._feature_cols = feature_cols

        self._cps = cps

        self._de_var = DifficultyEstimator()

    @wrap_feature_cols
    def train(
        self,
        dataset: ServiceTimeDataset,
        test_size: float = 0.2,
        random_state: int = 42,
        # num_boost_round = N_BOOSTS,
    ):
        # further split test train into calibration and training
        X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(
            dataset.train_feature_array,
            dataset.train_label_array,
            test_size=test_size,
            random_state=random_state,
        )

        self._model_prop.fit(
            X_prop_train,
            y_prop_train,
            # num_boost_round=num_boost_round,
        )

        y_hat_cal = self._model_prop.predict(X_cal)
        residuals = y_cal - y_hat_cal

        self._de_var.fit(X=X_cal, residuals=residuals, scaler=True)

        # sigmas_cal_var = self._de_var.apply(X_cal)
        bins_cal, self._bin_thresholds = binning(y_hat_cal, bins=20)

        self._model.fit(
            residuals,
            # sigmas_cal_var,
            bins=bins_cal,
        )

    @wrap_feature_cols
    def predict_samples(self, dataset: ServiceTimeDataset, n: int = 1):
        # assert dataset.test_is_grouped is False, "Test dataset must not be grouped"

        rng = np.random.default_rng()

        y_hat_test = self._model_prop.predict(dataset.test_feature_array)
        bins_test = binning(y_hat_test, bins=self._bin_thresholds)
        # sigmas_test = self._de_var.apply(dataset.test_feature_array)

        samples = self._model.predict(
            y_hat=y_hat_test,
            # y=dataset.te,
            return_cpds=True,
            bins=bins_test,
            y_min=0,
            # sigmas=sigmas_test,
        )

        if n == 1:
            return pl.Series(
                name=self._name + "_samples",
                values=np.array([rng.choice(s) for s in samples]),
            )
        return np.array([rng.choice(s, size=n) for s in samples])

    @wrap_feature_cols
    def predicted_quantiles(self, dataset: ServiceTimeDataset, *args, **kwargs):
        # assert dataset.test_is_grouped is False, "Test dataset must not be grouped"
        # dataset.group_test_df()

        test_df = dataset.test_df.clone()
        y_hat_test = self._model_prop.predict(dataset.test_feature_array)
        bins_test = binning(y_hat_test, bins=self._bin_thresholds)

        sigmas_test = self._de_var.apply(dataset.test_feature_array)

        for q in self.quantiles:
            res = self._model.predict(
                y_hat=y_hat_test,
                lower_percentiles=[
                    q * 100,
                ],
                bins=bins_test,
                y_min=0,
                # sigmas=sigmas_test,
            )

            test_df = test_df.with_columns(
                pl.Series(
                    name=self._name + f"_quant_{q}",
                    values=res,
                ),
            )

        return test_df

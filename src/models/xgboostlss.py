from typing import Any, Dict, Union
import multiprocessing as mp

import xgboost as xgb
from xgboostlss.distributions.distribution_utils import DistributionClass
from xgboostlss.distributions import LogNormal, StudentT
from xgboostlss.model import XGBoostLSS
import polars as pl

from src.models.baseline import BaseModelDatasetAPI, wrap_feature_cols
from src.models.dataset import ServiceTimeDataset

N_CPU = mp.cpu_count()

# OPT_PARAMS = {
#     "eta": 0.04745083489010809,
#     "max_depth": 2,
#     "gamma": 0.014864286418162793,
#     "subsample": 0.5428946911410379,
#     "colsample_bytree": 0.6767232365345294,
#     "min_child_weight": 9.02642750608808,
#     "booster": "gbtree",
#     "lambda": 6.16767924247347e-08,
#     "objective": None,
#     "base_score": 0,
#     "num_target": 2,
#     "disable_default_eval_metric": True,
# }

OPT_PARAMS = {
    "eta": 0.0618395139051438,
    "max_depth": 6,
    "gamma": 2.2950047919458975e-08,
    "subsample": 0.7603400886545297,
    "colsample_bytree": 0.8178498395531972,
    "colsample_bylevel": 0.816745121057533,
    "min_child_weight": 478.2395546918263,
    "booster": "gbtree",
    "tree_method": "approx",
    "reg_lambda": 7.149643241922024e-07,
    # "opt_rounds": 45,
}


OPT_SEATTLE_PARAMS = {
    "eta": 0.0053532658068767515,
    "max_depth": 5,
    "gamma": 3.566016799314632e-08,
    "subsample": 0.7149225620234747,
    "colsample_bytree": 0.8803850618728377,
    "colsample_bylevel": 0.9414007733678754,
    "min_child_weight": 46.46291808639897,
    "booster": "gbtree",
    "tree_method": "approx",
    "reg_lambda": 0.0003486594086260876,
    "opt_rounds": 515,
}


OPT_PARAMS_SUB_TAGS = {
    "eta": 0.035917779022831546,
    "max_depth": 4,
    "gamma": 0.0002741193135042281,
    "subsample": 0.76442868028754,
    "colsample_bytree": 0.8239498279057161,
    "colsample_bylevel": 0.8803398155230415,
    "min_child_weight": 487.6037799693987,
    "booster": "gbtree",
    "tree_method": "approx",
    "reg_lambda": 0.005644030899765208,
}

OPT_SEATTLE_PARAMS_SUB_TAGS = {
    "eta": 0.09891020223707622,
    "max_depth": 2,
    "gamma": 0.001200877235925647,
    "subsample": 0.9985738759698634,
    "colsample_bytree": 0.9995962164883357,
    "colsample_bylevel": 0.8216638556397792,
    "min_child_weight": 160.94522508684165,
    "booster": "gbtree",
    "tree_method": "hist",
    "reg_lambda": 9.408130727505625,
    "opt_rounds": 152,
}

N_BOOSTS = 1500


class XGBoostLSSModel(BaseModelDatasetAPI):
    def __init__(
        self,
        xgboost_params: Union[Dict[str, Any], None] = None,
        distribution: DistributionClass = None,
        name: str = "XGBoostLSS",
        feature_cols: list[str] = None,
    ):
        super().__init__(name=name)
        self.xgboost_params = xgboost_params or OPT_PARAMS
        self.distribution = distribution or LogNormal.LogNormal(
            loss_fn="nll",
            stabilization="None",
            response_fn="softplus",
        )

        self._model = XGBoostLSS(
            self.distribution,
        )

        self._feature_cols = feature_cols

    @wrap_feature_cols
    def train(self, dataset: ServiceTimeDataset, num_boost_round: int = N_BOOSTS):
        print(self.name, dataset.feature_cols[:5], dataset.feature_cols[-5:])

        d_train = xgb.DMatrix(
            dataset.train_feature_array, label=dataset.train_label_array, nthread=N_CPU
        )

        self._model.train(
            self.xgboost_params,
            d_train,
            num_boost_round=num_boost_round,
        )

    @wrap_feature_cols
    def predict_samples(self, dataset: ServiceTimeDataset, n: int = 1):
        # assert dataset.test_is_grouped is False, "Test dataset must not be grouped"
        d_test = xgb.DMatrix(dataset.test_feature_array, nthread=N_CPU)
        res = self._model.predict(d_test, pred_type="samples", n_samples=n)
        if n == 1:
            return pl.Series(
                name=self._name + "_samples",
                values=res["y_sample0"],
            )
        else:
            return res.to_numpy()

    @wrap_feature_cols
    def predicted_quantiles(self, dataset: ServiceTimeDataset, *args, **kwargs):
        # assert dataset.test_is_grouped is False, "Test dataset must not be grouped"
        # dataset.group_test_df()

        d_test = xgb.DMatrix(dataset.test_feature_array, nthread=N_CPU)
        res = self._model.predict(
            d_test, pred_type="quantiles", quantiles=self.quantiles
        )
        return dataset.test_df.join(
            pl.from_pandas(res)
            .rename({f"quant_{q}": f"{self._name}_quant_{q}" for q in self.quantiles})
            .with_columns(
                region_id=dataset.test_df["region_id"],
            ),
            on="region_id",
        )

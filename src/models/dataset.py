from copy import deepcopy
from functools import cached_property
from os import PathLike
from typing import Dict, Generator, List, Self, Union
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_pinball_loss, mean_absolute_error
from properscoring import crps_ensemble
from src.osm_tags import build_tag_filter
import polars as pl
from src.config import CargoBikeConfig
from src.models.baseline import BaseModelDatasetAPI
from sklearn.model_selection import KFold, train_test_split


class ServiceTimeDataset:
    def __init__(
        self,
        config: CargoBikeConfig,
        cities: Union[List[str], None] = None,
        feature_cols: List[str] = tuple(map(str, range(50))),
        pred_col: str = "service_time",
        allow_empty: bool = False,
    ) -> None:
        self.config = config
        self._feature_cols = list(feature_cols)
        self._pred_col = pred_col
        if cities is not None:
            self.config.Cities = [
                city for city in self.config.Cities if city.name in cities
            ]

        self.df = (
            self.h3_pl.lazy()
            .join(
                self._load_service_times(),
                on="region_id",
                how="left" if allow_empty else "inner",
            )
            .with_columns(
                pl.col("region_count").fill_null(0),
            )
            .join(self.h3_pl, on="region_id", how="inner")
            .join(self.embedding_df, on="region_id", how="inner")
            .pipe(self._fix_brussels)
            .collect()
        )

        self._train_df = None
        self._test_df = None

        self.test_is_grouped = False

    @property
    def train_df(self) -> pl.DataFrame:
        return self._train_df.clone()

    @property
    def test_df(self) -> pl.DataFrame:
        return self._test_df.clone()

    @test_df.setter
    def test_df(self, df: pl.DataFrame) -> None:
        assert (
            self._test_df["region_id"] == df["region_id"]
        ).all(), "Region IDs must match"
        self._test_df = df

    def set_train_df(self, df: pl.DataFrame, override: bool = False) -> "Self":
        if not override:
            assert (
                self._train_df["region_id"] == df["region_id"]
            ).all(), "Region IDs must match"
        self._train_df = df
        return self

    @property
    def train_feature_array(self) -> np.ndarray:
        return self._train_df.select(self.feature_cols).to_numpy().copy()

    @property
    def train_label_array(self) -> np.ndarray:
        return self._train_df[self.pred_col].to_numpy().copy()

    @property
    def test_feature_array(self) -> np.ndarray:
        return self._test_df.select(self.feature_cols).to_numpy().copy()

    @property
    def feature_cols(self) -> List[str]:
        return self._feature_cols

    @property
    def pred_col(self) -> str:
        return self._pred_col

    def set_feature_cols(self, cols: List[str]) -> "Self":
        self._feature_cols = cols
        return self

    def set_pred_col(self, col: str) -> "Self":
        self._pred_col = col
        return self

    def update_df(self, df: pl.DataFrame) -> "Self":
        self.df = df
        return self

    @cached_property
    def cities(self) -> List[str]:
        return [city.name for city in self.config.Cities]

    @cached_property
    def _h3s(self) -> pl.DataFrame:
        return pd.concat(
            [
                gpd.read_parquet(city.h3_file).assign(city=city.name)
                for city in self.config.Cities
            ],
            axis=0,
        ).query("is_city")

    @cached_property
    def h3_pl(self) -> pl.LazyFrame:
        return pl.from_pandas(self._h3s.reset_index()[["region_id", "city"]]).lazy()

    @cached_property
    def embedding_df(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.config.GeoVex.embedding_file)

    @cached_property
    def h3s(self) -> List[str]:
        return self.h3_pl["region_id"].unique().to_list()

    def _load_service_times(self) -> pl.DataFrame:
        add_cols = [
            "package_num",
            "height",
            "width",
            "depth",
            "has_time_window",
            "executor_capacity_cm3",
        ]
        dtype = {
            "package_num": int,
            "height": float,
            "width": float,
            "depth": float,
            "has_time_window": bool,
            "executor_capacity_cm3": float,
        }

        def _column_fixer(df: pl.LazyFrame) -> pl.LazyFrame:
            for col in add_cols:
                if col not in df.columns:
                    df = df.with_columns(pl.lit(1).cast(dtype[col]).alias(col))
            return df

        return pl.concat(
            pl.scan_parquet(city.file)
            .pipe(_column_fixer)
            .select(
                [
                    "region_id",
                    pl.col(city.service_time_col).cast(float).alias("service_time"),
                    *(pl.col(col).cast(dtype[col]).alias(col) for col in add_cols),
                ]
            )
            for city in self.config.ServiceTime
        ).with_columns(
            pl.col("service_time").log().alias("log_service_time"),
            pl.count().over("region_id").alias("region_count"),
        )

    def _fix_brussels(self, df: pl.DataFrame) -> pl.DataFrame:
        # TODO: Remove this once the data is fixed
        # return df.filter(
        #     pl.when(
        #         pl.col("city").str.contains("Brussels"),
        #     )
        #     .then(pl.col("service_time") > 50)
        #     .otherwise(pl.lit(True))
        # )
        return df

    def add_super_tags(
        self,
    ) -> "Self":
        target_tags = build_tag_filter(self.config)
        target_tag_list = [f"{t}_{st}" for t, sts in target_tags.items() for st in sts]
        super_tags = list(target_tags.keys())

        count_df = (
            pl.concat(
                pl.scan_parquet(city.count_file).with_columns(
                    pl.lit(city.name).alias("city")
                )
                for city in self.config.Cities
            )
            .filter(pl.col("region_id").is_in(self.df["region_id"]))
            .collect()
        )

        target_tag_list = sorted(
            set(count_df.columns).intersection(set(target_tag_list))
        )

        self.df = self.df.join(
            count_df.select(
                *(
                    pl.sum_horizontal(pl.col(f"^{st}_.*$")).alias(f"{st}")
                    for st in super_tags
                ),
                "region_id",
            ),
            on="region_id",
            how="inner",
        )

        return self

    def add_tags(
        self,
    ) -> Self:
        target_tags = build_tag_filter(self.config)
        target_tag_list = [f"{t}_{st}" for t, sts in target_tags.items() for st in sts]

        count_df = (
            pl.concat(
                pl.scan_parquet(city.count_file).with_columns(
                    pl.lit(city.name).alias("city")
                )
                for city in self.config.Cities
            )
            .filter(pl.col("region_id").is_in(self.df["region_id"]))
            .collect()
        )

        target_tag_list = sorted(
            set(count_df.columns).intersection(set(target_tag_list))
        )

        self.df = self.df.join(
            count_df.select(
                *(pl.col(f"{tag}") for tag in target_tag_list),
                "region_id",
            ),
            on="region_id",
            how="inner",
        )

        return self

    def add_van_label(self, col_name: str = "is_van") -> "Self":
        self.df = self.df.with_columns(
            pl.col("city").str.contains("USA").alias(col_name)
        )
        self._feature_cols.append(col_name)
        return self

    def add_cluster_labels(self, cluster_df: Union[PathLike, pl.DataFrame]) -> "Self":
        assert "cluster" not in self.df.columns, "Cluster labels already added"

        cluster_df = (
            pl.read_parquet(cluster_df)
            if isinstance(cluster_df, PathLike)
            else cluster_df
        )

        self.df = self.df.join(cluster_df.drop("city"), on="region_id", how="inner")

        return self

    def _split_test_train(
        self,
        test_hexes: List[str],
        train_hexes: List[str],
        seed: int = 42,
        assertion_override: bool = False,
    ) -> "Self":
        if not assertion_override:
            assert len(set(test_hexes).intersection(set(train_hexes))) == 0

        self._train_df = self.df.filter(pl.col("region_id").is_in(train_hexes)).sample(
            fraction=1, shuffle=True, seed=seed
        )
        self._test_df = self.df.filter(pl.col("region_id").is_in(test_hexes)).sample(
            fraction=1,
            shuffle=True,
            seed=seed,
        )

        self.test_is_grouped = False

        return self

    def split_test_train(
        self,
        train_cities: Union[List[str], None] = None,
        test_cities: Union[List[str], None] = None,
        train_size: float = 0.8,
        min_points_per_hex_train: int = 20,
        min_points_per_hex_test: int = 20,
        normalize_test_set_by_total: bool = False,
        seed: int = 42,
        assertion_override: bool = False,
        whole_city_test: bool = False,
    ) -> "Self":
        if train_cities is None:
            train_cities = self.cities
        if test_cities is None:
            test_cities = self.cities

        # assert (
        #     min_points_per_hex_train <= min_points_per_hex_test
        # ), "min_points_per_hex_train must be smaller than min_points_per_hex_test. Not sure how to handle the alternative case."

        train_df = self.df.filter(
            pl.col("city").is_in(train_cities)
            & (pl.col("region_count") >= min_points_per_hex_train)
        )
        test_df = self.df.filter(
            pl.col("city").is_in(test_cities)
            & (pl.col("region_count") >= min_points_per_hex_test)
        )

        if (not set(test_cities).issubset(set(train_cities))) or whole_city_test:
            # If test cities are a subset of train cities, we can use the same h3s for both
            # This is useful for testing purposes
            print(
                "Test cities are not a subset of train cities. Setting train size to 1.0"
            )
            train_size = 1.0

            train_regions = train_df["region_id"].unique().to_list()
            test_regions = test_df["region_id"].unique().to_list()

        else:
            if normalize_test_set_by_total:
                # we want an 80/20 split of the total number of hexes, not the number satisfy the min_points_per_test_hex_criteria

                total_hexes = train_df["region_id"].n_unique()
                testable_hexes = test_df["region_id"].n_unique()

                test_size = round((total_hexes * (1 - train_size)) / testable_hexes, 3)

            else:
                test_size = 1 - train_size

            train_regions, test_regions = train_test_split(
                test_df["region_id"].unique().to_list(),
                test_size=test_size,
                random_state=seed,
            )

            train_regions = set(train_regions).union(
                set(train_df["region_id"].unique().to_list()).difference(
                    set(test_regions)
                )
            )
            test_regions = set(test_regions).union(
                set(test_df["region_id"].unique().to_list()).difference(
                    set(train_regions)
                )
            )

        return self._split_test_train(
            test_hexes=list(test_regions),
            train_hexes=list(train_regions),
            seed=seed,
            assertion_override=assertion_override,
        )

    def _group_test_df(
        self,
        opps: Union[List[pl.Expr], None] = None,
        quantiles: List[float] = [0.05, 0.5, 0.95],
    ) -> pl.DataFrame:
        if opps is None:
            opps = [
                # pl.col(self.feature_cols).first(),
                pl.col(self.pred_col).mean().alias(f"{self.pred_col}_mean"),
                pl.col(self.pred_col).std().alias(f"{self.pred_col}_std"),
                *(
                    pl.col(self.pred_col)
                    .quantile(q)
                    .alias(f"{self.pred_col}_q{str(q)}")
                    for q in quantiles
                ),
                pl.count(),
                # pl.col("city").first(),
                pl.all().first(),
            ]

        return self.test_df.groupby("region_id").agg(opps)

    def group_test_df(
        self,
        opps: Union[List[pl.Expr], None] = None,
        quantiles: List[float] = [0.05, 0.5, 0.95],
    ) -> "Self":
        self._test_df = self._group_test_df(opps=opps, quantiles=quantiles)
        self.test_is_grouped = True
        return self

    def kfold_split(
        self,
        cities: Union[List[str], None] = None,
        k: int = 5,
        min_points_per_hex: int = 20,
        seed: int = 42,
        split_hex_level: bool = True,
    ) -> Generator["ServiceTimeDataset", None, None]:
        assert k > 1, "k must be greater than 1"

        if cities is None:
            cities = self.cities

        df = self.df.filter(
            (pl.col("region_count") >= min_points_per_hex)
            & pl.col("city").is_in(cities)
        )

        if not split_hex_level:
            folder = KFold(
                n_splits=k,
                shuffle=True,
                random_state=seed,
            )

            for train_idx, test_idx in folder.split(df):
                return_self = self.copy()
                return_self._train_df = df[train_idx].clone()
                return_self._test_df = df[test_idx].clone()
                yield return_self
            
            return

        regions = df["region_id"].unique().to_list()

        folder = KFold(
            n_splits=k,
            shuffle=True,
            random_state=seed,
        )

        for train_idx, test_idx in folder.split(regions):
            train_regions = [regions[i] for i in train_idx]
            test_regions = [regions[i] for i in test_idx]

            yield self.copy()._split_test_train(
                test_hexes=test_regions,
                train_hexes=train_regions,
                seed=seed,
            )

    def copy(
        self,
    ) -> "ServiceTimeDataset":
        return deepcopy(self)

    def set_test_vehicle_type(
        self,
        vehicle_type: str,
    ) -> "Self":
        assert (
            "is_van" in self.train_df.columns
        ), "Must add van label (`is_van`) to train_df before fitting model"

        assert vehicle_type in [
            "van",
            "cargo_bike",
        ], "vehicle_type must be `van` or `cargo_bike`"

        self._test_df = self._test_df.with_columns(
            pl.lit(vehicle_type == "van").alias("is_van")
        )

        return self

    def apply_model(
        self,
        model: "BaseModelDatasetAPI",
        vehicle_type: str = None,
        **kwargs,
    ) -> "Self":
        # fit the model to the test set
        # print(self.)

        model.train(
            dataset=self,
            **kwargs,
        )

        self.test_apply_model(model, vehicle_type=vehicle_type)

    def test_apply_model(
        self,
        model: "BaseModelDatasetAPI",
        vehicle_type: str = None,
    ) -> "Self":
        if vehicle_type is not None:
            self.set_test_vehicle_type(vehicle_type)

        # predict the test set, and add the predictions to the test set
        self._test_df = self._test_df.with_columns(model.predict_samples(self))
        # this is kinda dumb. Not very memory efficient
        q_df = model.predicted_quantiles(self.copy().group_test_df())

        self._test_df = self._test_df.join(
            q_df.select(
                [
                    "region_id",
                    pl.col(f"^{model.name}_.*$"),
                ]
            ),
            on="region_id",
            how="inner",
        )

        return self

    def score_test(
        self,
        models: List["BaseModelDatasetAPI"],
        crps_n: int = 30,
        transform_log: bool = True,
    ) -> pl.DataFrame:
        assert not self.test_is_grouped
        if transform_log and "log" not in self.pred_col:
            transform_log = False

        model_names = [model.name for model in models]

        target_quantiles = models[0].quantiles

        # sort the quantiles
        target_quantiles = sorted(target_quantiles)
        width = target_quantiles[-1] - target_quantiles[0]
        
        grouped_df = self._group_test_df(
            opps=[
                *(
                    pl.col(self.pred_col)
                    .quantile(q)
                    .alias(f"{self.pred_col}_q{str(q)}")
                    for q in target_quantiles
                ),
                *(
                    pl.col(f"{model}_quant_{str(q)}").first()
                    for q in target_quantiles
                    for model in model_names
                ),
                # *pl.col(f"{model}_quant_{str(target_quantiles[-1])}")
                *(
                    pl.col(self.pred_col)
                    .is_between(
                        pl.col(f"{model}_quant_{str(target_quantiles[0])}"),
                        pl.col(f"{model}_quant_{str(target_quantiles[-1])}"),
                    )
                    .mean()
                    .alias(f"{model}_coverage")
                    for model in model_names
                ),
            ]
        )

        results = []
        for model in models:
            samples = model.predict_samples(
                self,
                n=crps_n,
            )

            res_dict = {
                "model": model.name,
                "crps": crps_ensemble(
                    observations=self._scale_column(
                        self.test_df[self.pred_col].to_numpy(),
                        scale=transform_log,
                    ),
                    forecasts=self._scale_column(
                        samples,
                        scale=transform_log,
                    ),
                ).mean(),
                "mean_coverage": grouped_df[f"{model.name}_coverage"].mean(),
                "mean_interval_width": (
                    self._scale_column(
                        grouped_df[f"{model.name}_quant_{str(target_quantiles[-1])}"],
                        scale=transform_log,
                    )
                    - self._scale_column(
                        grouped_df[f"{model.name}_quant_{str(target_quantiles[0])}"],
                        scale=transform_log,
                    )
                ).mean(),
                "MAE_0.5": mean_absolute_error(
                    self._scale_column(
                        self.test_df[self.pred_col].to_numpy(),
                        scale=transform_log,
                    ),
                    self._scale_column(
                        self.test_df[f"{model.name}_quant_0.5"],
                        scale=transform_log,
                    ),
                ),
            }

            for q in target_quantiles:
                res_dict.update(
                    {
                        f"pinball_{q}": mean_pinball_loss(
                            y_true=self._scale_column(
                                self.test_df[self.pred_col].to_numpy(),
                                scale=transform_log,
                            ),
                            y_pred=self._scale_column(
                                self.test_df[f"{model.name}_quant_{str(q)}"].to_numpy(),
                                scale=transform_log,
                            ),
                            alpha=q,
                        ),
                    }
                )

            results.append(res_dict)

        return pl.DataFrame(results)

    @staticmethod
    def _scale_column(col: Union[pl.Series, np.ndarray], scale: bool) -> np.ndarray:
        if not scale:
            return col
        if isinstance(col, pl.Series):
            return col.exp()
        return np.exp(col)

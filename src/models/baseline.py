import numpy as np
from scipy.stats import lognorm
import polars as pl
import pandas as pd
import h3
from tqdm import tqdm
from joblib import Parallel, delayed

# from xgboostlss.distributions.LogNormal import LogNormal
from torch.distributions import LogNormal

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.dataset import ServiceTimeDataset


# make a wrap function that copies the old features, sets the new ones, and then when done, sets the old ones back
def wrap_feature_cols(func):
    def wrapper(self, dataset: "ServiceTimeDataset", *args, **kwargs):
        old_features = dataset.feature_cols.copy()
        if self._feature_cols is not None:
            dataset.set_feature_cols(
                self._feature_cols,
            )
        res = func(self, dataset, *args, **kwargs)
        dataset.set_feature_cols(old_features)
        return res

    return wrapper


class BaseModelDatasetAPI:
    quantiles = [0.05, 0.50, 0.95]

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self):
        return self._name

    def predict_samples(
        self, dataset: "ServiceTimeDataset", n: int = 1, *args, **kwargs
    ):
        raise NotImplementedError("Predict method must be implemented by subclasses")

    def predicted_quantiles(self, dataset: "ServiceTimeDataset", *args, **kwargs):
        raise NotImplementedError("Predict method must be implemented by subclasses")

    def train(self, dataset: "ServiceTimeDataset", *args, **kwargs):
        raise NotImplementedError("Train method must be implemented by subclasses")


class CityWideAverageModelDataset(BaseModelDatasetAPI):
    def __init__(self, name: str = "CityWideAverageModelDataset") -> None:
        super().__init__(name=name)

    def train(self, dataset: "ServiceTimeDataset", *args, **kwargs):
        self._model = dataset.df.group_by("city").agg(
            pl.col(dataset.pred_col).quantile(q).alias(f"{self._name}_quant_{q}")
            for q in self.quantiles
        )

    def predict_samples(
        self, dataset: "ServiceTimeDataset", n: int = 1, *args, **kwargs
    ):
        assert dataset.test_is_grouped is False, "Test dataset must not be grouped"
        tmp_df = dataset.test_df.join(self._model, on="city")

        def log_transform(x):
            if "log" in dataset.pred_col:
                return x
            else:
                return np.log(x)

        # TODO: Introduce rejection sampling to ensure that the samples are within the 0.05 and 0.95 quantiles
        mu = log_transform(tmp_df[f"{self._name}_quant_0.5",].to_numpy())
        sigma = (
            log_transform(tmp_df[f"{self._name}_quant_0.95",])
            - log_transform(tmp_df[f"{self._name}_quant_0.05",])
        ) / 3.92
        samples = lognorm(s=sigma, scale=np.exp(mu)).rvs(size=(sigma.shape[0], n))

        # hacky quick fix
        if "log" in dataset.pred_col:
            samples = np.log(samples)

        if n == 1:
            return pl.Series(
                name=self._name + "_samples",
                values=samples.ravel(),
            )
        else:
            return samples

    def predicted_quantiles(self, dataset: "ServiceTimeDataset", *args, **kwargs):
        # assert dataset.test_is_grouped is False, "Test dataset must not be grouped"
        # dataset.group_test_df()
        return dataset.test_df.join(self._model, on="city")


class KRingModelDataset(BaseModelDatasetAPI):
    def __init__(
        self,
        name: str = "KRingModelDataset",
        k: int = 3,
    ) -> None:
        super().__init__(name=name)
        self.k = k
        self._model = None

    def train(self, dataset: "ServiceTimeDataset", *args, **kwargs):
        pass

    def predict_samples(
        self, dataset: "ServiceTimeDataset", n: int = 1, *args, **kwargs
    ):
        from h3ronpy.arrow import cells_to_string, grid_disk
        import h3

        def log_transform(x):
            if "log" in dataset.pred_col:
                return x
            else:
                return np.log(x)

        nearby_df = (
            dataset.test_df["region_id"]
            .unique()
            .to_frame()
            .with_columns(
                pl.col("region_id")
                .map_elements(
                    lambda x: cells_to_string(
                        grid_disk([h3.str_to_int(x)], self.k, flatten=True)
                    ).tolist()  # noqa: F821
                )
                .alias("neighborhood")
            )
            .explode("neighborhood")
            .filter(pl.col("neighborhood") != pl.col("region_id"))
        )

        self._model = (
            nearby_df.join(
                dataset.df.select(dataset.pred_col, "region_id"),
                left_on="neighborhood",
                right_on="region_id",
            )
            .group_by("region_id")
            .agg(
                pl.col(dataset.pred_col).quantile(q).alias(f"{self._name}_quant_{q}")
                for q in self.quantiles
            )
            .with_columns(
                pl.col(f"{self._name}_quant_{q}").fill_null(
                    pl.col(f"{self._name}_quant_{q}").mean()
                )
                for q in self.quantiles
            )
        )

        tmp_df = dataset.test_df.join(self._model, on="region_id")

        # tmp_df = tmp_df.with_columns(
        #     pl.col(col).fill_null(pl.col(col).mean())
        #     for col in [
        #         f"{self._name}_quant_0.5",
        #         f"{self._name}_quant_0.05",
        #         f"{self._name}_quant_0.95",
        #     ]
        # )

        mu = log_transform(tmp_df[f"{self._name}_quant_0.5",].to_numpy())
        sigma = (
            log_transform(tmp_df[f"{self._name}_quant_0.95",])
            - log_transform(tmp_df[f"{self._name}_quant_0.05",])
        ) / 3.92

        # replace sigma = 0 with mean sigma
        sigma = np.where(sigma == 0, np.mean(sigma), sigma)

        samples = lognorm(s=sigma, scale=np.exp(mu)).rvs(size=(sigma.shape[0], n))

        if "log" in dataset.pred_col:
            samples = np.log(samples)

        if n == 1:
            return pl.Series(
                name=self._name + "_samples",
                values=samples.ravel(),
            )
        else:
            return samples

    def predicted_quantiles(self, dataset: "ServiceTimeDataset", *args, **kwargs):
        # assert dataset.test_is_grouped is False, "Test dataset must not be grouped"
        # dataset.group_test_df()
        return dataset.test_df.join(self._model, on="region_id")


class BaseModel:
    def __init__(self, service_time_df):
        self.validate_dataframe(service_time_df)
        self.service_time_df = service_time_df
        self.service_time_df = self.service_time_df.with_columns(
            [pl.col("service_time").log().alias("service_time_log")]
        )

    @staticmethod
    def validate_dataframe(df):
        required_columns = {
            "service_time": pl.Float64,  # Assuming service_time should be a floating point number
            "h3": pl.Utf8,  # Assuming h3 is a string (hexadecimal)
        }
        for column, expected_type in required_columns.items():
            if column not in df.columns:
                raise ValueError(f"Missing required column: {column}")
            if df[column].dtype != expected_type:
                raise TypeError(
                    f"Column '{column}' must be of type {expected_type}. Got {df[column].dtype} instead."
                )

    def get_h3_quantiles(self):
        # Default implementation: city-wide quantiles
        quantile_df = self.service_time_df.groupby("h3").agg(
            [
                pl.col("service_time_log").quantile(0.05).alias("q05"),
                pl.col("service_time_log").quantile(0.50).alias("q50"),
                pl.col("service_time_log").quantile(0.95).alias("q95"),
            ]
        )
        return quantile_df

    def generate_samples(self, capped=False):
        quantile_df = self.get_h3_quantiles().to_pandas()  # Convert to pandas DataFrame
        sample_data = [
            self._sample_from_row(row, capped) for index, row in quantile_df.iterrows()
        ]

        # Convert the list of dictionaries to a pandas DataFrame
        sample_df = pd.DataFrame(sample_data)
        return sample_df

    def _sample_from_row(self, row, capped=False, min_count=10):
        # Check if the h3_count is below the minimum threshold
        h3_code = row["h3"]
        h3_count = (
            self.service_time_df.filter(pl.col("h3") == h3_code)
            .select("h3_count")[0]
            .item()
        )
        if h3_count < min_count:
            return {"h3": h3_code, "samples": []}  # Skip or handle low-count hexagons

        mean_log = row["q50"]
        std_log = (row["q95"] - row["q05"]) / 3.92

        # Check for valid mean_log and std_log
        if np.isnan(mean_log) or np.isnan(std_log) or mean_log <= 0 or std_log <= 0:
            return {"h3": h3_code, "samples": []}

        # Generate a larger number of samples to account for filtering
        initial_sample_size = h3_count * 10  # Adjust the multiplier as needed
        samples = lognorm(s=std_log, scale=np.exp(mean_log)).rvs(initial_sample_size)

        if capped:
            q05, q95 = np.exp(row["q05"]), np.exp(row["q95"])
            samples = [sample for sample in samples if q05 <= sample <= q95]
            samples = (
                np.random.choice(samples, size=h3_count, replace=False)
                if len(samples) >= h3_count
                else samples
            )

        return {"h3": h3_code, "samples": samples}


class CityWideAverageModel(BaseModel):
    def __init__(self, service_time_df):
        super().__init__(service_time_df)
        # Calculate city-wide quantiles
        self.city_quantiles = self.service_time_df.select(
            [
                pl.col("service_time_log").quantile(0.05).alias("q05"),
                pl.col("service_time_log").quantile(0.50).alias("q50"),
                pl.col("service_time_log").quantile(0.95).alias("q95"),
            ]
        ).to_dict(as_series=False)

    def get_h3_quantiles(self):
        # Get unique h3 values
        unique_h3 = self.service_time_df.select(pl.col("h3")).unique()

        # Create a DataFrame with each unique h3 and the same city-wide quantiles for each
        quantile_df = unique_h3.with_columns(
            [
                pl.lit(self.city_quantiles["q05"][0]).alias("q05"),
                pl.lit(self.city_quantiles["q50"][0]).alias("q50"),
                pl.lit(self.city_quantiles["q95"][0]).alias("q95"),
            ]
        )
        return quantile_df


class KRingModel(BaseModel):
    def __init__(self, service_time_df, k):
        super().__init__(service_time_df)
        self.k = k

    def get_h3_quantiles(self):
        hexagons = self.service_time_df["h3"].unique().to_list()

        def k_ring_average_quantiles(hexagon_id):
            neighboring_hexes = h3.grid_disk(hexagon_id, self.k)
            neighboring_hexes = set(neighboring_hexes) - {
                hexagon_id
            }  # exlude the hexagon itself
            relevant_data = self.service_time_df.filter(
                pl.col("h3").is_in(neighboring_hexes)
            )

            if len(relevant_data) == 0:
                return [hexagon_id, None, None, None]  # or an appropriate default value

            try:
                quantiles = (
                    relevant_data.select(
                        [
                            pl.col("service_time_log").quantile(0.05).alias("q05"),
                            pl.col("service_time_log").quantile(0.50).alias("q50"),
                            pl.col("service_time_log").quantile(0.95).alias("q95"),
                        ]
                    )
                    .to_numpy()
                    .flatten()
                )
                return [hexagon_id] + list(quantiles)
            except Exception as e:
                print(f"Error calculating quantiles for hexagon {hexagon_id}: {e}")
                return [hexagon_id, None, None, None]  # Handle the error gracefully

        quantiles_per_hexagon = [k_ring_average_quantiles(id) for id in tqdm(hexagons)]

        # Create a DataFrame from the results
        quantile_df = pl.DataFrame(quantiles_per_hexagon)
        quantile_df.columns = ["h3", "q05", "q50", "q95"]
        return quantile_df

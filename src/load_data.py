import geopandas as gpd
import pandas as pd
import polars as pl
from pathlib import Path

def load_h3_data(cities, clusters_file_path):
    """
    Loads and processes H3 data for given cities.

    :param cities: List of city configurations.
    :param h3_file_path: Path to H3 data file.
    :param clusters_file_path: Path to clusters data file.
    :return: A DataFrame with processed H3 data.
    """
    h3_df = pd.concat(
        [gpd.read_parquet(city.h3_file).assign(city=city.name) for city in cities],
        axis=0,
    ).query("is_city")

    clusters_df = pd.read_parquet(clusters_file_path).drop("city", axis=1)
    h3_df = h3_df.merge(clusters_df, on="region_id")

    return h3_df

def open_service_time_data(city, service_time_columns=["package_num", "volume"]):
    """
    Opens and processes service time data for a given city.

    :param city: City configuration object.
    :param service_time_columns: List of service time related column names.
    :return: Processed DataFrame with service time data.
    """
    df = pl.read_parquet(city.file)

    # Ensure required columns are present
    for column in service_time_columns:
        if column not in df.columns:
            df = df.with_columns(pl.lit(None).alias(column))

    # Select and cast relevant columns
    region_col = "region_id" if "region_id" in df.columns else "h3"
    return df.select([
        pl.col(region_col).alias("region_id"),
        pl.col(city.service_time_col).cast(float).alias("service_time"),
        pl.col("package_num").cast(int),
        pl.col("volume").cast(float),
    ])

def aggregate_service_time_data(service_time_df, h3_df, constraints=
                                { "city_name_filter": "Brussels",
                                 "service_time_threshold": 50}):
    """
    Aggregates service time data with H3 data and applies city-specific filtering.

    :param service_time_df: DataFrame with service time data.
    :param h3_df: DataFrame with H3 data.
    :param constraints: Dictionary containing city-specific constraints.
    :return: Aggregated and filtered DataFrame.
    """

    # Join with H3 data
    aggregated_df = service_time_df.join(
        pl.DataFrame(h3_df.reset_index()[["region_id", "city", "cluster"]]),
        on="region_id",
        how="inner",
    ).with_columns(
        [pl.col("service_time").log().alias("service_time_log")]
    )

    # Apply city-specific filtering
    aggregated_df = aggregated_df.filter(
        pl.when(pl.col("city").str.contains(constraints["city_name_filter"]))
        .then(pl.col("service_time") > constraints["service_time_threshold"])
        .otherwise(pl.lit(True))
    )

    # Add count over region_id
    aggregated_df = aggregated_df.with_columns(pl.count().over("region_id").alias("h3_count"))

    return aggregated_df


def process_all_city_data(cities, service_time_cities, 
                          clusters_file_path, 
                          service_time_columns = ["package_num", "volume"],
                          constraints=
                                { "city_name_filter": "Brussels",
                                 "service_time_threshold": 50}):
    """
    Processes all city data by loading, merging, and applying constraints.

    :param cities: List of city configurations for H3 data.
    :param service_time_cities: List of city configurations for service time data.
    :param h3_file_path: Path to H3 data file.
    :param clusters_file_path: Path to clusters data file.
    :param constraints: Dictionary containing city-specific constraints for filtering.
    :return: Processed and aggregated DataFrame with city data.
    """
    # Load and process H3 data
    h3_df = load_h3_data(cities, clusters_file_path)

    # Load and process service time data for each city
    service_time_df = pl.concat([
        open_service_time_data(city, service_time_columns) for city in service_time_cities
    ])

    # Aggregate service time data with H3 data and apply constraints
    aggregated_df = aggregate_service_time_data(service_time_df, h3_df, constraints)

    return aggregated_df

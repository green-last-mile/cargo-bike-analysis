import gc
from pathlib import Path
import sys
import json
from typing import Generator, Set, Tuple
import click
import polars as pl
import uuid
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.polars import add_h3  # noqa: E402
from src.config import load_config, CargoBikeConfig, ServiceTimeCity  # noqa: E402

# from src.osm_tags import build_tag_filter  # noqa: E402


FOLDERS = ["almrrc2021-data-evaluation", "almrrc2021-data-training"]


def process_tt_json(
    f: Path, keep_stops: Set[Tuple[str, str, str]]
) -> Generator[Tuple[str, str, str, int], None, None]:
    with open(f, "r") as file:
        data = json.load(file)
        for r_id, from_stop, to_stop in tqdm(keep_stops):
            if r_id in data:
                yield (r_id, from_stop, to_stop, data[r_id][from_stop][to_stop])


def iterate_files(base_dir: Path, file_regex: Path) -> Generator[Path, None, None]:
    for f in FOLDERS:
        p = base_dir / f
        if matches := list(p.glob(file_regex)):
            yield matches[0]


def open_route_df(config: ServiceTimeCity, h3_res: int) -> Path:
    route_dfs = []

    for f in iterate_files(config.almrcc_download_path, "*route_data.json"):
        route_df = pl.from_dicts(
            [
                {
                    "stop_id": k,
                    "route_id": route_id,
                    "station_code": route_data["station_code"],
                    "departure_datetime": route_data["date_YYYY_MM_DD"]
                    + " "
                    + route_data["departure_time_utc"],
                    "executor_capacity_cm3": route_data["executor_capacity_cm3"],
                    **v,
                }
                for route_id, route_data in json.load(open(f, "r")).items()
                for k, v in route_data["stops"].items()
            ]
        )

        route_dfs.append(
            route_df.with_columns(
                [
                    pl.col("departure_datetime")
                    .str.strptime(pl.Datetime)
                    .dt.replace_time_zone("UTC")
                    .dt.convert_time_zone("EST"),
                    pl.lit(f.parents[0].stem).alias("dataset"),
                ]
            )
        )

    return pl.concat(route_dfs).pipe(add_h3, h3_res, "lat", "lng", "region_id")


def add_package_information(route_df: pl.DataFrame, config: ServiceTimeCity):

    package_dfs = []
    for package_file in iterate_files(
        config.almrcc_download_path, "*package_data.json"
    ):
        package_dfs.append(
            pl.from_records(
                [
                    {
                        "route_id": k,
                        "stop_id": s,
                        "package_id": p,
                        "status": p_data.get("scan_status", None),
                        "time_window_start": p_data["time_window"]["start_time_utc"]
                        if isinstance((p_data["time_window"]["start_time_utc"]), str)
                        else None,
                        "time_window_end": p_data["time_window"]["end_time_utc"]
                        if isinstance((p_data["time_window"]["end_time_utc"]), str)
                        else None,
                        "planned_service_time": p_data["planned_service_time_seconds"],
                        "width": p_data["dimensions"]["width_cm"],
                        "depth": p_data["dimensions"]["depth_cm"],
                        "height": p_data["dimensions"]["height_cm"],
                    }
                    for k, k_data in json.load(open(package_file)).items()
                    for s, s_data in k_data.items()
                    for p, p_data in s_data.items()
                ],
                infer_schema_length=None,
            )
            .with_columns(
                [
                    pl.when(pl.col("height") == 0)
                    .then(pl.lit(1))
                    .otherwise(pl.col("height"))
                    .alias("height")
                ]
            )
            .with_columns(
                [
                    pl.col("time_window_start").is_not_null().alias("has_time_window"),
                    (pl.col("width") * pl.col("depth") * pl.col("height")).alias(
                        "volume"
                    ),
                ]
            )
            .group_by(["route_id", "stop_id"])
            .agg(
                [
                    pl.col("package_id").n_unique().alias("package_num"),
                    pl.col("has_time_window").any(),
                    pl.col("planned_service_time").sum(),
                    pl.col("width").mean(),
                    pl.col("depth").mean(),
                    pl.col("height").mean(),
                    pl.col("volume").sum(),
                    (pl.col("status") == "DELIVERED").all().alias("status"),
                ]
            ),
        )

    route_df = route_df.join(
        pl.concat(package_dfs),
        on=["route_id", "stop_id"],
        how="left",
    )
        

    return route_df.with_columns(
        [
            pl.when(pl.col("status").is_null() & (pl.col("type") == "Station"))
            .then(pl.lit(True))
            .otherwise(pl.col("status"))
            .alias("status")
        ]
    )


def add_sequence_data(route_df: pl.DataFrame, config: ServiceTimeCity) -> pl.DataFrame:
    sequences = []
    for f in iterate_files(config.almrcc_download_path, "*sequences.json"):
        sequence_data = json.load(open(f, "r"))
        # drop the actual level from the sequence data
        sequence_data = pl.DataFrame(
            [
                (k, b, o)
                for k, v in sequence_data.items()
                for b, o in v["actual"].items()
            ],
            schema=["route_id", "stop_id", "order"],
        )

        sequences.append(sequence_data)

    route_df = (
        route_df.join(
            pl.concat(sequences),
            on=["route_id", "stop_id"],
            how="left",
        )
        .with_columns(
            pl.col("order").fill_null(0)  # the eval data doesn't have a sequence
        )
        .sort(["route_id", "order"])
    )

    return route_df


def concat_return_to_station(df: pl.DataFrame) -> pl.DataFrame:
    # add on the return back to the station. This shifts the travel time so that the
    # row contains the time to get to the rows stop_id from the previous row
    # and the service time for that stop
    return df.extend(df.select(pl.all().take(0))).with_columns(
        [
            pl.col("stop_id")
            .shift_and_fill(periods=1, fill_value=pl.col("stop_id").last())
            .alias("prior_id"),
        ]
    )


def _add_tt(route_df: pl.DataFrame, tt_df: pl.DataFrame) -> pl.DataFrame:
    return (
        route_df.lazy()
        .join(
            tt_df.lazy(),
            on=["route_id", "stop_id", "prior_id"],
            how="left",
        )
        .collect()
    )


def add_travel_time(route_df: pl.DataFrame, config: ServiceTimeCity) -> pl.DataFrame:
    route_df = route_df.group_by("route_id").map_groups(concat_return_to_station)

    if (config.almrcc_download_path / "all_tt_df.parquet").exists():
        tt_df = pl.read_parquet(config.almrcc_download_path / "all_tt_df.parquet")

        return _add_tt(route_df, tt_df)

    dump_file = config.almrcc_download_path / f"{uuid.uuid4()}_route_df.parquet"
    # sink route df to disk to save memory
    route_df.write_parquet(dump_file)

    keep_stops = {
        rdf["dataset"][0]: set(
            map(
                tuple,
                rdf[["route_id", "stop_id", "prior_id"]].unique().to_numpy().tolist(),
            )
        )
        for rdf in route_df.partition_by("dataset")
    }

    del route_df
    gc.collect()

    tt_dfs = []

    for f in iterate_files(config.almrcc_download_path, "*travel_times.json"):
        tt_df = pl.DataFrame(
            process_tt_json(f, keep_stops[f.parents[0].stem]),
            schema=["route_id", "stop_id", "prior_id", "travel_time"],
        )

        tt_dfs.append(tt_df)

    # save the travel time data to disk
    tt_df = pl.concat(tt_dfs).with_columns(
        [
            pl.col("travel_time").cast(pl.Float32),
        ]
    )

    tt_df.write_parquet(config.almrcc_download_path / "all_tt_df.parquet")

    route_df = _add_tt(pl.scan_parquet(dump_file), tt_df)

    dump_file.unlink()

    return route_df


def add_trip_information(route_df: pl.DataFrame) -> pl.DataFrame:
    return (
        route_df.with_columns(
            [
                (pl.col("travel_time") + pl.col("planned_service_time")).alias(
                    "time_total"
                )
            ]
        )
        .with_columns(
            [pl.col("time_total").cumsum().over("route_id").alias("cumulative_time")]
        )
        .with_columns(
            [
                (
                    pl.col("departure_datetime")
                    + (pl.col("cumulative_time") * 1e6).cast(pl.Duration)
                ).alias("arrival_datetime")
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("arrival_datetime").dt.hour() * 3600)
                    + (pl.col("arrival_datetime").dt.minute() * 60)
                    + pl.col("arrival_datetime").dt.second() * 1
                ).alias("time_of_day")
            ]
        )
    )


def add_city_name(route_df: pl.DataFrame) -> pl.DataFrame:
    city_map = {
        "BO": "Boston, USA",
        "LA": "Los Angeles, USA",
        "SE": "Seattle, USA",
        "AU": "Austin, USA",
        "CH": "Chicago, USA",
    }
    return route_df.with_columns(
        [
            pl.col("station_code")
            .str.slice(offset=1, length=2)
            .map_dict(city_map)
            .alias("city")
        ]
    )


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def _main(config_file: Path) -> None:
    config = load_config(config_file)
    c = CargoBikeConfig.get_service_time_city(
        config,
        "amazon",
    )

    route_df = (
        open_route_df(c, h3_res=config.H3_Options.resolution)
        .pipe(add_package_information, c)
        # drop non-sucessful deliveries
        .pipe(add_sequence_data, c)
        .sort(
            ["route_id", "order"],
        )
        .pipe(add_travel_time, c)
        .pipe(add_trip_information)
        .pipe(add_city_name)
        .drop(
            [
                "cumulative_time",
                "time_total",
                "prior_id",
                "zone_id",
                "departure_datetime",
            ]
        )
        .filter(pl.col("status"))
    )

    # route_df = route_df.filter(pl.col("city").is_in([city.name for city in c.city]))

    route_df.filter(pl.col("type").str.starts_with("D")).write_parquet(c.file)

    route_df.write_parquet(c.file.parent / (c.file.stem + "_w_depot.parquet"))

    print('Finished')


if __name__ == "__main__":
    _main()

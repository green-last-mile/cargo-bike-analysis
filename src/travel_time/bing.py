import os
from typing import Tuple
import requests
import asyncio
import aiohttp
import polars as pl
import json

from tqdm import tqdm


def get_route_distance(points):
    """
    Get the distance of a route using the Bing Maps API.

    The route is split into chunks of 25 points to avoid the  limit of the API.

    Parameters
    ----------
    points : list of tuples of lat, lon
    """

    latlonzipper = list(zip(points, points[1:]))
    distances = []
    while latlonzipper:
        q_pos = list(latlonzipper.pop(0))
        while len(q_pos) < 25 and latlonzipper:
            q_pos.extend([latlonzipper.pop(0)[1]])
        params = {
            f"wp.{i}": ",".join(str(p) for p in point) for i, point in enumerate(q_pos)
        }
        params["key"] = os.environ["BING_KEY"]
        params["routePath"] = "true"
        response = requests.get(
            "https://dev.virtualearth.net/REST/v1/Routes/Driving", params=params
        )
        if response.status_code != 200:
            raise ValueError(f"Error {response.status_code}, {response.text}")
        data = response.json()
        # return data['resourceSets'][0]['resources'][0]['routeLegs']
        distances.extend(
            [
                r["travelDistance"]
                for r in data["resourceSets"][0]["resources"][0]["routeLegs"]
            ]
        )

    return distances



async def get_travel_chunk(
    chunk_df: pl.DataFrame,
    session: aiohttp.ClientSession,
    url: str,
) -> pl.DataFrame:
    request_str = json.dumps(
        {
            "origins": chunk_df["origin"].to_list(),
            "destinations": chunk_df["destination"].to_list(),
            "travelMode": "driving",
            "timeUnit": "second",
        }
    )

    async with session.post(
        url,
        headers={
            "Content-Type": "application/json",
            "Content-Length": f"{len(request_str)}",
        },
        data=request_str,
    ) as resp:
        j = await resp.json()

    try:
        keep_results = list(
            filter(
                lambda x: x["destinationIndex"] == x["originIndex"],
                j["resourceSets"][0]["resources"][0]["results"],
            )
        )
    except Exception as e:
        print(j)
        raise e


    return pl.DataFrame(keep_results).with_columns(
        pl.Series(chunk_df["row_nr"]).alias("row_nr")
    )


async def get_travel_matrix(
    df: pl.DataFrame,
    api_key: str,
    chunk_size: int = 50,
    origin_lat_col: str = "lat",
    origin_lng_col: str = "lng",
    destination_lat_col: str = "last_lat",
    destination_lng_col: str = "last_lng",
    additional_partition_cols: Tuple[str] = ("city", ),
) -> pl.DataFrame:
    df = df.with_row_count().with_columns(
        pl.struct(
            pl.col(destination_lat_col).alias("latitude"),
            pl.col(destination_lng_col).alias("longitude"),
        ).alias("destination"),
        pl.struct(
            pl.col(origin_lat_col).alias("latitude"),
            pl.col(origin_lng_col).alias("longitude"),
        ).alias("origin"),
        (
                    (pl.col("row_nr").cumcount() // chunk_size)
                    .over(additional_partition_cols)
                    .alias("chunk_id")
                ) if additional_partition_cols else (pl.col("row_nr").cumcount() // chunk_size).alias("chunk_id"),
    )

    dfs = []
    tasks = []
    connector = aiohttp.TCPConnector(limit_per_host=5)  # I have no idea what the Bing API limit is. 5 works
    async with aiohttp.ClientSession(connector=connector) as session:
        for chunk_df in tqdm(df.partition_by(["chunk_id", *additional_partition_cols])):
            url = f"https://dev.virtualearth.net/REST/v1/Routes/DistanceMatrix?key={api_key}"

            tasks.append(asyncio.create_task(get_travel_chunk(chunk_df, session, url)))

        dfs = await asyncio.gather(*tasks)

    return df.join(
        pl.concat(dfs),
        on="row_nr",
        how="left",
    ).drop("chunk_id")
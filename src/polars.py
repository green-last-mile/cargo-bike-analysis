import polars as pl
from h3ronpy.polars import cells_to_string
from h3ronpy.polars.vector import coordinates_to_cells


def add_h3(
    df: pl.DataFrame,
    resolution: int,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    h3_col: str = "h3",
) -> pl.DataFrame:
    """
    Add a h3 column to a DataFrame.
    """

    # make sure that the lat and lon columns are floats
    return df.with_columns(
        [
            pl.col(lat_col).cast(pl.Float64).alias(lat_col),
            pl.col(lon_col).cast(pl.Float64).alias(lon_col),
        ]
    ).pipe(
        lambda df: df.with_columns(
            pl.Series(
                cells_to_string(
                    coordinates_to_cells(df[lat_col], df[lon_col], resolution)
                )
            ).alias(h3_col)
        )
    )

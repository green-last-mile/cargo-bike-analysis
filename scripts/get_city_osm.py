import sys
from pathlib import Path
from typing import List, Union

import click
import geopandas as gpd
from srai.joiners import IntersectionJoiner
from srai.loaders import OSMPbfLoader
from srai.embedders import CountEmbedder
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf
from srai.loaders.osm_loaders.filters import OsmTagsFilter
from srai.h3 import ring_buffer_h3_regions_gdf
from shapely import to_geojson, from_geojson

sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config, CargoBikeConfig, City  # noqa: E402
from src.osm_tags import build_tag_filter  # noqa: E402


def pull_city_boundary(city: City, config: CargoBikeConfig) -> None:
    if not city.boundary_file.exists():
        area_gdf = geocode_to_region_gdf(city.osm_name)
    else:
        area_gdf = gpd.read_file(city.boundary_file)
    
    
    regionalizer = H3Regionalizer(resolution=config.H3_Options.resolution)
    base_h3_regions = regionalizer.transform(area_gdf)

    buffered_h3_regions = ring_buffer_h3_regions_gdf(
        base_h3_regions, distance=config.H3_Options.city_buffer_ring
    )
    # buffered_h3_geometry = buffered_h3_regions.unary_union

    # dump the buffered geometry to geojson
    area_gdf.to_file(city.boundary_file, driver="GeoJSON")

    # dump the hexagons, mark as buffer or city
    buffered_h3_regions = (
        buffered_h3_regions
        # .drop("geometry", axis=1)
        .assign(
            is_city=False,
        )
    )
    buffered_h3_regions.loc[base_h3_regions.index, "is_city"] = True
    buffered_h3_regions.to_parquet(city.h3_file)


def pull_city_data(
    city: City, config: CargoBikeConfig, tag_filter: OsmTagsFilter
) -> None:
    # load the regions
    buffered_regions = gpd.read_parquet(city.h3_file).drop("is_city", axis=1)

    # check if the pbf file exists
    pbf_file = None
    if city.pbf_files.exists():
        files_ = list(city.pbf_files.glob("*.pbf"))
        if len(files_) == 1:
            pbf_file = files_[0]

    tags_df = OSMPbfLoader(pbf_file=pbf_file, download_directory=city.pbf_files).load(
        area=buffered_regions.geometry.unary_union,
        tags=tag_filter,
    )

    joiner = IntersectionJoiner()
    joint_gdf = joiner.transform(buffered_regions, tags_df)
    # create the counts table
    count_df = (CountEmbedder(
        expected_output_features=[
            f"{t}_{st}" for t, sts in tag_filter.items() for st in sts
        ]
    ).transform(
        regions_gdf=buffered_regions,
        features_gdf=tags_df,
        joint_gdf=joint_gdf,
    ))
    # save data
    count_df.to_parquet(city.count_file)


def _process_city(
    city: City, config: CargoBikeConfig, tag_filter: OsmTagsFilter
) -> None:
    if not city.data_folder.exists():
        city.data_folder.mkdir(parents=True)

    pull_city_boundary(city, config)
    pull_city_data(city, config, tag_filter)


def _main(config_file: Path, cities: Union[List[str], None]) -> None:
    config = load_config(config_file)

    tag_filter = build_tag_filter(config)

    for city in config.Cities:
        if cities is not None and city.name.split(",")[0].lower() not in cities:
            continue

        if not city.data_folder.exists():
            city.data_folder.mkdir(parents=True)

        _process_city(
            city=city,
            config=config,
            tag_filter=tag_filter,
        )


@click.command()
@click.argument("config_file", type=Path)
@click.option(
    "--cities",
    type=str,
    default=None,
    help="Comma separated list of cities to process",
    multiple=True,
)
def main(config_file: Path, cities: Union[List[str], None]) -> None:
    _main(config_file, cities)


if __name__ == "__main__":
    # run the main function
    main()

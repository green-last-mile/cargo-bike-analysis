## MAPPING TO CATEGORIES GENERATED USING ChatGPT4, using the following prompt:

# for all these tags, generate a number of high level descriptors that describe the function or type of urban make up to describe h3 hexagons using their associated osm tags.
# This will be used to then do analyses to compare and give an intuition behind the different urban regions.
# In a second step, we used chatGPT4 to generate regex rules to map the OSM tags to the generated high-level categories.


# INITIAL OUTPUT:
# Built Environment:
#     Buildings (residential, commercial, office, public, etc.)
#     Infrastructure (bridges, roads, railway, etc.)

# Transportation:
#     Roads (motorways, highways, residential roads, paths, etc.)
#     Public transport (bus stops, tram stops, railway, subway, etc.)
#     Parking (parking spaces, parking entrance, car parks, etc.)

# Natural Elements:
#     Vegetation (trees, grasslands, forests, parks, etc.)
#     Water bodies (streams, rivers, ponds, etc.)

# Amenities:
#     Services (shops, restaurants, cafes, bars, etc.)
#     Public facilities (schools, hospitals, police, fire hydrants, etc.)

# Leisure and Recreation:
#     Sports facilities (sports centers, stadiums, tennis courts, etc.)
#     Public spaces (parks, gardens, playgrounds, etc.)
#     Tourism (viewpoints, artwork, museums, attractions, etc.)

# Barriers and Boundaries:
#     Walls, fences, gates, kerbs, etc.
#     Administrative, political, postal boundaries

# Utilities and Services:
#     Energy (power poles, power lines, transformers, etc.)
#     Waste Management (waste baskets, recycling, waste disposal, etc.)
#     Communication (street lamps, surveillance, telecom offices, etc.)

# Commerce and Industry:
#     Retail spaces (shops, malls, marketplaces, etc.)
#     Industrial spaces (warehouses, factories, construction sites, etc.)

# Historical and Cultural:
#     Historical sites (monuments, memorials, castles, etc.)
#     Cultural spaces (arts centers, theaters, galleries, etc.)

import re
import polars as pl
import pandas as pd
import numpy as np

mapping_rules = {
    # Historical and Cultural
    r"amenity_place_of_worship|(historic_|place_)": "Historical and Cultural",
    # Transportation
    r"amenity_charging_station|amenity_taxi|parking.*|amenity_parking_space|highway_.*|railway_.*|aeroway_.*|public_transport_.*|route_.*": "Transportation",
    # Barriers and Boundaries
    r"man_made_embankment|barrier_.*|boundary_.*": "Barriers and Boundaries",
    # Built Environment
    r"amenity_college|landuse_(residential|construction|railway|military|garages)|building_.*|residential|military_.*|telecom_.*|power_.*|man_made_.*": "Built Environment",
    # Commerce and Industry
    r"landuse_(retail|commercial|industrial)|office_.*|shop_.*|craft_.*": "Commerce and Industry",
    # Leisure and Recreation
    r"landuse_recreation_ground|sport_.*|tourism_.*|leisure_.*": "Leisure and Recreation",
    # Amenities
    r"amenity_.*|emergency_.*|healthcare_.*": "Amenities",
    # Natural Elements
    r"water*|natural_|landuse_": "Natural Elements",
    # Default rule
    r".*": "Other",
}


def map_key_to_category(key):
    # If the key ends with '.area', it should be excluded from the mapping
    if key.endswith(".area"):
        return "Other"

    for pattern, category in mapping_rules.items():
        if re.match(pattern, key):
            return category
    return "Other"

def map_tags_to_categories(df):
    """
    TODO: rewrite to using polars.
    Maps OpenStreetMap tags to predefined categories and aggregates them by H3 hexagonal cells.

    Args:
        df (pandas.DataFrame): The input DataFrame containing OpenStreetMap tags as columns and H3 hexagonal cells as rows.

    Returns:
        pandas.DataFrame: The resulting DataFrame containing the aggregated counts of each category for each H3 hexagonal cell.
    """

    # Create a dictionary for the mapping, but exclude 'region_id' and 'city' from the mapping process
    columns_to_map = [col for col in df.columns if col not in ["h3", "city"]]
    mapping = {column: map_key_to_category(column) for column in columns_to_map}

    # Find out which columns map to "Other"
    other_tags = [key for key, value in mapping.items() if value == "Other"]
    print("Tags mapped to 'Other':", other_tags)

    # Rename columns based on the mapping
    df = df.rename(columns=mapping)

    # Identify unmatched columns. Exclude 'region_id', 'city', and 'Other' from the unmatched columns check
    unmatched_columns = (
        set(df.columns) - set(mapping.values()) - {"h3", "city", "Other"}
    )
    print("Unmatched columns:", unmatched_columns)

    # Drop unmatched columns and "Other" column
    # df = df.drop(columns=list(unmatched_columns) + ["Other", "city"])

    # Aggregate columns based on high-level categories
    df = df.T.groupby(by=df.columns).sum().T

    return df


def merge_pop_with_categories(pop_gdf, categories_df, clusters_df):
    """
    Merge population data with categories and set missing clusters to -1.

    Args:
    - pop_gdf (GeoDataFrame): Population GeoDataFrame with an 'h3' column.
    - categories_df (DataFrame): Categories DataFrame with an 'h3' column.
    - clusters_df (DataFrame): Clusters DataFrame with an 'h3' column.

    Returns:
    - GeoDataFrame: Merged GeoDataFrame with population, categories, and clusters.
    """
    merged_categories = categories_df.merge(clusters_df, on=["h3", "city"], how="left").assign(
        cluster=lambda df: df["cluster"].fillna(-1).astype(int)
    )

    return pop_gdf.merge(merged_categories, on=["h3", "city"], how="left")



def find_cluster_centroids(df):
    # Group by city and cluster
    grouped = df.groupby(['city', 'cluster'])

    centroids = []

    for (city, cluster), group in grouped:
        # Calculate the centroid of the embeddings
        centroid = group.iloc[:, 3:].mean().values  # Assuming embeddings start from the 4th column

        # Calculate the Euclidean distance of each hex's embedding to the centroid
        distances = group.iloc[:, 3:].sub(centroid)
        distances = np.linalg.norm(distances, axis=1)

        # Convert distances to a Pandas Series to use idxmin
        distances_series = pd.Series(distances, index=group.index)

        # Find the index of the closest hex
        closest_hex_idx = distances_series.idxmin()

        # Get the H3 index of the closest hex
        closest_hex_h3 = group.loc[closest_hex_idx, 'region_id']

        centroids.append({'city': city, 'cluster': cluster, 'centroid_h3': closest_hex_h3})

    return pd.DataFrame(centroids)




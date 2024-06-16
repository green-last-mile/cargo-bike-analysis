from pathlib import Path
import geopandas as gpd
import pandas as pd
import polars as pl

from src.config import load_config, CargoBikeConfig
import src.urban_tools as ut

class DataLoader:
    def __init__(self, config_path):
        """Initializes the DataLoader with configuration from config file."""
        self.config = self.load_config(config_path)

    @staticmethod
    def load_config(config_path):
        """Loads configuration from file."""
        return load_config(config_path)

    def load_cities(self, *cities):
        """Creates a CityData object for the specified cities."""
        return CityData(self, cities)
    
    def _find_city_config(self, city_name):
        """Utility function to find city configuration with flexible matching."""
        city_name_lower = city_name.lower()
        for city in self.config['Cities']:
            if city_name_lower in city['name'].lower():
                return city
        return None

    def _load_h3_data(self, city_name):
        """Loads H3 data for a given city using Polars with flexible city name matching."""
        city_config = self._find_city_config(city_name)
        if city_config:
            h3_df = pl.read_parquet(city_config['h3_file']).with_columns(pl.lit(city_config['name']).alias("city"))
            return h3_df.filter(pl.col("is_city"))
        return pl.DataFrame()

    def _load_osm_tag_count(self, city_name):
        """Loads OSM tag count data for a given city using Polars with flexible city name matching."""
        city_config = self._find_city_config(city_name)
        if city_config:
            count_df = pl.read_parquet(city_config['count_file'])
            return count_df
        else:
            print(f"OSM tag count data not found for {city_name}.")
            return pl.DataFrame()

    def _load_population_data(self, city_name):
        """Loads population data for a given city using Polars, excluding the geometry column, with flexible city name matching."""
        city_config = self._find_city_config(city_name)
        if city_config:
            pop_df = pl.read_parquet(city_config['pop_file'])
            if 'geometry' in pop_df.columns:
                pop_df = pop_df.drop('geometry')
            return pop_df
        else:
            print(f"Population data not found for {city_name}.")
            return pl.DataFrame()

    def _get_osm_category_tags(self, city_name):
        """Maps OSM tags to categories for a given city and returns a Polars DataFrame."""
        # Load the OSM tag count data using the existing method
        osm_tag_count_df = self._load_osm_tag_count(city_name)

        # Check if the DataFrame is not empty
        if not osm_tag_count_df.is_empty():
            # Convert to Pandas DataFrame for the utility function
            osm_tag_count_pd = osm_tag_count_df.to_pandas()

            # Assuming ut.map_tags_to_categories is your utility function that operates on Pandas DataFrame
            # and returns a new DataFrame with tags mapped to categories
            osm_category_pd = ut.map_tags_to_categories(osm_tag_count_pd)

            # Convert the processed Pandas DataFrame back to Polars if needed
            osm_category_df = pl.from_pandas(osm_category_pd)

            return osm_category_df

        else:
            print(f"No OSM tag count data available for {city_name}.")
            return pl.DataFrame()  # Return an empty DataFrame if no data is available

        
    def _load_clusters_data(self):
        """Loads cluster data."""
        return pl.read_parquet(self.config['GeoVex']['cluster_file'])

    def _load_embeddings(self):
        """Loads embeddings data using Polars."""
        embeddings_file = self.config['GeoVex']['embedding_file']
        embeddings_df = pl.read_parquet(embeddings_file)
        return embeddings_df

    COLUMN_GROUPS = {
        'parking_distance': ['parking_distance'],
        'package_size': ['package_num', 'volume'],
        'amazon_features': [
            'executor_capacity_cm3', 'package_num', 'has_time_window',
            'width', 'depth', 'height', 'volume', 'travel_time'
        ],
        'other_amazon': [
            'station_code', 'lat', 'lng', 'type', 'dataset', 'region_id',
            'status', 'order', 'arrival_datetime', 'time_of_day', 'city'
        ],
        'other_cargo_bike': ['sequence', 'rider']
    }

    # Define known types for columns
    dtype = {
        "region_id": pl.Int64,
        "package_num": pl.Int64,
        "height": pl.Float64,
        "width": pl.Float64,
        "depth": pl.Float64,
        "has_time_window": pl.Boolean,
        "executor_capacity_cm3": pl.Float64,
        "parking_distance": pl.Float64,
        "travel_time": pl.Float64,
        "volume": pl.Float64,
        # Add data types for other known columns here
        # "sequence": pl.Int64,  # Example for other types
        # "rider": pl.Utf8,     # Example for other types
    }

    def _load_service_time_data(self, city_name, **column_groups):
        service_time_dfs = []
        all_columns_ordered = ['region_id', 'service_time']  # Ensure 'city' is included

        # Dynamically construct the list of all columns to include
        for group, include in column_groups.items():
            if include:
                all_columns_ordered.extend(self.COLUMN_GROUPS[group])

        # Ensure uniqueness and preserve order
        all_columns_ordered = list(dict.fromkeys(all_columns_ordered))

        # Iterate through each city configuration to find matching city
        for city_config in self.config['ServiceTime']:
            city_list = [city.name for city in city_config['city']]
            city_name = self._find_city_config(city_name).name
            
            if city_name in city_list:  # Check if the city_name is listed under the city field
                print(f"Loading service time data for {city_name}.")
                df = pl.read_parquet(city_config['file'])
                service_time_col = city_config['service_time_col']
                selected_columns = []

                # Dynamically add specified columns based on column_groups, with correct casting
                for col in all_columns_ordered:
                    if col == 'region_id':
                        col_expr = pl.col("region_id" if "region_id" in df.columns else "h3").alias("region_id")
                    elif col == 'service_time':
                        col_expr = pl.col(service_time_col).cast(pl.Float64).alias("service_time")
                    elif col in df.columns:
                        col_expr = pl.col(col).cast(self.dtype[col])
                    else:
                        # For missing columns, add them with NA values of the correct type
                        col_expr = pl.lit(None, dtype=self.dtype.get(col, pl.Float64)).alias(col)
                    selected_columns.append(col_expr)

                # Adding city name as a column to the DataFrame
                selected_columns.append(pl.lit(city_name).alias("city"))

                # Ensure the DataFrame includes all specified columns, in order
                df_selected = df.select(selected_columns)
                service_time_dfs.append(df_selected)

        # Concatenate all DataFrames, now uniformly structured
        combined_service_time_df = pl.concat(service_time_dfs) if service_time_dfs else pl.DataFrame()

        return combined_service_time_df
    

    def _merge_data_frames(self, data_frames, merge_key="region_id"):
        merged_df = data_frames[0]
        for df in data_frames[1:]:
            merged_df = merged_df.join(df, on=merge_key, how='left')
        return merged_df

    def _load_city_data(self, city, *columns):
        """Loads and merges specified data for a given city."""
        data_frames = []

        if "service time" in columns:
            # Assuming a method to load service time data already exists
            service_time_df = self._load_service_time_data(city)
            # You might need additional logic here to filter or merge embeddings based on city
            data_frames.append(service_time_df)

        if "embeddings" in columns:
            # Assuming all cities use the same embeddings file
            embeddings_df = self._load_embeddings()
            # You might need additional logic here to filter or merge embeddings based on city
            data_frames.append(embeddings_df)

        if "parking distance" in columns:
            # to do
            pass
            

        # Additional conditions for other data types

        # Merge all DataFrames based on a common key, e.g., 'region_id'
        merged_df = self._merge_data_frames(data_frames, merge_key="region_id")
        return merged_df



class CityData:
    def __init__(self, data_loader, cities):
        self.data_loader = data_loader
        self.cities = cities
        self.data = {}

    def get_columns(self, *columns, merge=False):
        """
        Retrieves specified columns of data for the previously selected cities.
        Optionally merges multiple data sources into a unified DataFrame.

        Args:
        - columns (tuple): Column names to retrieve data for.
        - merge (bool): If True, merges data across cities into a single DataFrame.
                        If False, keeps data split by city.

        Returns:
        - If merge is False, returns a dictionary containing the requested data for each city.
        - If merge is True, returns a single DataFrame with all cities' data merged.
        """
        merged_data = []
        for city in self.cities:
            city_data = self.data_loader._load_city_data(city, *columns)
            if merge:
                merged_data.append(city_data)
            else:
                self.data[city] = city_data

        if merge:
            # Merge all cities' data into a single DataFrame
            # Ensure there is a 'city' column to differentiate data from different cities
            return pl.concat(merged_data) if merged_data else pl.DataFrame()
        else:
            return self.data

from pathlib import Path
from typing import Generator, Tuple, List
import ijson.backends.yajl2 as ijson
import polars as pl
from tqdm import tqdm
import re
import uuid

def build_df(objects: List[Tuple[str, dict]]) -> pl.DataFrame:

    df = pl.DataFrame(objects)

    # save to a unique file
    df.write_parquet(f"/Users/max/Development/green-last-mile/cargo-bike-analysis/data/tmp/almrrc2021-data-training/travel_times_{uuid.uuid4()}.parquet")


def get_route_id_count(path: str) -> int:
    with open(path, "r") as f:
        # count occurances of using regex
        regex = re.compile('RouteID')

        total = len(list(regex.finditer(f.read())))

    return total


def split_json(path: Path) -> Generator[List[Tuple[str, dict]], None, None]:

    with open(path, "r") as f:
        objects = ijson.kvitems(f, "")

        yield from tqdm(objects, total=get_route_id_count(path))


def process_json(path: Path, save_path: Path, chunk_size: int = 100) -> None:
    
    dfs = []

    # find all travel time files in path and subdirectories
    for i, file_ in enumerate(path.glob("**/travel_times.json")):
        print(f"Processing {file_}")

        for i, chunk in enumerate(split_json(file_)):
            dfs.append(chunk)

            if (i > 0) and (i % chunk_size == 0):
                build_df(dfs)
                dfs = []


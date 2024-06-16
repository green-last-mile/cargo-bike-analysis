from dataclasses import dataclass, field
import os
import sys
from pathlib import Path
from typing import Any, List, Optional
from omegaconf import OmegaConf, ValidationError

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
os.environ["CARGO_BIKE_ROOT"] = str(repo_root)


@dataclass
class ServiceTimeCity:
    name: str
    file: Path
    service_time_col: str
    city: List[Any]
    almrcc_download_path: Optional[Path] = None


@dataclass
class Tag:
    name: str
    problem_tags: Optional[List[str]] = field(default_factory=list)


@dataclass
class Tags:
    tag_file: Path
    keep_tags: List[Tag]


@dataclass
class City:
    name: str
    data_folder: Path
    boundary_file: Path
    pbf_files: Path
    h3_file: Path
    count_file: Path
    pop_file: Path
    osm_name: Optional[str] = "${.name}"


@dataclass
class H3_Options:
    resolution: int
    city_buffer_ring: int


@dataclass
class GeoVex:
    radius: int
    embedding_file: Path
    embedding_file_cropped: Path
    cluster_file: Path


@dataclass
class CargoBikeConfig:
    GeoVex: GeoVex
    ServiceTime: List[ServiceTimeCity]
    Cities: List[City]
    Tags: Tags
    H3_Options: H3_Options

    @staticmethod
    def get_service_time_city(config: "CargoBikeConfig", name: str) -> ServiceTimeCity:
        for c in config.ServiceTime:
            if c.name == name:
                return c
        raise ValueError(f"Unknown service time city: {name}")


def load_config(file_path: str) -> CargoBikeConfig:
    """Load config from yaml file."""
    config = OmegaConf.load(file_path)
    schema = OmegaConf.structured(CargoBikeConfig)
    try:
        return OmegaConf.merge(schema, config)
    except ValidationError as e:
        raise ValueError(f"Invalid config: {e}") from e

import sys
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import click


sys.path.append(str(Path(__file__).parent.parent))

from src.config import load_config, CargoBikeConfig, ServiceTimeCity  # noqa: E402
# from src.osm_tags import build_tag_filter  # noqa: E402


DOWNLOADS = [
    "almrrc2021/almrrc2021-data-training/model_build_inputs",
    "almrrc2021/almrrc2021-data-evaluation/model_apply_inputs",
    "almrrc2021/almrrc2021-data-evaluation/model_score_inputs",
]


def get_almrcc(config: ServiceTimeCity):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # list all the objects in the bucket
    resp = s3.list_objects_v2(Bucket="amazon-last-mile-challenges")

    # get the list of objects
    objects = resp["Contents"]

    output_path = config.almrcc_download_path
    output_path.mkdir(parents=True, exist_ok=True)

    # download all files in DOWNLOADS
    for obj in objects:
        if any(obj["Key"].startswith(d) for d in DOWNLOADS) & obj["Key"].endswith(
            ".json"
        ):
            print(f"Downloading {obj['Key']}")
            file_path = output_path / obj["Key"].split("/")[1] / obj["Key"].split("/")[-1]
            
            if not file_path.exists():
                
                file_path.parent.mkdir(parents=True, exist_ok=True)

                s3.download_file(
                    Bucket="amazon-last-mile-challenges",
                    Key=obj["Key"],
                    Filename=file_path,
                )


@click.command()
@click.argument("config_file", type=click.Path(exists=True))
def _main(config_file):
    config = load_config(config_file)
    get_almrcc(CargoBikeConfig.get_service_time_city(config, "amazon"))


if __name__ == "__main__":
    _main()

import json

from srai.loaders.osm_loaders.filters import GroupedOsmTagsFilter
from src.config import CargoBikeConfig


def build_tag_filter(config: CargoBikeConfig) -> GroupedOsmTagsFilter:
    # load the raw tag json
    with open(config.Tags.tag_file, "r") as f:
        tag_json = json.load(f)

    keep_supertags = {t.name: t.problem_tags for t in config.Tags.keep_tags}

    return {
        tag: list(filter(lambda x: x not in keep_supertags[tag], subtags))
        for tag, subtags in tag_json.items()
        if tag in set(keep_supertags.keys())
    }

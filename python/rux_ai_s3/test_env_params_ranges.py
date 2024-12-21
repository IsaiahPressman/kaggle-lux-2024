import json
from pathlib import Path

from luxai_s3.params import env_params_ranges

ENV_PARAMS_RANGES_JSON = (
    Path(__file__).parents[2] / "src" / "data" / "env_params_ranges.json"
)


def test_env_params_ranges() -> None:
    with open(ENV_PARAMS_RANGES_JSON) as f:
        param_ranges = json.load(f)

    assert param_ranges == env_params_ranges

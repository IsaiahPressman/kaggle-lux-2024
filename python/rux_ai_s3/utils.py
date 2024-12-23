from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import yaml
from pydantic import BaseModel

_BaseModelT = TypeVar("_BaseModelT", bound=BaseModel)


def to_json(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, (list, tuple)):
        return [to_json(s) for s in obj]

    if isinstance(obj, dict):
        out = {}
        for k in obj:
            out[k] = to_json(obj[k])

        return out

    return obj


def load_from_yaml(model_cls: type[_BaseModelT], path: Path) -> _BaseModelT:
    with open(path) as f:
        data = yaml.safe_load(f)

    return model_cls(**data)

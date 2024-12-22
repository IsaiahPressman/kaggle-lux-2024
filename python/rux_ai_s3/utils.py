from typing import Any

import numpy as np


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

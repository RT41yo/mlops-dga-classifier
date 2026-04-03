from typing import Any
import numpy as np


class Preprocess(object):
    def __init__(self):
        pass

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        domain = body.get("domain")
        if domain is None:
            raise ValueError("Field 'domain' is required")
        return [domain]

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        if isinstance(data, np.ndarray):
            data = data.tolist()

        if isinstance(data, list) and len(data) == 1:
            value = data[0]
        else:
            value = data

        try:
            value = int(value)
        except Exception:
            pass

        return {"label": value}

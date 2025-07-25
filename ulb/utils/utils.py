from pathlib import Path
import yaml
import json
from types import SimpleNamespace


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d

def get_opt(fpath):
    print('config file: ', fpath)
    suffix = Path(fpath).suffix.lstrip('.')

    with open(fpath, 'r') as file:
        if suffix == "json":
            data = json.load(file)
        elif suffix in ("yaml", "yml"):
            data = yaml.safe_load(file)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    return dict_to_namespace(data)
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class AttrDict(dict):
    """A dict with attribute access and recursive conversion."""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[item] = value
        return value

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _recursive_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _recursive_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path) -> AttrDict:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Config at {path} did not parse as a dict.')
    cfg['_config_path'] = str(path)
    return AttrDict(cfg)


def to_plain_dict(obj):
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain_dict(v) for v in obj]
    return obj


def apply_overrides(cfg: AttrDict, overrides: Optional[Dict[str, Any]] = None) -> AttrDict:
    if not overrides:
        return cfg
    data = copy.deepcopy(to_plain_dict(cfg))
    data = _recursive_update(data, overrides)
    return AttrDict(data)

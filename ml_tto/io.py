"""HDF5 persistence compatible with historical ml-tto measurement files."""

from datetime import datetime
from enum import Enum

import h5py
import numpy as np
import pandas as pd


class H5Saver:
    def dump(self, data: dict, filepath: str) -> None:
        with h5py.File(filepath, "w") as handle:
            for name, value in data.items():
                self._write(handle, str(name), value)

    def load(self, filepath: str) -> dict:
        with h5py.File(filepath, "r") as handle:
            return {name: self._read(value) for name, value in handle.items()}

    def _write(self, parent, name, value):
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, pd.DataFrame):
            group = parent.create_group(name)
            group.attrs["_type"] = "dataframe"
            group.attrs["columns"] = list(value.columns)
            for column in value:
                self._write(group, str(column), value[column].to_numpy())
            self._write(group, "_index", value.index.to_numpy())
        elif isinstance(value, (dict, list, tuple)):
            group = parent.create_group(name)
            group.attrs["_type"] = type(value).__name__
            items = value.items() if isinstance(value, dict) else enumerate(value)
            for key, item in items:
                self._write(group, str(key), item)
        elif value is None:
            parent.create_dataset(name, data="None").attrs["_type"] = "none"
        elif isinstance(value, datetime):
            parent.create_dataset(name, data=value.isoformat()).attrs["_type"] = (
                "datetime"
            )
        elif isinstance(value, np.str_):
            parent.create_dataset(name, data=str(value))
        elif isinstance(value, np.ndarray) and value.dtype.kind in "UO":
            group = parent.create_group(name)
            group.attrs["_type"] = "object_array"
            group.attrs["shape"] = value.shape
            for index, item in enumerate(value.flat):
                self._write(group, str(index), item)
        elif isinstance(value, (str, bytes, int, float, bool, np.ndarray, np.generic)):
            parent.create_dataset(name, data=value)
        else:
            raise TypeError(
                f"Cannot serialize {type(value).__name__} at {parent.name}/{name}"
            )

    def _read(self, node):
        kind = node.attrs.get("_type")
        if isinstance(kind, bytes):
            kind = kind.decode()
        if isinstance(node, h5py.Dataset):
            value = node[()]
            if kind == "none":
                return None
            if isinstance(value, bytes):
                value = value.decode()
            if kind == "datetime":
                return datetime.fromisoformat(value)
            return value

        values = {name: self._read(value) for name, value in node.items()}
        if kind in {"list", "tuple", "object_array"}:
            ordered = [values[str(i)] for i in range(len(values))]
            if kind == "object_array":
                result = np.empty(len(ordered), dtype=object)
                result[:] = ordered
                return result.reshape(tuple(node.attrs["shape"]))
            return tuple(ordered) if kind == "tuple" else ordered
        if kind == "dataframe":
            index = values.pop("_index", None)
            columns = node.attrs.get("columns", list(values))
            return pd.DataFrame(
                {name: list(values[name]) for name in columns}, index=index
            )
        return values

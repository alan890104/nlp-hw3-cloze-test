from typing import Any
import pickle
import os


def dump_pkl(data: Any, path: str, force: bool = False):
    if force:
        if os.path.exists(path):
            os.remove(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as G:
        pickle.dump(data, G)


def load_pkl(path: str) -> Any:
    if os.path.exists(path):
        with open(path, 'rb') as P:
            training_set = pickle.load(P)
            return training_set
    raise FileNotFoundError

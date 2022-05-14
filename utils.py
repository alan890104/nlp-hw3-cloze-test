from typing import Any, List,Iterable,Tuple
import pickle
import os
import csv
import random

def argmax(array: Iterable) -> Tuple[int, Any]:
    '''
    argmax with random initial index
    '''
    index = random.choice(list(range(len(array))))
    value = array[index]
    for i, v in enumerate(array):
        if v > value:
            index, value = i, v
    return index, value

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


def dict_writer(obj: dict, path: str,header:List[str]=['id', 'label']):
    with open(path, 'w',newline='') as csv_out:
        writer = csv.writer(csv_out)
        writer.writerow(header)
        for key, value in obj.items():
            writer.writerow([key, value])

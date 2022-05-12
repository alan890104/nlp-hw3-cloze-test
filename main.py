# Author: Yu-Lun Hsu
# Student ID: 0716235
# HW ID: Hw3
# Due Date: 05/19/2022

import json
import os
import re
import string
from collections import Counter
from glob import glob
from typing import Any, Iterable, List, Tuple

import pandas as pd
import spacy
from nltk import ngrams
from tqdm import tqdm

import utils

# from spacy.lang.en import stop_words

# NLP = spacy.load("en_core_web_sm")
PUNCT_SET = set(string.punctuation)


def argmax(array: Iterable) -> Tuple[int, Any]:
    index, value = 0, array[0]
    for i, v in enumerate(array):
        if v > value:
            index, value = i, v
    return index, value


def interpolation(start_idx: int, article: List[str], ops: List[str], ngram1: Counter, ngram2: Counter, ngram3: Counter, ngram4: Counter) -> int:
    '''
    do interpolation and return (current '_' index , argmax of ops)
    "_" in article will be replace by words
    '''
    idx: int = article.index("_", start_idx)
    weight_arr = [0, 0, 0, 0]
    for i, op in enumerate(ops):
        substring = article[max(0, idx-3):idx]
        substring.append(op.lower())
        substring = tuple(substring)
        weight = 0.001*ngram1[substring[-1:]] + 0.02*ngram2[substring[-2:]] + \
            0.23*ngram3[substring[-3:]] + 1.0*ngram4[substring[-4:]]
        weight_arr[i] = weight

    argmax_i, _ = argmax(weight_arr)
    return idx, argmax_i


def Solve(ngram1: Counter, ngram2: Counter, ngram3: Counter, ngram4: Counter, path: str = "result.csv", debug: bool = True):
    answer_dict = {}
    test_list = glob(os.path.join("./hw3/test", "*.json"))
    EXCEPTION_LIST = "'_ "
    print("- Start Solving")
    with tqdm(total=len(test_list)) as pbar:
        for file in test_list:
            with open(file, 'r') as F:
                question: dict = json.load(F)
                start_idx: int = 0
                for (ques_num, ops) in question["options"].items():
                    article = question["article"].lower()
                    article = re.sub(r'[^\w'+EXCEPTION_LIST+']',
                                     '', article).lower().split()
                    start_idx, argmax_i = interpolation(start_idx, article, ops, ngram1,
                                                        ngram2, ngram3, ngram4)
                    answer_dict[ques_num] = chr(argmax_i+ord('A'))
                    start_idx += 1
            pbar.update(1)
    pd.DataFrame.from_dict({
        "id": answer_dict.keys(),
        "label": answer_dict.values(),
    }).to_csv(
        path, index=False)


def GenNGram(training_set: List[List[str]], n: int) -> Counter:
    '''
    Find n-gram by given training set
    '''
    assert n >= 1, "n should bigger or equal to 1"
    base_path: str = "./hw3/var/{}_gram.pkl".format(n)
    try:
        print("- Use cache of {}-gram".format(n))
        return utils.load_pkl(base_path)
    except:
        print("- Finding {}-gram".format(n))
        result: Counter = Counter()
        for ts in training_set[:-n]:
            ng = ngrams(ts, n)  # , pad_right=True, right_pad_symbol='</s>'
            result += Counter(ng)
        utils.dump_pkl(result, base_path, True)
        return result


def LoadTrainSet() -> List[List[str]]:
    '''
    Load all json files from hw3/train for traning n-gram
    '''
    pkl_path: str = "./hw3/var/train.pkl"
    try:
        print("- Use Cache Training Set")
        return utils.load_pkl(pkl_path)
    except:
        training_set: List[List[str]] = []
        train_list = glob(os.path.join("./hw3/train", "*.json"))
        EXCEPTION_LIST = "' "
        print("- Start Loading Training Set")
        with tqdm(total=len(train_list)) as pbar:
            for file in train_list:
                with open(file, 'r') as F:
                    data: dict = json.load(F)
                    result: str = data["article"]
                    answers: List[str] = [data["answers"][_]
                                          for _ in data["answers"]]
                    options: List[str] = [''] * len(answers)
                    for i, [(_, v), ans] in enumerate(zip(data["options"].items(), answers)):
                        options[i] = v[ord(ans)-ord('A')]
                    for op in options:
                        result = result.replace(" _ ", " {} ".format(op), 1)
                    result = re.sub(r'[^\w'+EXCEPTION_LIST+']', '', result)
                    tokens: List[str] = result.lower().split()
                    training_set.append(tokens)
                pbar.update(1)
        utils.dump_pkl(training_set, pkl_path, True)
        return training_set


'''
https://www.kaggle.com/competitions/nycu-cs-nlp-hw3/discussion

https://github.com/susantabiswas/Word-Prediction-Ngram


'''
if __name__ == "__main__":
    training_set = LoadTrainSet()
    ngm_4 = GenNGram(training_set, 4)
    ngm_3 = GenNGram(training_set, 3)
    ngm_2 = GenNGram(training_set, 2)
    ngm_1 = GenNGram(training_set, 1)
    print(ngm_4.most_common(5))
    print(ngm_3.most_common(5))
    print(ngm_2.most_common(5))
    print(ngm_1.most_common(5))
    Solve(ngm_1, ngm_2, ngm_3, ngm_4, "result.csv")

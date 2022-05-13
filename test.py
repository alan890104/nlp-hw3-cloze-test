import json
import os
import random
import string
from glob import glob
from itertools import combinations
from typing import Any, Dict, Iterable, List, Tuple, Union

from nltk import sent_tokenize, word_tokenize
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from tqdm import tqdm

import utils

PUNCTUATON = set(string.punctuation)


def LoadRawJson() -> List[dict]:
    print("- Start Loading Jsons")
    src = glob(os.path.join("./hw3/train", "*.json"))
    raw: List[dict] = []
    with tqdm(total=len(src)) as pbar:
        for file in src:
            with open(file, 'r') as F:
                data: dict = json.load(F)
                raw.append(data)
            pbar.update(1)
    return raw


def TrainTestSplit(dataset: List[dict], test_size: Union[int, float] = 0.1, seed: int = None) -> Tuple[list, list]:
    random.seed(seed)
    size: int = 0
    if isinstance(test_size, int):
        assert test_size <= len(dataset), "test_size <= len(dataset)"
        size = test_size
    elif isinstance(test_size, float):
        assert test_size < 1.0, "0 <= test_size <= 1"
        size = int(len(dataset)*test_size)
    idxs = list(range(len(dataset)))
    test_set_idx = random.choices(idxs, k=size)
    train_set_idx = list(set(idxs)-set(test_set_idx))

    train_set = [dataset[i] for i in train_set_idx]
    test_set = [dataset[i] for i in test_set_idx]

    return train_set, test_set


def ResolveTrainingSet(dataset: List[dict]) -> List[List[str]]:
    training_set: List[List[str]] = []
    for data in dataset:
        result: str = data["article"]
        answers: List[str] = [data["answers"][_]
                              for _ in data["answers"]]
        options: List[str] = [''] * len(answers)
        for i, [(_, v), ans] in enumerate(zip(data["options"].items(), answers)):
            options[i] = v[ord(ans)-ord('A')]
        for op in options:
            result = result.replace(" _ ", " {} ".format(op), 1)
        training_set.append(result)
    return training_set


def ResolveTestingSet(dataset: List[dict]) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    '''
    Return (list of {articles,options}, answers)
    '''
    testing_set: List[Dict[str, str]] = []
    answers_set: Dict[str, str] = {}
    for data in dataset:
        answers: Dict[str, str] = data["answers"]
        options: List[Dict[str, List[str]]] = data["options"]
        answers_set = {**answers_set, **answers}
        testing_set.append({
            "article": data["article"],
            "options": options
        })
    return testing_set, answers_set


def Tokenizer(contexts: List[str]) -> List[List[str]]:
    '''
    contexts is a list of paragraphs
    '''
    print("- Start Tokenize")
    tknz_texts = []
    with tqdm(total=len(contexts)) as pbar:
        for context in contexts:
            pbar.update(1)
            tokenized_text = [x for x in [word_tokenize(sent)
                                          for sent in sent_tokenize(context.lower())]]
            tknz_texts.extend(tokenized_text)
    return tknz_texts


def Train(n_gram: int, tknz: List[List[str]]) -> MLE:
    print("- Start Training")
    train_data, padded_sents = padded_everygram_pipeline(n_gram, tknz)
    model = MLE(n_gram)
    model.fit(train_data, padded_sents)
    return model


def Prediction(model: MLE, n_gram: int, dataset: List[dict],) -> Dict[str, str]:
    print("- Start Prediction")
    answer_dict: Dict[str, str] = {}
    for question in dataset:
        start_idx: int = 0
        article_token = [x for x in word_tokenize(
            question["article"].lower())]
        for (ques_num, ops) in question["options"].items():
            argmax_i, start_idx = getArgmaxScores(
                model, n_gram, start_idx, article_token, ops)
            answer_dict[ques_num] = chr(argmax_i+ord('A'))
            start_idx += 1
    return answer_dict


def getArgmaxScores(model: MLE, n_gram: int, start_idx: int, article_token: List[Union[str, Any]], ops: List[str]) -> Tuple[int, int]:
    '''
    return (argmax element, new start_idx)
    NOTE: the article_token will be modified
    '''
    idx: int = article_token.index("_", start_idx)
    scores: List[int] = [0] * len(ops)
    for i, op in enumerate(ops):
        subset: List[str] = []
        subset.extend(article_token[max(0, idx-n_gram+1):idx])
        assert len(subset) <= n_gram-1, "lenght of subset({}) needs to be equal to n_gram-1({})".format(
            len(subset), n_gram-1)
        lower_op = op.lower()
        score = model.score(lower_op, subset)
        scores[i] = (score)
    argmax_i, _ = utils.argmax(scores)
    article_token[idx] = ops[argmax_i]
    return argmax_i, idx


def Evaluation(pred: Dict[str, str], actual: Dict[str, str]):
    assert len(pred) == len(actual), "length of two input must be same"
    labels = ord(max(max(pred.values()), max(actual.values())))-ord('A')+1
    metric = [[0]*labels for _ in range(labels)]
    # Compute confusion matrix by metric[actual][pred]
    for k, v in pred.items():
        metric[ord(actual[k])-ord('A')][ord(v)-ord('A')] += 1
    # Internal variables
    correct_elem: List[float] = [metric[i][i] for i in range(labels)]
    actual_elem: List[float] = [sum(metric[i]) for i in range(labels)]
    pred_elem: List[float] = [sum(x) for x in zip(*metric)]
    # Compute accuracy
    # Compute precision (預測A而且正確的/所有預測是A的)
    # Compute recall (預測A而且正確的/所有真正是A的)
    accuracy: float = round(sum(correct_elem)/len(pred), 4)
    precision: List[float] = [round(correct_elem[i]/pred_elem[i], 2) if pred_elem[i] != 0 else "NaN"
                              for i in range(labels)]
    recall: List[float] = [round(correct_elem[i]/actual_elem[i], 2) if actual_elem[i] != 0 else "NaN"
                           for i in range(labels)]
    f1_score: List[float] = [round(2*precision[i]*recall[i]/(precision[i]+recall[i]), 2)
                             if precision[i]+recall[i] != 0 else "NaN" for i in range(labels)]
    print("====================================")
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("Recall    : {}".format(recall))
    print("F1-Score  : {}".format(f1_score))
    print("Confusion Matrix:")
    for i in range(labels+1):
        if i == 0:
            print("{:<7s}".format(''), end='')
        else:
            print("{:<7s}".format(chr(i-1+ord('A'))+'?'), end='')
    print()
    for a in range(labels):
        print("{:<7s}".format(chr(a + ord('A'))), end='')
        for p in range(labels):
            print("{:<7d}".format(metric[a][p]), end='')
        print()
    print("====================================")


def Solve(model: MLE, n_gram: int, path: str = "result.csv"):
    answer_dict = {}
    test_list = glob(os.path.join("./hw3/test", "*.json"))
    print("- Start Solving")
    with tqdm(total=len(test_list)) as pbar:
        for file in test_list:
            with open(file, 'r') as F:
                question: dict = json.load(F)
                start_idx: int = 0
                article_token = [x for x in word_tokenize(
                    question["article"].lower())]
                for (ques_num, ops) in question["options"].items():
                    argmax_i, start_idx = getArgmaxScores(
                        model, n_gram, start_idx, article_token, ops)
                    answer_dict[ques_num] = chr(argmax_i+ord('A'))
                    start_idx += 1
            pbar.update(1)
    utils.dict_writer(answer_dict, path)


if __name__ == "__main__":
    ngram = 3
    dataset = LoadRawJson()
    training_set, testing_set = TrainTestSplit(dataset,test_size=0.3)
    training_set = ResolveTrainingSet(training_set)
    testing_set, actual = ResolveTestingSet(testing_set)
    model = Train(ngram, Tokenizer(training_set))
    preds = Prediction(model,ngram,testing_set)
    Evaluation(preds,actual)

    # model.generate(50, "i am very".split())

    # print(model.score("Nowadays".lower(), "<s> <s>".split()))
    # print(word_tokenize("today is a good day\ni _ to _ namin."))

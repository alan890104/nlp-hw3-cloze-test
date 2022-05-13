from typing import Dict, List


class Dataset:
    article: str
    options: Dict[str, List[str]]
    answers: Dict[str, str]
    source: str

    def __init__(self, article: str, options: Dict[str, List[str]], answers: Dict[str, str], source: str, *args, **kwargs) -> None:
        self.article = article
        self.options = options
        self.answers = answers
        self.source = source
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __str__(self):
        return str({
            "article": self.article,
            "options": self.options,
            "answers": self.answers,
            "source": self.source,
        })

    def question_index()->List[int]:
        pass

    def resolve() -> List[dict]:
        pass

if __name__ == "__main__":
    d = Dataset(article="1331sdf", options={
                "a": [1, 2, 3, 4]}, answers={"a": "a"}, source="sdf")
    print(d.resolve())

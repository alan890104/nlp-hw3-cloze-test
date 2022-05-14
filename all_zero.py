import csv
import spacy
from nltk.lm.preprocessing import padded_everygram_pipeline,everygrams

from nltk.lm import Vocabulary,MLE

from nltk.util import flatten

NLP = spacy.load('en_core_web_sm', disable=["tok2vec", "ner", "textcat"])

text1 = "a b c d e".split()
text2 = "a c d e z".split()

# train_data, padded_sents = padded_everygram_pipeline(3, [text1,text2])
train = list(everygrams(text1))
train.extend(list(everygrams(text2)))
print(train)
m = MLE(3)
m.fit(train,vocabulary_text="abcdez".split())
print(m.vocab.counts)

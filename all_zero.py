import csv
import spacy
from nltk.lm.preprocessing import padded_everygram_pipeline,everygrams

from nltk import DefaultTagger,UnigramTagger,BigramTagger,TrigramTagger
from nltk.lm import Vocabulary,MLE
from nltk.tag.hmm import _market_hmm_example
from nltk.tag import HiddenMarkovModelTagger, TrigramTagger
from nltk.util import flatten

# NLP = spacy.load('en_core_web_sm', disable=["tok2vec", "ner", "textcat"])

# text1 = "a b c d e".split()
# text2 = "a c d e z".split()

# # train_data, padded_sents = padded_everygram_pipeline(3, [text1,text2])
# train = list(everygrams(text1))
# train.extend(list(everygrams(text2)))
# print(train)
# m = MLE(3)
# m.fit(train,vocabulary_text="abcdez".split())
# print(m.vocab.counts)

# model, states, symbols = _market_hmm_example()
# print("Testing", model)

# for test in [
#     ["up", "up"],
#     ["up", "down", "up"],
#     ["down"] * 5,
#     ["unchanged"] * 5 + ["up"],
# ]:

#     sequence = [(t, None) for t in test]

#     print("Testing with state sequence", test)
#     print("probability =", model.probability(sequence))
#     print("tagging =    ", model.tag([word for (word, tag) in sequence]))
#     print("p(tagged) =  ", model.probability(sequence))
#     print("H =          ", model.entropy(sequence))
#     print("H_exh =      ", model._exhaustive_entropy(sequence))
#     print("H(point) =   ", model.point_entropy(sequence))
#     print("H_exh(point)=", model._exhaustive_point_entropy(sequence))
#     print()

tagger = HiddenMarkovModelTagger()
tagger.train()

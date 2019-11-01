import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import re
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix

def wm2df(wm, feat_names):
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,
                      columns=feat_names)
    return(df)


def tokenize(corpus):
    pattern = re.compile(r'\b\w\w+\b')
    return (re.findall(pattern, corpus))


def set_weights(tokens):
    token_counts = defaultdict(int)
    for token in tokens:
        token_counts[token] += 1
    return (token_counts)


def simple_vectorizer(corpora):
    feat_names = []
    doc_counts = []
    matrix_seed = []

    for corpus in corpora:
        tokens = tokenize(corpus)
        doc_count = set_weights(tokens)
        doc_counts.append(doc_count)
        feat_names.extend(doc_count.keys())

    unique_feat_names = list(set(feat_names))

    for doc_count in doc_counts:
        matrix_row = [doc_count.get(feat_name, 0) \
                      for feat_name in unique_feat_names]
        matrix_seed.append(matrix_row)

    matrix = csr_matrix(matrix_seed)
    return (csr_matrix(matrix_seed), unique_feat_names)


import spacy
from html import unescape

spacy.load('en')
lemmatizer = spacy.lang.en.English()

def my_preprocessor(doc):
    return(unescape(doc).lower())

def my_tokenizer(doc):
    tokens = lemmatizer(doc)
    return([token.lemma_ for token in tokens])


class MyAnalyzer(object):

    def __init__(self):
        spacy.load('en')
        self.lemmatizer_ = spacy.lang.en.English()

    def __call__(self, doc):
        doc_clean = unescape(doc).lower()
        tokens = self.lemmatizer_(doc_clean)
        return ([token.lemma_ for token in tokens])

analyzer = MyAnalyzer()

corpora = pd.read_csv("new_sms_train.csv", error_bad_lines=False, delimiter='\t')
custom_vec = CountVectorizer(preprocessor=my_preprocessor, tokenizer=my_tokenizer, analyzer=analyzer, ngram_range=(1,2), stop_words='english')
cwm = custom_vec.fit_transform(corpora["v2"])
tokens = custom_vec.get_feature_names()
pd.set_option('display.max_columns', 30)
print(wm2df(cwm, tokens).iloc[:10,1200:1230])
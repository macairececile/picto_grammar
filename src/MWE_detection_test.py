import pandas as pd

from print_sentences_from_grammar import *
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from disambiguate import *
from create_final_lexicon import *
from nltk.corpus import wordnet as wn
from analysis_results_PE import *
import csv
import spacy
from text_to_num import text2num

import warnings

warnings.filterwarnings("ignore")


def read_sentences(file):
    data = pd.read_csv(file, sep="\t")
    return data["text_process"].tolist()


def load_MWE_model():
    tokenizer = AutoTokenizer.from_pretrained("bvantuan/camembert-mwer")
    model = AutoModelForTokenClassification.from_pretrained("bvantuan/camembert-mwer")
    mwe_classifier = pipeline('token-classification', model=model, tokenizer=tokenizer)
    return mwe_classifier


def test_mwe(mwe_classifier, sentences):
    mwe_all = []
    for s in sentences:
        mwes = mwe_classifier(s)
        out_mwe = mwe_classifier.group_entities(mwes)
        for m in out_mwe:
            mwe_all.append(m["word"])

    f = open("out_mwe.txt", "w")
    for el in sorted(list(set(mwe_all))):
        if len(el.split(" ")) > 1:
            f.write(el + '\n')
    f.close()


def main():
    mwe_model = load_MWE_model()
    sentences = read_sentences(
        "/data/macairec/PhD/Grammaire/corpus/output_grammar/commonvoice/test_commonvoice_grammar.csv")
    test_mwe(mwe_model, sentences)


if __name__ == '__main__':
    main()

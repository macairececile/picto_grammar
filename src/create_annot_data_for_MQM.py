import json
import random
import csv
from generate_html_from_json_ortolang import *

corpus = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative", "ofrom", "reunions", "tcof",
          "tufs", "valibel"]


def read_json_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    ids = [d['audio'] for d in data]
    sentences = [d['sentence'] for d in data]
    pictos = [d['pictos'] for d in data]
    tokens = [d['pictos_tokens'] for d in data]
    return ids, sentences, pictos, tokens


def get_random_sentences(sentences, n):
    random_numbers = [random.randint(0, len(sentences)) for _ in range(n)]
    return random_numbers


def select_data(ids, sentences, pictos, tokens, n):
    random_numbers = get_random_sentences(sentences, n)
    ids_subset = [ids[i] for i in random_numbers]
    sentences_subset = [sentences[i] for i in random_numbers]
    pictos_subset = [pictos[i] for i in random_numbers]
    tokens_subset = [tokens[i] for i in random_numbers]
    return ids_subset, sentences_subset, pictos_subset, tokens_subset


def get_doc_id_and_seg_id(id, corpus_name):
    doc_id = "-".join(id.split("-")[:-1]).split("cefc-" + corpus_name + "-")[1]
    seg_id = id.split("-")[-1]
    return doc_id, seg_id


def select_data_per_corpus(dir, n=20):
    all_selected_ids, all_selected_sentences, all_selected_pictos, all_selected_tokens, all_corpus = [], [], [], [], []
    for c in corpus:
        ids, sentences, pictos, tokens = read_json_file(dir + c + "/" + "corpus_" + c + ".json")
        ids_subset, sentences_subset, pictos_subset, tokens_subset = select_data(ids, sentences, pictos, tokens, n)
        all_selected_ids.extend(ids_subset)
        all_selected_sentences.extend(sentences_subset)
        all_selected_pictos.extend(pictos_subset)
        all_selected_tokens.extend(tokens_subset)
        all_corpus.extend([c] * n)

    with open('MQM_eval.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["system", "doc", "doc_id", "seg_id", "rater", "source", "target"]
        writer.writerow(field)
        for i, j in enumerate(all_selected_ids):
            doc_id, seg_id = get_doc_id_and_seg_id(j, all_corpus[i])
            writer.writerow(["grammar", j, doc_id, seg_id, "rater1", all_selected_sentences[i], all_selected_tokens[i]])

    name_html = "MQM_eval.html"
    html_file(name_html, all_selected_ids, all_selected_sentences, all_selected_pictos, all_selected_tokens)


def main():
    dir = "/data/macairec/PhD/Grammaire/corpus/ortolang/propicto-orfeo/"
    select_data_per_corpus(dir)


if __name__ == '__main__':
    main()
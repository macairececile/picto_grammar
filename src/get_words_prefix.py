import json
import pandas as pd

prefix = ['dés', 'intra', 'inter', 'demi', 'més', 'post', 'semi', 'anti', 'col', 'com', 'con', 'co', 'dis', 'il', 'im', 'in', 'ir', 'mi', 'mé', 'non', 'per', 'pré', 're']


def read_json_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    sentences = [d['sentence'] for d in data]
    return sentences


def read_corpus_orfeo(dir):
    corpus_name = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative", "ofrom", "reunions", "tcof", "tufs", "valibel"]
    all_sentences = []
    for n in corpus_name:
        all_sentences.extend(read_json_file(dir+"/"+n+"/corpus_"+n+".json"))
    return all_sentences


def get_words_prefix(sentences):
    out = {}
    for p in prefix:
        words_with_p = list(set([w for s in sentences for w in s.split() if w.startswith(p)]))
        out[p] = words_with_p
    df = pd.DataFrame.from_dict(out, orient='index').transpose()
    csv_file_path = 'words_prefix.csv'
    df.to_csv(csv_file_path, index=False)


def main(dir):
    sentences = read_corpus_orfeo(dir)
    get_words_prefix(sentences)


if __name__ == '__main__':
    main("/data/macairec/PhD/Grammaire/corpus/ortolang/propicto-orfeo/")
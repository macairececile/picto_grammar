import pandas as pd
import json
import numpy as np


def read_tsv(csv_file):
    """
        Read csv file in a dataframe

        Arguments
        ---------
        csv_file: .csv file

        Returns
        -------
        A dataframe with the data.
    """
    return pd.read_csv(csv_file, sep='\t')


def read_json_file(json_file):
    """
        Read the .json file.

        Arguments
        ---------
        json_file: str

        Returns
        -------
        The list of items in the .json file.
    """
    f = open(json_file)
    data = json.load(f)
    clip_names = [item["audio"] for item in data]
    pictos = [item["pictos"] for item in data]
    tokens = [item["pictos_tokens"] for item in data]
    return clip_names, pictos, tokens


def read_all_json(dir):
    corpus_name = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative",
                   "ofrom", "reunions", "tcof", "tufs", "valibel"]
    clip_names, pictos, tokens = [], [], []

    for c in corpus_name:
        data = read_json_file(dir + c + '/corpus_' + c + '.json')
        clip_names.extend(data[0])
        pictos.extend(data[1])
        tokens.extend(data[2])
    return clip_names, pictos, tokens


def add_info_to_csv(file_name, clip_names, pictos, tokens, csv_data):
    pictos_to_add, tokens_to_add = [], []
    for i, row in csv_data.iterrows():
        if row["clips"] in clip_names:
            pos = clip_names.index(row['clips'])
            pictos_to_add.append(pictos[pos])
            tokens_to_add.append(tokens[pos])
        else:
            pictos_to_add.append(np.nan)
            tokens_to_add.append(np.nan)

    csv_data["tgt"] = tokens_to_add
    csv_data["pictos"] = pictos_to_add
    csv_data_remove_empty_lines = csv_data.dropna()
    csv_data_remove_empty_lines.to_csv(file_name + "_s2p.csv", sep="\t", index=False)


def main(file_name, dir):
    csv_data = read_tsv(file_name)
    clip_names, pictos, tokens = read_all_json(dir)
    add_info_to_csv(file_name.split(".csv")[0], clip_names, pictos, tokens, csv_data)


if __name__ == '__main__':
    main("/data/macairec/PhD/Grammaire/corpus/s2p/test_grammar_clean_cfpb.csv",
         "/data/macairec/PhD/Grammaire/corpus/ortolang/propicto-orfeo/")

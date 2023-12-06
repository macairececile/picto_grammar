import pandas as pd
import ast


def read_wold_data_with_sense_keys(file):
    wolf_data = {}
    with open(file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            columns = line.split("\t")
            if columns[0] not in wolf_data:
                wolf_data[columns[0]] = ast.literal_eval(columns[1][:-1])
            else:
                wolf_data[columns[0]].extend(ast.literal_eval(columns[1][:-1]))

    return wolf_data


def read_lexique(lexicon):
    """
        Read the lexicon.

        Arguments
        ---------
        lexicon: str

        Returns
        -------
        The dataframe with the information.
    """
    df = pd.read_csv(lexicon, sep='\t')
    df.loc[:, 'keyword_no_cat'] = df['lemma'].apply(lambda a: str(a).split(' #')[0].split('?')[0].split('!')[0].strip())
    df["sense_keys"] = df["sense_keys"].apply(lambda x: ast.literal_eval(x))
    df["synsets"] = df["synsets"].apply(lambda x: ast.literal_eval(x))
    return df


def create_lexicon_from_wolf(wolf_data, lexicon):
    lemmas = []
    id_pictos = []
    synsets = []
    sense_keys = []
    for k, v in wolf_data.items():
        for sense in v:
            for i, row in lexicon.iterrows():
                if sense in row["sense_keys"]:
                    lemmas.append(k)
                    id_pictos.append(row["id_picto"])
                    synsets.append(row["synsets"])
                    sense_keys.append(row["sense_keys"])
        if k not in lemmas:
            lemmas.append(k)
            id_pictos.append("")
            synsets.append("")
            sense_keys.append(v)
    data = {'id_picto': id_pictos, 'lemma': id_pictos, 'synsets': synsets, 'sense_keys': sense_keys}
    df = pd.DataFrame(data)
    df.to_csv("wolf_merge_with_lexicon.csv", sep="\t", index=False)


def main(wolf_file, lexicon):
    wolf_data = read_wold_data_with_sense_keys(wolf_file)
    lexicon_data = read_lexique(lexicon)
    create_lexicon_from_wolf(wolf_data, lexicon_data)


if __name__ == '__main__':
    main("/data/macairec/PhD/Grammaire/dico/wolf_data.txt",
         "/data/macairec/PhD/Grammaire/dico/lexique_5_12_2023_11h.csv")

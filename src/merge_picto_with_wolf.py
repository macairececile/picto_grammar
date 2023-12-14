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
    result_data = {'id_picto': [], 'lemma': [], 'synsets': [], 'sense_keys': []}

    for k, v in wolf_data.items():
        found_match = False
        for sense in v:
            matching_rows = lexicon[lexicon['sense_keys'].apply(lambda keys: sense in keys)]

            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                result_data['id_picto'].append(row['id_picto'])
                result_data['lemma'].append(k)
                result_data['synsets'].append(row['synsets'])
                result_data['sense_keys'].append(row['sense_keys'])
                found_match = True
                break

        if not found_match:
            result_data['id_picto'].append('')
            result_data['lemma'].append(k)
            result_data['synsets'].append('')
            result_data['sense_keys'].append(v)

    result_df = pd.DataFrame(result_data)
    result_df.to_csv("wolf_merge_with_lexicon.csv", sep="\t", index=False)


def main(wolf_file, lexicon):
    wolf_data = read_wold_data_with_sense_keys(wolf_file)
    lexicon_data = read_lexique(lexicon)
    create_lexicon_from_wolf(wolf_data, lexicon_data)


if __name__ == '__main__':
    main("/data/macairec/PhD/Grammaire/dico/wolf/wolf_data.txt",
         "/data/macairec/PhD/Grammaire/dico/lexique_5_12_2023_11h.csv")

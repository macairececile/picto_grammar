"""
Script to generate the final lexicon.

Example of use: python create_final_lexicon.py --arasaac_file 'arasaac.fre30bis.csv' --dicoPicto 'dicoPicto.csv'
--wn_file 'index.sense' --out 'lexicon.csv'
"""

import pandas as pd
from argparse import ArgumentParser, RawTextHelpFormatter


def delete_last_digit_and_remove_underscore(lemma):
    """
        Delete specific character in a lexicon.

        Arguments
        ---------
        lemma: str

        Returns
        -------
        The cleaned lemma.
    """
    lem = lemma.split('_')
    if lem[-1].isdigit():
        return ' '.join(lem[:-1]).lower()
    else:
        return ' '.join(lem).lower()


def read_initial_voc_file(file_arasaac):
    """
        Read the lexicon created by Norre et al.

        Arguments
        ---------
        file_arasaac: str

        Returns
        -------
        The data from the lexicon.
    """
    df = pd.read_csv(file_arasaac)
    df.loc[:, 'lemma_2'] = df['lemma'].apply(lambda a: delete_last_digit_and_remove_underscore(str(a)))
    df.loc[:, 'plurals_2'] = df['lemma_plural'].apply(lambda a: delete_last_digit_and_remove_underscore(str(a)))
    return df[['idpicto', 'synset2', 'lemma_2', 'plurals_2']]


def read_dicopicto(dicopicto_file):
    """
        Read the lexicon created by us, which is called dicoPicto.

        Arguments
        ---------
        dicopicto_file: str

        Returns
        -------
        The data from the lexicon.
    """
    data = pd.read_csv(dicopicto_file, sep=',')
    data.loc[:, 'keyword_proc'] = data['keyword'].apply(lambda a: a.lower())
    data.loc[:, 'keyword_no_cat'] = data['keyword'].apply(lambda a: a.split(' #')[0].lower())
    return data


def parse_wn31_file(file):
    """
        Parse the WordNet 3.1 file.

        Arguments
        ---------
        file: str

        Returns
        -------
        A dataframe with the sense keys from WordNet 3.1.
    """
    try:
        data_wn31 = pd.read_csv(file, delimiter=" ", names=["sense_key", "synset", "id1", "id2"], header=None)
        return data_wn31
    except IOError:
        print("Could not read file, wrong file format.", file)
        return


def add_value(id_picto, lemma, synsets, sense_keys, row, df_wn, val):
    """
        Add a picto value if lemma from the init vocab is not defined in dicoPicto.

        Arguments
        ---------
        id_picto: list
        lemma: list
        synsets: list
        sense_keys: list
        row: dataframe
        df_wn: dataframe
        val: str
    """
    index = [i for i, x in enumerate(id_picto) if x == row["idpicto"]]
    added = False
    for i in index:
        if val == lemma[i]:
            synsets[i].append(row["synset2"])
            sense_keys[i] = list(set(sense_keys[i] + get_sense_keys(df_wn, [row["synset2"]])))
            return ""
    if not added:
        id_picto.append(row["idpicto"])
        lemma.append(val)
        synsets.append([row["synset2"]])
        sense_keys.append(get_sense_keys(df_wn, [row["synset2"]]))


def create_common_lexique(data_arasaac, data_dicopicto, df_wn):
    """
        Merge the two lexicons.

        Arguments
        ---------
        data_arasaac: dataframe
        data_dicopicto: dataframe
        df_wn: dataframe

        Returns
        ---------
        The lists with id_picto, lemma, synsets, and sense_keys
    """
    id_picto = []
    lemma = []
    synsets = []
    sense_keys = []
    id_picto.extend(data_dicopicto["id_picto"].tolist())
    lemma.extend(data_dicopicto["keyword_no_cat"].tolist())
    synsets.extend(data_dicopicto["synsets"].tolist())
    sense_keys.extend(data_dicopicto["sense_keys"].tolist())
    for i, row in data_arasaac.iterrows():
        if row["lemma_2"] not in data_dicopicto["keyword_no_cat"].values:
            add_value(id_picto, lemma, synsets, sense_keys, row, df_wn, row["lemma_2"])
        elif row["synset2"] != '\\N':
            if row["synset2"] not in data_dicopicto["synsets"].explode().tolist() and row["lemma_2"] not in \
                    data_dicopicto["keyword_no_cat"].values:
                add_value(id_picto, lemma, synsets, sense_keys, row, df_wn, row["lemma_2"])
        if row["plurals_2"] != '\\n':
            if row["plurals_2"] not in data_dicopicto["keyword_no_cat"].values and row["plurals_2"] not in lemma:
                add_value(id_picto, lemma, synsets, sense_keys, row, df_wn, row["plurals_2"])
    return id_picto, lemma, synsets, sense_keys


def get_pos(synset):
    """
        Get the part-of-speech of a synset.

        Arguments
        ---------
        synset: str

        Returns
        ---------
        The tag associated to the POS.
    """
    pos = synset.split('-')[1]
    tag = 0
    if pos == 'n':
        tag = 1
    elif pos == 'v':
        tag = 2
    elif pos == 'a':
        tag = 3
    elif pos == 'r':
        tag = 4
    elif pos == 's':
        tag = 5
    return tag


def get_pos_synset(tag):
    """
        Get the part-of-speech from the tag.

        Arguments
        ---------
        tag: int

        Returns
        ---------
        The POS.
    """
    pos = 0
    if tag == 1:
        pos = '-n'
    elif tag == 2:
        pos = '-v'
    elif tag == 3:
        pos = '-a'
    elif tag == 4:
        pos = '-r'
    elif tag == 5:
        pos = '-s'
    return pos


def get_sense_keys(wn_data, synsets):
    """
        Get the sense keys associated to the synset id.

        Arguments
        ---------
        wn_data: dataframe
        synsets: list

        Returns
        ---------
        The sense keys.
    """
    sense_keys = []
    if synsets:
        for s in synsets:
            if s not in ['\\N', "None", "closed"]:
                synset = s.split('-')[0]
                tag = get_pos(s)
                all_sense_keys = list(set(wn_data.loc[wn_data['synset'] == int(synset)]["sense_key"].tolist()))
                if all_sense_keys:
                    for sense in all_sense_keys:
                        sense_tag = sense.split('%')[1][0]
                        if int(sense_tag) == tag:
                            sense_keys.append(sense)
    return sense_keys


def add_synset_to_dicoPicto(df_arasaac, df_dicopicto, df_wn):
    """
        Add the synset to dicoPicto.

        Arguments
        ---------
        df_arasaac: dataframe
        df_dicopicto: dataframe
        df_wn: dataframe
    """
    synsets = []
    sense_keys = []
    for index, row in df_dicopicto.iterrows():
        synset = list(set(df_arasaac.loc[df_arasaac['idpicto'] == int(row['id_picto'])]["synset2"].tolist()))
        synsets.append(synset)
        senses = get_sense_keys(df_wn, synset)
        sense_keys.append(senses)
    df_dicopicto['synsets'] = synsets
    df_dicopicto['sense_keys'] = sense_keys


def create_final_lexicon(file_arasaac, dicopicto_file, wn_file, outfile):
    """
        Get the final lexicon by merging the two lexicons.

        Arguments
        ---------
        file_arasaac: str
        dicopicto_file: str
        wn_file: str
        outfile: str

        Returns
        ---------
        The sense keys.
    """
    df_arasaac = read_initial_voc_file(file_arasaac)
    df_dicopicto = read_dicopicto(dicopicto_file)
    df_wn = parse_wn31_file(wn_file)
    add_synset_to_dicoPicto(df_arasaac, df_dicopicto, df_wn)
    id_picto, lemma, synsets, sense_keys = create_common_lexique(df_arasaac, df_dicopicto, df_wn)
    final_lexique = pd.DataFrame({"id_picto": id_picto, "lemma": lemma, "synsets": synsets, "sense_keys": sense_keys})
    final_lexique.to_csv(outfile,
                         sep='\t', index=False)

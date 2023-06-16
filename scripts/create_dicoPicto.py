import pandas as pd

def read_initial_voc_file(file_arasaac):
    df = pd.read_csv(file_arasaac)
    return df[['idpicto', 'synset2']]


def read_dicoPicto(dicoPicto_file):
    return pd.read_csv(dicoPicto_file, sep='\t')


def parse_wn31_file(file):
    """Parse le fichier index.sense de wordnet 3.1"""
    try :
        data_wn31 = pd.read_csv(file, delimiter=" ", names=["sense_key", "synset", "id1", "id2"], header=None)
        return data_wn31
    except IOError:
        print("Could not read file, wrong file format.", file)
        return

def get_pos(synset):
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

def get_sense_keys(wn_data, synsets):
    # '07769568-n'
    sense_keys = []
    if synsets:
        for s in synsets:
            if s != '\\N':
                synset = s.split('-')[0]
                tag = get_pos(s)
                all_sense_keys = list(set(wn_data.loc[wn_data['synset'] == int(synset)]["sense_key"].tolist()))
                if all_sense_keys:
                    for sense in all_sense_keys:
                        sense_tag = sense.split('%')[1][0]
                        if int(sense_tag) == tag:
                            sense_keys.append(sense)
    return sense_keys


def add_synset_to_dicoPicto(df_arasaac, df_dicoPicto, df_wn):
    synsets = []
    sense_keys = []
    for index, row in df_dicoPicto.iterrows():
        synset = list(set(df_arasaac.loc[df_arasaac['idpicto'] == int(row['id_picto'])]["synset2"].tolist()))
        synsets.append(synset)
        senses = get_sense_keys(df_wn, synset)
        sense_keys.append(senses)
    df_dicoPicto['synsets'] = synsets
    df_dicoPicto['sense_keys'] = sense_keys


def create_dicoPicto(file_arasaac, dicoPicto_file, wn_file):
    df_arasaac = read_initial_voc_file(file_arasaac)
    df_dicoPicto = read_dicoPicto(dicoPicto_file)
    df_wn = parse_wn31_file(wn_file)
    add_synset_to_dicoPicto(df_arasaac, df_dicoPicto, df_wn)
    df_dicoPicto.to_csv("/data/macairec/PhD/Grammaire/dico/dicoPicto_synsets.csv", sep='\t')

create_dicoPicto("/data/macairec/PhD/Grammaire/dico/arasaac.fre30bis.csv", "/data/macairec/PhD/Grammaire/dico/dicoPicto.csv", "/data/macairec/PhD/Grammaire/dico/index.sense")
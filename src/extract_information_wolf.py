import xml.etree.ElementTree as ET
import pandas as pd

def read_wolf_data(file):
    # Charger le fichier XML
    tree = ET.parse(file)
    root = tree.getroot()

    wolf_data = {}

    # Parcourir chaque élément SYNSET
    for synset in root.findall('.//SYNSET'):
        # Extraire le contenu entre les balises LITERAL
        literal = synset.find('./SYNONYM/LITERAL').text

        # Extraire le contenu de la balise ID
        synset_id = synset.find('./ID').text.split("eng-30-")[1]

        # derivatives = [ilr.text.split("eng-30-")[1] for ilr in synset.findall('./ILR[@type="eng_derivative"]')]
        if literal != "_EMPTY_":
            if literal not in wolf_data.keys():
                wolf_data[literal] = [synset_id]
            else:
                wolf_data[literal].append(synset_id)
            # if derivatives:
            #     wolf_data[literal] = [synset_id] + derivatives
            # else:
            #     wolf_data[literal] = [synset_id]
    return wolf_data


def parse_wn30_file(file):
    """
        Parse the WordNet 3.0 file.

        Arguments
        ---------
        file: str

        Returns
        -------
        A dataframe with the sense keys from WordNet 3.1.
    """
    try:
        data_wn30 = pd.read_csv(file, delimiter=" ", names=["sense_key", "synset", "id1", "id2"], header=None)
        return data_wn30
    except IOError:
        print("Could not read file, wrong file format.", file)
        return


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
        pos = '-b'
    elif tag == 5:
        pos = '-a'
    return pos


def keep_relevant_sense_keys(sense_keys, pos):
    clean_senses = []
    for sense in sense_keys:
        sense_pos = int(sense.split("%")[1].split(":")[0])
        pos_synset = get_pos_synset(sense_pos)
        if pos_synset == '-'+pos:
            clean_senses.append(sense)
    return clean_senses



def get_sense_keys_from_synsets_wolf(wolf_data, data_wn30):
    f = open("wolf_data.txt", "w")
    for k,v in wolf_data.items():
        senses = []
        for s in v:
            sense = s.split('-')[0]
            pos = s.split('-')[1]
            all_sense_keys = list(set(data_wn30.loc[data_wn30['synset'] == int(sense)]["sense_key"].tolist()))
            senses.extend(keep_relevant_sense_keys(all_sense_keys, pos))
        f.write(k + '\t' + str(list(set(senses))) + '\n')
    f.close()


def process_wolf(wolf_file, wn30_file):
    wolf_data = read_wolf_data(wolf_file)
    data_wn30 = parse_wn30_file(wn30_file)
    get_sense_keys_from_synsets_wolf(wolf_data, data_wn30)

if __name__ == '__main__':
    process_wolf("/home/cecilemacaire/wolf.xml", "/home/cecilemacaire/index_wn_30.sense")





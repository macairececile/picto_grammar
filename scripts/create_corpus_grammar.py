import xml.etree.ElementTree as ET

import sox
from pydub import AudioSegment
from os import listdir
from os.path import join, isfile
import pandas as pd


def get_files_from_directory(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f)) if '.trs' in f or '.orfeo' in f]
    return files


def process_tcof_file(trs_file, data):
    # Charger le fichier XML
    tree = ET.parse(trs_file)
    root = tree.getroot()

    for turn in root.iter("Turn"):
        text = ' '.join(process_text_tcof("".join(turn.itertext())).split())
        starttime = turn.attrib.get("startTime")
        endtime = turn.attrib.get("endTime")

        if text != '':
            data.append({'file': trs_file, 'text': text, 'start': starttime, 'end': endtime})

def process_tcof_file_blocus(trs_file, data):
    # Analyse du XML
    # Analyse du XML
    tree = ET.parse(trs_file)
    root = tree.getroot()

    # Récupération des balises <Sync> avec leurs attributs
    for turn in root.iter("Turn"):
        endtime = turn.attrib.get("endTime")
        num_syncs = len(turn.findall("Sync"))
        for i, sync in enumerate(turn.findall("Sync")):
            text = ' '.join(process_text_tcof(sync.tail).split())
            start = sync.attrib.get("time")
            if i == 0 and text != '' and i + 1 < num_syncs:
                end = turn.findall("Sync")[i+1].attrib.get("time")
                data.append({'file': trs_file, 'text': text, 'start': start, 'end': end})
            elif i + 1 < num_syncs:
                end = turn.findall("Sync")[i + 1].attrib.get("time")
                data.append({'file': trs_file, 'text': text, 'start': start, 'end': end})
            else:
                data.append({'file': trs_file, 'text': text, 'start': start, 'end': endtime})



def process_text_tcof(text):
    to_replace = ['+', '///', '- ', ' -', '***']
    to_replace_2 = ['(', ')']
    for i in to_replace:
        text = text.replace(i, " ")
    for j in to_replace_2:
        text = text.replace(j, "")
    return text


def process_clapi_file(clapi_file, data):
    with open(clapi_file, 'r') as file:
        texte = None
        starttime = 0
        for line in file:
            if line == '\n':
                endtime = colonnes[11]
                data.append({'file': clapi_file, 'text': texte, 'start': starttime, 'end': endtime})
            elif line.startswith("# text"):
                texte = line.strip().split(" = ")[-1]
            elif not line.startswith("#"):
                colonnes = line.strip().split("\t")
                if colonnes[0] == '1':
                    starttime = colonnes[10]


def create_clips_from_timecode(df, save_dir):
    name_clips = []
    for index, row in df.iterrows():
        if 'clapi' in row['file']:
            wav_file = row['file'][:-6] + '.wav'
        else:
            wav_file = row['file'][:-4] + '.wav'
        tfm = sox.Transformer()
        if row['end'] < row['start']:
            name_clips.append('')
        else:
            tfm.trim(float(row['start']), float(row['end']))
            tfm.compand()
            file_clip = save_dir + wav_file[:-4].split('/')[-1] + "_{}.wav".format(index)
            tfm.build_file(wav_file, file_clip)
            name_clips.append(wav_file[:-4].split('/')[-1] + "_{}.wav".format(index))
    df['clips'] = name_clips
    # df.insert(0, 'clips', df.pop('clips'))


def create_corpus(folder_tcof, folder_clapi, save_clips):
    data = []
    trs_files_tcof = get_files_from_directory(folder_tcof)
    orfeo_files_clapi = get_files_from_directory(folder_clapi)
    for f in trs_files_tcof:
        if "Blocus" in f:
            process_tcof_file_blocus(folder_tcof + f, data)
        else:
            process_tcof_file(folder_tcof + f, data)
    for f in orfeo_files_clapi:
        process_clapi_file(folder_clapi + f, data)

    df = pd.DataFrame(data, columns=['file', 'text', 'start', 'end'])
    create_clips_from_timecode(df, save_clips)
    df = df[df.clips != '']
    df[['clips', 'text']].to_csv("/data/macairec/PhD/Grammaire/corpus/csv/corpus_grammar.csv", sep='\t', index=False)


create_corpus("/data/macairec/PhD/Grammaire/corpus/data/tcof/", "/data/macairec/PhD/Grammaire/corpus/data/clapi/",
              "/data/macairec/PhD/Grammaire/corpus/clips/")
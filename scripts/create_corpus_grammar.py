import xml.etree.ElementTree as ET

import sox
from os import listdir
from os.path import join, isfile
import pandas as pd
import textgrid
import re


def get_files_from_directory(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f)) if
             '.trs' in f or '.orfeo' in f or '.TextGrid' in f or '.xml' in f or ".orfeo_golded" in f]
    return files


def process_trs_file(trs_file, data):
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
                end = turn.findall("Sync")[i + 1].attrib.get("time")
                data.append({'file': trs_file, 'text': text, 'start': start, 'end': end})
            elif i + 1 < num_syncs:
                end = turn.findall("Sync")[i + 1].attrib.get("time")
                data.append({'file': trs_file, 'text': text, 'start': start, 'end': end})
            else:
                data.append({'file': trs_file, 'text': text, 'start': start, 'end': endtime})


def process_text_tcof(text):
    to_replace = ['+', '///', '- ', ' -', '***', '*', '=']
    to_replace_2 = ['(', ')']
    for i in to_replace:
        text = text.replace(i, " ")
    for j in to_replace_2:
        text = text.replace(j, "")
    return text


def process_text_pfc(text):
    sp_text = text.split(" ")
    new_text = ''
    for w in sp_text:
        if '/' not in w and '(' not in w and '<' not in w:
            if '>' in w:
                new_text += w.split('>')[0] + ' '
            else:
                new_text += w + ' '
    return new_text.strip()


def process_orfeo_file(clapi_file, data):
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


def process_cfpr_file(cfpr_file, data):
    tg = textgrid.TextGrid.fromFile(cfpr_file)
    for i in range(tg[0].__len__()):
        if tg[0][i].mark != '_':
            data.append({'file': cfpr_file, 'text': tg[0][i].mark, 'start': float(tg[0][i].minTime),
                         'end': float(tg[0][i].maxTime)})


def process_pfc_file(pfc_file, data):
    tree = ET.parse(pfc_file)
    root = tree.getroot()

    for s_elem in root.iter('S'):
        audio_elem = s_elem.find('AUDIO')
        form_elem = s_elem.find('FORM[@kindOf="ortho"]')

        if audio_elem is not None and form_elem is not None:
            text = process_text_pfc(form_elem.text.strip())
            data.append({'file': pfc_file, 'text': text, 'start': float(audio_elem.attrib['start']),
                         'end': float(audio_elem.attrib['end'])})


def create_clips_from_timecode(df, save_dir):
    name_clips = []
    for index, row in df.iterrows():
        if 'orfeo' in row['file']:
            wav_file = row['file'][:-6] + '.wav'
        elif 'TextGrid' in row['file']:
            wav_file = row['file'][:-9] + '.wav'
        else:
            wav_file = row['file'][:-4] + '.wav'
        tfm = sox.Transformer()
        if float(row['end']) <= float(row['start']):
            name_clips.append('')
        else:
            tfm.trim(float(row['start']), float(row['end']))
            tfm.compand()
            file_clip = save_dir + wav_file[:-4].split('/')[-1] + "_{}.wav".format(index)
            tfm.build_file(wav_file, file_clip)
            name_clips.append(wav_file[:-4].split('/')[-1] + "_{}.wav".format(index))
    df['clips'] = name_clips
    # df.insert(0, 'clips', df.pop('clips'))


def process_orfeo_adrien(file, df):
    name_clip = []
    current_sentence = ""
    sentences = []
    with open(file, 'r') as f:
        for line in f:
            if line.startswith("# sent_id"):
                if current_sentence:
                    s = current_sentence.replace("' ", "'").replace("#", ' ')
                    maj = re.findall(r'[A-Z]', s)
                    s = s.replace(' '.join(maj), ''.join(maj))
                    s = re.sub(r'\w+~', '', s)
                    s = re.sub(r'\w+-~', '', s)
                    sentences.append(' '.join(s.split()))
                    current_sentence = ""
                name_clip.append(line.split("=")[1].strip())
            elif line[0].isdigit():
                word = line.split("\t")[1]
                current_sentence += " " + word

        if current_sentence:
            s = current_sentence.replace("' ", "'").replace("#", ' ')
            maj = re.findall(r'[A-Z]', s)
            s = s.replace(' '.join(maj), ''.join(maj))
            s = re.sub(r'\w+~', '', s)
            s = re.sub(r'\w+-~', '', s)
            sentences.append(' '.join(s.split()))
    for i in range(len(name_clip)):
        df.append({'file': file.split('/')[-1], 'clips': name_clip[i], 'text': sentences[i]})


def create_corpus(folder_tcof, folder_cfpp, folder_ordeo, folder_cfpr, folder_pfc, save_clips):
    data = []
    trs_files_tcof = get_files_from_directory(folder_tcof)
    trs_files_cfpp = get_files_from_directory(folder_cfpp)
    orfeo_files = get_files_from_directory(folder_ordeo)
    cfpr_files = get_files_from_directory(folder_cfpr)
    pfc_files = get_files_from_directory(folder_pfc)
    for f in trs_files_tcof:
        process_trs_file(folder_tcof + f, data)
    for f in trs_files_cfpp:
        process_trs_file(folder_cfpp + f, data)
    for f in orfeo_files:
        process_orfeo_file(folder_ordeo + f, data)
    for f in cfpr_files:
        process_cfpr_file(folder_cfpr + f, data)
    for f in pfc_files:
        process_pfc_file(folder_pfc + f, data)
    df = pd.DataFrame(data, columns=['file', 'text', 'start', 'end'])

    create_clips_from_timecode(df, save_clips)
    df = df[df.clips != '']
    select_20_sentences_per_file(df)
    df[['clips', 'text']].to_csv(
        "/data/macairec/PhD/Grammaire/corpus/csv/corpus_grammar_selected_sentences_from_audio.csv", sep='\t',
        index=False)


def select_20_sentences_per_file(df):
    selected_phrases_list = []

    grouped = df.groupby('file')

    for file_name in df['file'].unique():
        group_phrases = grouped.get_group(file_name)
        shuffled_phrases = group_phrases.sample(frac=1).reset_index(drop=True)
        # selected = shuffled_phrases[
        #     shuffled_phrases['text'].apply(lambda x: 4 <= len(x.split()) <= 14)
        # ]
        selected = shuffled_phrases[
            shuffled_phrases['text'].apply(lambda x: 14 <= len(x.split()) <= 25)
        ]
        selected_phrases = selected.head(2)
        selected_phrases_list.append(selected_phrases)

    selected_df = pd.concat(selected_phrases_list, ignore_index=True)
    selected_df[['clips', 'text']].to_csv(
        "/data/macairec/PhD/Grammaire/corpus/csv/corpus_grammar_selected_sentences_from_ordeo_adrien_20_large.csv",
        sep='\t',
        index=False)


def create_corpus_from_adrien(folder_ordeo):
    data = []
    orfeo_files = get_files_from_directory(folder_ordeo)
    for f in orfeo_files:
        process_orfeo_adrien(folder_ordeo + f, data)
    df = pd.DataFrame(data, columns=['file', 'clips', 'text'])
    df = df[df.clips != '']
    select_20_sentences_per_file(df)
    df[['clips', 'text']].to_csv(
        "/data/macairec/PhD/Grammaire/corpus/csv/corpus_grammar_selected_sentences_from_ordeo_adrien.csv",
        sep='\t',
        index=False)


create_corpus_from_adrien(
    "/data/macairec/PhD/Grammaire/corpus/data/orfeo_adrien/")

# create_corpus("/data/macairec/PhD/Grammaire/corpus/data/tcof/", "/data/macairec/PhD/Grammaire/corpus/data/cfpp/",
#               "/data/macairec/PhD/Grammaire/corpus/data/orfeo/",
#               "/data/macairec/PhD/Grammaire/corpus/data/cfpr/", "/data/macairec/PhD/Grammaire/corpus/data/pfc/",
#               "/data/macairec/PhD/Grammaire/corpus/clips/")

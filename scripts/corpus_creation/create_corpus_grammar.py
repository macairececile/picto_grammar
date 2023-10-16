"""
Script to process the orfeo corpus, extract aligned transcriptions with clip name.

Example of use: python create_corpus_grammar.py --datadir /.../ --save_dir corpus/
"""

import xml.etree.ElementTree as ET

import sox
from os import listdir
from os.path import join, isfile
import pandas as pd
import textgrid
import re
from argparse import ArgumentParser, RawTextHelpFormatter


def get_files_from_directory(dir):
    """
        Get the files from a specific format in a directory.

        Arguments
        ---------
        dir: str
            Path of the folder.

        Returns
        -------
        A list of files.
    """
    files = [f for f in listdir(dir) if isfile(join(dir, f)) if
             '.trs' in f or '.orfeo' in f or '.TextGrid' in f or '.xml' in f]
    return files


def process_trs_file(trs_file, data):
    """
        Retrieve specific information from a trs file.

        Arguments
        ---------
        trs_file: str
        data: list
    """
    tree = ET.parse(trs_file)
    root = tree.getroot()

    for turn in root.iter("Turn"):
        text = ' '.join(process_text_tcof("".join(turn.itertext())).split())
        starttime = turn.attrib.get("startTime")
        endtime = turn.attrib.get("endTime")

        if text != '':
            data.append({'file': trs_file, 'text': text, 'start': starttime, 'end': endtime})


def process_tcof_file_format_bis(trs_file, data):
    """
        Process files with a specific format

        Arguments
        ---------
        trs_file: str
        data: list
    """
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
    """
        Process text from TCOF corpus.

        Arguments
        ---------
        text: str

        Returns
        -------
        A processed text.
    """
    to_replace = ['+', '///', '- ', ' -', '***', '*', '=']
    to_replace_2 = ['(', ')']
    for i in to_replace:
        text = text.replace(i, " ")
    for j in to_replace_2:
        text = text.replace(j, "")
    return text


def process_text_pfc(text):
    """
        Process text from PFC corpus.

        Arguments
        ---------
        text: str

        Returns
        -------
        A processed text.
    """
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
    """
        Process file from ORFEO.

        Arguments
        ---------
        clapi_file: str
        data: list
    """
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
    """
        Process file from CFPR.

        Arguments
        ---------
        cfpr_file: str
        data: list
    """
    tg = textgrid.TextGrid.fromFile(cfpr_file)
    for i in range(tg[0].__len__()):
        if tg[0][i].mark != '_':
            data.append({'file': cfpr_file, 'text': tg[0][i].mark, 'start': float(tg[0][i].minTime),
                         'end': float(tg[0][i].maxTime)})


def process_pfc_file(pfc_file, data):
    """
        Process file from PFC.

        Arguments
        ---------
        pfc_file: str
        data: list
    """
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
    """
        Create clips from the retrieved information, and save the clip names in the dataframe.

        Arguments
        ---------
        df: dataframe
            Dataframe with information
        save_dir: str
            Directory to save the created clips
    """
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


def process_orfeo_cleaned(file, df):
    """
        Retrieve information from the cleaned orfeo repository.

        Arguments
        ---------
        df: dataframe
            Dataframe to store information
        file: str
            File from orfeo.
    """
    name_clip = []
    current_sentence = ""
    sentences = []
    with open(file, 'r') as f:
        for line in f:
            if line.startswith("# sent_id"):
                if current_sentence:
                    s = current_sentence.replace("' ", "'").replace("#", ' ').replace("«", ' ').replace("»", ' ')
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


def create_corpus(args):
    """
        Create a csv file with clip names and associated text, as well as a directory with clip files.
    """
    data = []
    trs_files_tcof = get_files_from_directory(args.folder_tcof)
    trs_files_cfpp = get_files_from_directory(args.folder_cfpp)
    orfeo_files = get_files_from_directory(args.folder_ordeo)
    cfpr_files = get_files_from_directory(args.folder_cfpr)
    pfc_files = get_files_from_directory(args.folder_pfc)
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

    create_clips_from_timecode(df, args.save_clips)
    df = df[df.clips != '']
    final_df = df.drop_duplicates()
    final_df[['clips', 'text']].to_csv(args.save_dir + "corpus.csv", sep='\t', index=False)


def create_corpus_from_orfeo(args):
    """
        Create a csv file with clip names and associated text, as well as a directory with clip files.
    """
    orfeo_dir = args.datadir
    data = []
    orfeo_files = get_files_from_directory(args.datadir)
    for f in orfeo_files:
        process_orfeo_cleaned(orfeo_dir + f, data)
    df = pd.DataFrame(data, columns=['file', 'clips', 'text'])
    df = df[df.clips != '']
    final_df = df.drop_duplicates()
    final_df[['clips', 'text']].to_csv(args.save_dir + "corpus.csv", sep='\t', index=False)


parser = ArgumentParser(description="Create corpus for grammar.",
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('--datadir', type=str, required=True,
                    help="Directory where the data are stored")
parser.add_argument('--save_dir', type=str, required=True,
                    help="Directory to save the corpus.")
parser.set_defaults(func=create_corpus_from_orfeo)
args = parser.parse_args()
args.func(args)

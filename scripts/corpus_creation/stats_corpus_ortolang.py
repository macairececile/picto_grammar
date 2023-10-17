"""
Script to calculate the main statistics on the ortolang corpora.

Example of use: python stats_corpus_ortolang.py --path_clips '/clips/' --json_orfeo 'orfeo.json' --json_eval 'eval.json
"""

import json
import librosa
from tabulate import tabulate
from argparse import ArgumentParser, RawTextHelpFormatter


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
    sentences = [item["sentence"] for item in data]
    pictos = [item["pictos"] for item in data]
    tokens = [item["pictos_tokens"] for item in data]
    return clip_names, sentences, pictos, tokens


def get_total_utterances(sentences):
    """
        Get the number of sentences.

        Arguments
        ---------
        sentences: list

        Returns
        -------
        The number of sentences.
    """
    return len(sentences)


def get_total_num_words(sentences):
    """
        Get the total number of words in the corpus.

        Arguments
        ---------
        sentences: list

        Returns
        -------
        The number of words in the corpus.
    """
    total_words = []
    for i in sentences:
        total_words.extend(i.split(" "))
    return len(total_words)


def get_total_unique_words(sentences):
    """
        Get the total number of unique words in the corpus.

        Arguments
        ---------
        sentences: list

        Returns
        -------
        The number of unique words in the corpus.
    """
    total_unique_words = []
    for i in sentences:
        total_unique_words.extend(i.split(" "))
    return total_unique_words, len(list(set(total_unique_words)))


def get_total_num_pictos(pictos):
    """
        Get the total number of pictograms in the corpus.

        Arguments
        ---------
        pictos: list

        Returns
        -------
        The number of pictograms in the corpus.
    """
    return sum(map(len, pictos))


def get_total_num_unique_pictos(pictos):
    """
        Get the total number of unique pictograms in the corpus.

        Arguments
        ---------
        pictos: list

        Returns
        -------
        The number of unique pictograms in the corpus.
    """
    all_pictos = [p for innerList in pictos for p in innerList]
    return all_pictos, len(list(set(all_pictos)))


def get_duration(path_clips, clips_names):
    """
        Get the duration of the corpus in minutes.

        Arguments
        ---------
        path_clips: str
        clips_names: list

        Returns
        -------
        The total duration of clips in the corpus.
    """
    return sum(librosa.get_duration(path=path_clips + c + '.wav') / 60 for c in clips_names)


def compute_stats_and_print(clip_names, sentences, pictos, path_clips):
    """
        Calculate stats for one corpus.

        Arguments
        ---------
        clip_names: list
        sentences: list
        pictos: list
        path_clips: str

        Returns
        -------
        The stats
    """
    num_utt = get_total_utterances(sentences)
    # duration = get_duration(path_clips, clip_names)
    num_all_pictos = get_total_num_pictos(pictos)
    all_pictos, num_unique_pictos = get_total_num_unique_pictos(pictos)
    num_all_words = get_total_num_words(sentences)
    unique_words, num_unique_words = get_total_unique_words(sentences)
    return num_utt, all_pictos, num_all_pictos, num_unique_pictos, num_all_words, unique_words, num_unique_words


def stats_corpus_orfeo_ortolang(path_clips, path_json, names):
    """
        Calculate stats for the all corpus.

        Arguments
        ---------
        path_clips: str
        path_json: str
        names: list
    """
    num_utts, pictos_num, unique_pictos, words, unique_words_num = ["#utterances"], ["#pictos"], ["#unique_pictos"], [
        "#words"], ["#unique_words"]
    all_unique_pictos = []
    all_unique_words = []
    for n in names:
        json_file = path_json + n + "/corpus_" + n + ".json"
        clip_names, sentences, pictos, tokens = read_json_file(json_file)
        num_utt, all_pictos, num_all_pictos, num_unique_pictos, num_all_words, unique_words, num_unique_words = compute_stats_and_print(
            clip_names, sentences, pictos, path_clips)
        num_utts.append(num_utt)
        pictos_num.append(num_all_pictos)
        unique_pictos.append(num_unique_pictos)
        words.append(num_all_words)
        unique_words_num.append(num_unique_words)
        all_unique_pictos.extend(all_pictos)
        all_unique_words.extend(unique_words)
    total_unique_words = len(list(set(all_unique_words)))
    total_unique_pictos = len(list(set(all_unique_pictos)))
    data = [
        num_utts + [sum(num_utts[1:])],
        # ["Duration", f"{duration} minutes"],
        words + [sum(words[1:])],
        unique_words_num + [total_unique_words],
        pictos_num + [sum(pictos_num[1:])],
        unique_pictos + [total_unique_pictos]
    ]

    table = tabulate(data, headers=["Corpus_name", "cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron",
                                    "frenchoralnarrative", "ofrom", "reunions", "tcof",
                                    "tufs", "valibel", "all"], tablefmt="grid")
    print(table)


def stats_corpus_eval_ortolang(json_file):
    """
        Calculate stats for the eval corpus.

        Arguments
        ---------
        json_file: str
    """
    num_utts, pictos_num, unique_pictos, words, unique_words_num = ["#utterances"], ["#pictos"], ["#unique_pictos"], [
        "#words"], ["#unique_words"]
    clip_names, sentences, pictos, tokens = read_json_file(json_file)
    num_utt, all_pictos, num_all_pictos, num_unique_pictos, num_all_words, unique_words, num_unique_words = compute_stats_and_print(
        clip_names, sentences, pictos, "path_clips")
    num_utts.append(num_utt)
    pictos_num.append(num_all_pictos)
    unique_pictos.append(num_unique_pictos)
    words.append(num_all_words)
    unique_words_num.append(num_unique_words)
    data = [
        num_utts,
        # ["Duration", f"{duration} minutes"],
        words,
        unique_words_num,
        pictos_num,
        unique_pictos
    ]

    table = tabulate(data, headers=["Corpus_name", "corpus-eval"], tablefmt="grid")
    print(table)


def main(args):
    names = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative", "ofrom", "reunions", "tcof",
             "tufs", "valibel"]
    stats_corpus_orfeo_ortolang(args.path_clips, args.json_orfeo, names)
    stats_corpus_eval_ortolang(args.json_eval)


parser = ArgumentParser(description="Calculate general stats on corpora from ortolang.",
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('--path_clips', type=str, required=True,
                    help="")
parser.add_argument('--json_orfeo', type=str, required=True,
                    help="")
parser.add_argument('--json_eval', type=str, required=True,
                    help="")
parser.set_defaults(func=main)
args = parser.parse_args()
args.func(args)

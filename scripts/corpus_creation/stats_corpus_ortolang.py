import json
import librosa
from tabulate import tabulate


def read_json_file(json_file):
    f = open(json_file)
    data = json.load(f)
    clip_names = [item["audio"] for item in data]
    sentences = [item["sentence"] for item in data]
    pictos = [item["pictos"] for item in data]
    tokens = [item["pictos_tokens"] for item in data]
    return clip_names, sentences, pictos, tokens


def get_total_utterances(sentences):
    return len(sentences)


def get_total_num_words(sentences):
    total_words = []
    for i in sentences:
        total_words.extend(i.split(" "))
    return len(total_words)


def get_total_unique_words(sentences):
    total_unique_words = []
    for i in sentences:
        total_unique_words.extend(i.split(" "))
    return total_unique_words, len(list(set(total_unique_words)))


def get_total_num_pictos(pictos):
    return sum(map(len, pictos))


def get_total_num_unique_pictos(pictos):
    all_pictos = [p for innerList in pictos for p in innerList]
    return all_pictos, len(list(set(all_pictos)))


def get_duration(path_clips, clips_names):
    return sum(librosa.get_duration(path=path_clips + c + '.wav') / 60 for c in clips_names)


def compute_stats_and_print(clip_names, sentences, pictos, path_clips):
    num_utt = get_total_utterances(sentences)
    # duration = get_duration(path_clips, clip_names)
    num_all_pictos = get_total_num_pictos(pictos)
    all_pictos, num_unique_pictos = get_total_num_unique_pictos(pictos)
    num_all_words = get_total_num_words(sentences)
    unique_words, num_unique_words = get_total_unique_words(sentences)
    return num_utt, all_pictos, num_all_pictos, num_unique_pictos, num_all_words, unique_words, num_unique_words


def stats_propicto_orfeo(path_clips, names):
    num_utts, pictos_num, unique_pictos, words, unique_words_num = ["#utterances"], ["#pictos"], ["#unique_pictos"], [
        "#words"], ["#unique_words"]
    all_unique_pictos = []
    all_unique_words = []
    for n in names:
        json_file = "/data/macairec/PhD/Grammaire/corpus/ortolang/propicto-orfeo/" + n + "/corpus_" + n + ".json"
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


def stats_propicto_eval(json_file):
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

    table = tabulate(data, headers=["Corpus_name", "propicto-eval"], tablefmt="grid")
    print(table)


if __name__ == '__main__':
    # names = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative", "ofrom", "reunions", "tcof",
    #          "tufs", "valibel"]
    # stats_propicto_orfeo("", names)

    stats_propicto_eval("/data/macairec/PhD/Grammaire/corpus/ortolang/propicto-eval/propicto-eval.json")

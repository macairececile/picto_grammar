# Author: Cécile MACAIRE
# Version: 22/11/2023

import json
import evaluate

# import the metrics from the evaluate library
bleu = evaluate.load("bleu")
meteor = evaluate.load('meteor')
wer = evaluate.load('wer')


def read_json_file_submitted(json_file):
    """
        Read the json test file and retrieve the reference and hypothesis.

        Arguments
        ---------
        json_file: str

        Returns
        ---------
        A dictionary with the data submitted by the participant.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    subm_data = {item["id"]: item.get("hyp") for item in data}
    return subm_data


def read_json_file_gold(json_file):
    """
        Read the json test file and retrieve the reference and hypothesis.

        Arguments
        ---------
        json_file: str

        Returns
        ---------
        A dictionary with the gold data.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    gold_data = {item["id"]: item.get("ref") for item in data}
    return gold_data


def match_data(gold_data, subm_data):
    """
        From the dictionaries, get missing items, and retrieve refs and hyps for eval.

        Arguments
        ---------
        gold_data: dict
        subm_data: dict

        Returns
        ---------
        Two lists with the references and the hypothesis
    """
    num_missing_ids = 0
    missing_ids = []
    refs_eval = []
    hyps_eval = []
    for k, v in gold_data.items():
        if k in subm_data.keys():
            refs_eval.append(v)
            hyps_eval.append(subm_data[k])
        else:
            refs_eval.append(v)
            hyps_eval.append("")
            num_missing_ids += 1
            missing_ids.append(k)
    if num_missing_ids > 0:
        print("--------------------------------------------------------------------------------------------------")
        print("For " + str(
            num_missing_ids) + " utterance(s), you did not provide any hypothesis. "
                               "This will subsequently affect the result.")
        print("Here is the list of missing items:")
        for el in missing_ids:
            print(el)
        print("--------------------------------------------------------------------------------------------------")
    return refs_eval, hyps_eval


def compute_scores(refs, hyps):
    """
        Compute the scores to evaluate the translation from speech or text input.

        Arguments
        ---------
        refs: list
        hyps: list

        Returns
        ---------
        The BLEU, PictoER and METEOR scores.
    """
    bleu_score = bleu.compute(predictions=hyps, references=refs)
    picto_term_score = wer.compute(predictions=hyps, references=refs)
    meteor_score = meteor.compute(predictions=hyps, references=refs)
    return bleu_score["bleu"], picto_term_score * 100, meteor_score["meteor"]


def print_automatic_eval(bleu_score, pictoer_score, meteor_score):
    """
        Print the results from the evaluation.

        Arguments
        ---------
        bleu_score: float
        pictoer_score: float
        meteor_score: float
    """
    print("\n----------------- EVALUATION ------------------")
    print("|  BLEU  | Picto-term Error Rate (%) | METEOR |")
    print("|---------------------------------------------|")
    print("| {:<6.3f} | {:<25.3f} | {:<6.3f} |".format(bleu_score * 100, pictoer_score, meteor_score * 100))
    print("-----------------------------------------------")


def main():
    gold_data = read_json_file_gold("test_gold.json")
    subm_data = read_json_file_submitted('test.json')
    refs, hyps = match_data(gold_data, subm_data)
    bleu_score, picto_term_error_rate, meteor_score = compute_scores(refs, hyps)
    print_automatic_eval(bleu_score, picto_term_error_rate, meteor_score)


if __name__ == '__main__':
    main()

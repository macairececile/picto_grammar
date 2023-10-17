"""
Script to evaluate the grammar translation in pictograms from the ASR predictions.

Example of use:
    python evaluate_grammar_from_asr.py --eval_choice "all" --file_ref "whisper_results_small_grammar.csv"
    --file_asr_grammar "corpus_all_grammar_pictos.csv"
    python evaluate_grammar_from_asr.py --eval_choice "all" --file_ref "whisper_results_small_grammar.csv"
    --file_asr_grammar "corpus_all_grammar_pictos.csv" --file_asr "whisper_results.csv"

"""

import numpy as np
import pandas as pd
import evaluate
from argparse import ArgumentParser, RawTextHelpFormatter

bleu = evaluate.load("bleu")
meteor = evaluate.load('meteor')
wer = evaluate.load('wer')

pd.options.mode.chained_assignment = None


def get_corpus_name(clip_name):
    """
        Get the corpus name.

        Arguments
        ---------
        clip_name: str

        Returns
        -------
        The clip name.
    """
    names = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative", "ofrom", "reunions", "tcof",
             "tufs", "valibel"]
    for n in names:
        if n in clip_name:
            return n


def clean_data_asr(data_asr):
    """
        Clean the prediction from ASR.

        Arguments
        ---------
        data_asr: dataframe
    """
    for i, row in data_asr.iterrows():
        if isinstance(row['text'], float):
            row['tokens'] = ''


def get_ref_pictos_and_tokens(data_asr, data_refs):
    """
        Get the associated reference sequence of pictograms and tokens from the ref data.

        Arguments
        ---------
        data_asr: dataframe
        data_refs: dataframe

        Returns
        ---------
        The dataframe with ref picto + tokens + corpus_name per row.
    """
    tokens_refs = []
    pictos_refs = []
    corpus_name = []
    for i, row in data_asr.iterrows():
        clip = row["clips"]
        row_ref = data_refs.loc[data_refs['clips'] == clip]
        if row_ref.empty:
            tokens_refs.append("to_delete")
            pictos_refs.append("to_delete")
            corpus_name.append(get_corpus_name(clip))
        else:
            tokens_refs.append(row_ref['tokens'].values[0])
            pictos_refs.append(row_ref['pictos'].values[0])
            corpus_name.append(get_corpus_name(clip))

    data_asr["ref_tokens"] = tokens_refs
    data_asr["ref_pictos"] = pictos_refs
    data_asr["corpus_name"] = corpus_name
    data = data_asr[data_asr.ref_tokens != 'to_delete']
    return data


def get_ref_tokens_v2(clip_name, data_refs):
    """
        Get the associated reference sequence of pictograms and tokens from the ref data.

        Arguments
        ---------
        clip_name: str
        data_refs: dataframe

        Returns
        ---------
        The token ref sequence.
    """
    row_ref = data_refs.loc[data_refs['clips'] == clip_name]
    if row_ref.empty:
        return "to_delete"
    else:
        return row_ref['tokens'].values[0]


def compute_scores(data_asr, corpus_name=None):
    """
        Compute the scores to evaluate the translation from ASR + grammar with the translation ref from grammar.

        Arguments
        ---------
        data_asr: dataframe
        corpus_name: str

        Returns
        ---------
        The BLEU, WER and METEOR scores.
    """
    if corpus_name:
        by_corpus = data_asr.loc[data_asr['corpus_name'] == corpus_name]
    else:
        by_corpus = data_asr
    references = []
    references_bleu = []
    predictions = []
    for i, row in by_corpus.iterrows():
        references.append(row["ref_tokens"])
        references_bleu.append(row["ref_tokens"])
        predictions.append(row["tokens"])
    results_wer = wer.compute(predictions=predictions, references=references)
    wer_score = round(results_wer, 3) * 100
    results_bleu = bleu.compute(predictions=predictions, references=references_bleu)
    bleu_score = round(results_bleu["bleu"], 3)
    results_meteor = meteor.compute(predictions=predictions, references=references)
    meteor_score = round(results_meteor["meteor"], 3)
    return bleu_score, wer_score, meteor_score


def print_automatic_eval(bleu_score, term_error_rate_score, meteor_score):
    """
        Print the results from the evaluation.

        Arguments
        ---------
        bleu_score: float
        term_error_rate_score: float
        meteor_score: float
    """
    print("-------------------")
    print("| BLEU  | METEOR | PER |")
    print("|----------------------------------|")
    print("| {:<5.3f} | {:<6.3f} | {:<4.1f} |".format(bleu_score, meteor_score, term_error_rate_score))
    print("-------------------")


def evaluate(file_asr, file_ref):
    """
        Run the evaluation.

        Arguments
        ---------
        file_asr: str
        file_ref: str
    """
    data_asr = pd.read_csv(file_asr, sep='\t')
    clean_data_asr(data_asr)
    data_refs = pd.read_csv(file_ref, sep='\t')
    data_asr = get_ref_pictos_and_tokens(data_asr, data_refs)
    names = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative", "ofrom", "reunions", "tcof",
             "tufs", "valibel"]
    for n in names:
        print("Score for the corpus " + n + ": ")
        bleu_score, wer_score, meteor_score = compute_scores(data_asr, n)
        print_automatic_eval(bleu_score, wer_score, meteor_score)
    bleu_score, wer_score, meteor_score = compute_scores(data_asr)
    print("Score all corpus: ")
    print_automatic_eval(bleu_score, wer_score, meteor_score)


def select_by_wer_scores(data_asr, data_asr_grammar, data_ref):
    """
        Select the data from a specific ASR WER range before running the evaluation.

        Arguments
        ---------
        data_asr: dataframe
        data_asr_grammar: dataframe
        data_ref: dataframe
    """
    print("Size of the dataset : ", len(data_asr))
    ints = [i for i in range(0, 51, 10)]
    identifier = 0
    for i in np.arange(0, 0.51, 0.10):
        if identifier < 5:
            df_by_wer = data_asr[(data_asr['wer'] >= i) & (data_asr['wer'] < i + 0.10)]
            print("Size subdataset : ", len(df_by_wer))
            print("For WER between " + str(ints[identifier]) + " and " + str(ints[identifier] + 10) + " : ")
        else:
            df_by_wer = data_asr[data_asr['wer'] >= i]
            print("Size subdataset : ", len(df_by_wer))
            print("For WER sup of  " + str(ints[identifier]) + " : ")
        ref_tokens = []
        tokens = []
        for t, row in df_by_wer.iterrows():
            infos_grammar = data_asr_grammar.loc[data_asr_grammar['clips'] == row['clips']]
            tokens_pred = infos_grammar["tokens"].values[0]
            tokens.append(tokens_pred)
            ref_tokens.append(get_ref_tokens_v2(row['clips'], data_ref))
        df_by_wer.loc[:, "ref_tokens"] = ref_tokens
        df_by_wer.loc[:, "tokens"] = tokens
        final_data = df_by_wer[df_by_wer.ref_tokens != 'to_delete']
        bleu_score, wer_score, meteor_score = compute_scores(final_data)
        print_automatic_eval(bleu_score, wer_score, meteor_score)
        identifier += 1


def evaluate_by_wer(file_asr, file_asr_grammar, file_ref):
    """
        Evaluate by ASR WER range.

        Arguments
        ---------
        file_asr: str
        file_asr_grammar: str
        file_ref: str
    """
    data_asr = pd.read_csv(file_asr, sep='\t')
    data_asr_grammar = pd.read_csv(file_asr_grammar, sep='\t')
    data_refs = pd.read_csv(file_ref, sep='\t')
    clean_data_asr(data_asr_grammar)
    select_by_wer_scores(data_asr, data_asr_grammar, data_refs)


def main(args):
    if args.eval_choice == "all":
        evaluate(args.file_asr_grammar, args.file_ref)
    elif args.eval_choice == "wer":
        evaluate_by_wer(args.file_asr, args.file_asr_grammar, args.file_ref)


parser = ArgumentParser(description="Evaluate the grammar from ASR predictions.",
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('--eval_choice', type=str, required=True, choices=["all", "wer"],
                    help="")
parser.add_argument('--file_ref', type=str, required=True,
                    help="")
parser.add_argument('--file_asr_grammar', type=str, required=True,
                    help="")
parser.add_argument('--file_asr', type=str, required=False,
                    help="")
parser.set_defaults(func=main)
args = parser.parse_args()
args.func(args)

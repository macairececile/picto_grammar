"""
Script to evaluate the grammar translation in pictograms from the ASR predictions.

Example of use:
    python evaluate_grammar_from_asr.py --eval_choice "all" --file_ref "whisper_results_small_grammar.csv"
    --file_asr_grammar "corpus_all_grammar_pictos.csv"
    python evaluate_grammar_from_asr.py --eval_choice "all" --file_ref "whisper_results_small_grammar.csv"
    --file_asr_grammar "corpus_all_grammar_pictos.csv" --file_asr "whisper_results.csv"

"""
import ast

import numpy as np
import pandas as pd
import evaluate
from argparse import ArgumentParser, RawTextHelpFormatter
from print_sentences_from_grammar import *

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
        return "to_delete", "", ""
    else:
        return row_ref['tokens'].values[0], row_ref['pictos'].values[0], row_ref['text'].values[0]


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
        references.append(row["pictos_grammar_tokens"])
        references_bleu.append(row["pictos_grammar_tokens"])
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
        ref_pictos = []
        ref_texts = []
        pictos_preds = []
        for t, row in df_by_wer.iterrows():
            infos_grammar = data_asr_grammar.loc[data_asr_grammar['clips'] == row['clips']]
            tokens_pred = infos_grammar["tokens"].values[0]
            pictos_pred = infos_grammar["pictos"].values[0]
            pictos_preds.append(pictos_pred)
            tokens.append(tokens_pred)
            ref_tok, pic_ref, ref_text = get_ref_tokens_v2(row['clips'], data_ref)
            ref_tokens.append(ref_tok)
            ref_pictos.append(pic_ref)
            ref_texts.append(ref_text)
        df_by_wer.loc[:, "pictos_grammar_tokens"] = ref_tokens
        df_by_wer.loc[:, "tokens"] = tokens
        df_by_wer.loc[:, "pictos_grammar"] = ref_pictos
        df_by_wer.loc[:, "ref_text"] = ref_texts
        df_by_wer.loc[:, "pictos"] = pictos_preds
        final_data = df_by_wer[df_by_wer.pictos_grammar_tokens != 'to_delete']
        bleu_score, wer_score, meteor_score = compute_scores(final_data)
        print_automatic_eval(bleu_score, wer_score, meteor_score)
        html_file(final_data, "out_whisper_large_" + str(identifier) + ".html")
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


def html_file(df, html_file):
    """
        Create the html file with same post-edition.

        Arguments
        ---------
        df: dataframe
        html_file: file
    """
    html = create_html_file(html_file)
    html.write("<div class = \"container-fluid\">")
    write_html_file(df, html)
    html.write("</div></body></html>")
    html.close()


def write_html_file(df, html_file):
    """
        Add to the html file post-edition that are different between annotators.

        Arguments
        ---------
        df: dataframe
        html_file: file
    """
    for i, row in df.iterrows():
        if not row['pictos'] != row['pictos']:
            html_file.write("<div class=\"shadow p-3 mb-5 bg-white rounded\">")
        write_header_info_per_sentence(html_file, "ID: " + row['clips'])
        write_header_info_per_sentence(html_file, "Ref: " + row['ref_text'])
        write_header_info_per_sentence(html_file, "Hyp: " + row['hyp'])
        write_header_info_per_sentence(html_file, "WER: " + str(row['wer']))
        html_file.write("<div class=\"container-fluid\">")
        for i, p in enumerate(ast.literal_eval(row['pictos_grammar'])):
            html_file.write(
                "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
                "<img src=\"https://static.arasaac.org/pictograms/" + str(p) + "/" + str(
                    p) + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                       "<figcaption class=\"figure-caption text-center\">Token : " +
                row['pictos_grammar_tokens'].split(' ')[i] + "</figcaption></figure>")
        html_file.write("</div>")
        html_file.write("<div class=\"container-fluid\">")
        for i, p in enumerate(ast.literal_eval(row['pictos'])):
            html_file.write(
                "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
                "<img src=\"https://static.arasaac.org/pictograms/" + str(p) + "/" + str(
                    p) + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                       "<figcaption class=\"figure-caption text-center\">Token: " +
                row['tokens'].split(' ')[i] + "</figcaption></figure>")
        html_file.write("</div>")
        html_file.write("</div>")


# def eval(args):
#     if args.eval_choice == "all":
#         evaluate(args.file_asr_grammar, args.file_ref)
#     elif args.eval_choice == "wer":
#         evaluate_by_wer(args.file_asr, args.file_asr_grammar, args.file_ref)

def eval():
    evaluate_by_wer("/data/macairec/PhD/Grammaire/exps_speech/whisper/out_with_wer/whisper_large_results_with_wer.csv", "/data/macairec/PhD/Grammaire/exps_speech/whisper/out_grammar/whisper_results_large_grammar.csv", "/data/macairec/PhD/Grammaire/corpus/output_grammar/out_with_translation/corpus_all_grammar_pictos.csv")


if __name__ == '__main__':
    eval()

# parser1 = ArgumentParser(description="Evaluate the grammar from ASR predictions.",
#                          formatter_class=RawTextHelpFormatter)
# parser1.add_argument('--eval_choice', type=str, required=True, choices=["all", "wer"],
#                      help="")
# parser1.add_argument('--file_ref', type=str, required=True,
#                      help="")
# parser1.add_argument('--file_asr_grammar', type=str, required=True,
#                      help="")
# parser1.add_argument('--file_asr', type=str, required=False,
#                      help="")
# parser1.set_defaults(func=eval)
# args = parser1.parse_args()
# args.func(args)

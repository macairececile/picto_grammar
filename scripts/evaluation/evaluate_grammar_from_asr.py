#!/usr/bin/python

import math

import numpy as np
import pandas as pd
import evaluate

bleu = evaluate.load("bleu")
meteor = evaluate.load('meteor')
wer = evaluate.load('wer')

pd.options.mode.chained_assignment = None


def read_csv_files(file):
    data = pd.read_csv(file, sep='\t')
    return data


def get_corpus_name(clip_name):
    names = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative", "ofrom", "reunions", "tcof",
             "tufs", "valibel"]
    for n in names:
        if n in clip_name:
            return n


def clean_data_asr(data_asr):
    for i, row in data_asr.iterrows():
        if isinstance(row['text'], float):
            row['tokens'] = ''


def get_ref_pictos_and_tokens(data_asr, data_refs):
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
    row_ref = data_refs.loc[data_refs['clips'] == clip_name]
    if row_ref.empty:
        return "to_delete"
    else:
        return row_ref['tokens'].values[0]


def compute_scores(data_asr, corpus_name=None):
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
    print("-------------------")
    print("| BLEU  | METEOR | WER  |")
    print("|----------------------------------|")
    print("| {:<5.3f} | {:<6.3f} | {:<4.1f} |".format(bleu_score, meteor_score, term_error_rate_score))
    print("-------------------")


def evaluate(file_asr, file_ref):
    data_asr = read_csv_files(file_asr)
    clean_data_asr(data_asr)
    data_refs = read_csv_files(file_ref)
    data_asr = get_ref_pictos_and_tokens(data_asr, data_refs)
    names = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative", "ofrom", "reunions", "tcof",
             "tufs", "valibel"]
    bleu_scores = []
    wer_scores = []
    meteor_scores = []
    # for n in names:
    #     bleu_score, wer_score, meteor_score = compute_scores(data_asr, n)
    #     bleu_scores.append(bleu_score)
    #     wer_scores.append(wer_score)
    #     meteor_scores.append(meteor_score)
    # print("Bleu : ", str(bleu_scores))
    # print("Meteor : ", str(meteor_scores))
    # print("WER : ", str(wer_scores))
    bleu_score, wer_score, meteor_score = compute_scores(data_asr)
    print("Score for all : ")
    print_automatic_eval(bleu_score, wer_score, meteor_score)


def read_data_and_select_from_wer_scores(data_asr, data_asr_grammar, data_ref):
    print("Size all dataset : ", len(data_asr))
    ints = [i for i in range(0, 51, 10)]
    id = 0
    for i in np.arange(0, 0.51, 0.10):
        if id < 5:
            df_by_wer = data_asr[(data_asr['wer'] >= i) & (data_asr['wer'] < i + 0.10)]
            print("Size subdataset : ", len(df_by_wer))
            print("For WER between " + str(ints[id]) + " and " + str(ints[id] + 10) + " : ")
        else:
            df_by_wer = data_asr[data_asr['wer'] >= i]
            print("Size subdataset : ", len(df_by_wer))
            print("For WER sup of  " + str(ints[id]) + " : ")
        ref_tokens = []
        tokens = []
        for i, row in df_by_wer.iterrows():
            infos_grammar = data_asr_grammar.loc[data_asr_grammar['clips'] == row['clips']]
            tokens_pred = infos_grammar["tokens"].values[0]
            tokens.append(tokens_pred)
            ref_tokens.append(get_ref_tokens_v2(row['clips'], data_ref))
        df_by_wer.loc[:, "ref_tokens"] = ref_tokens
        df_by_wer.loc[:, "tokens"] = tokens
        final_data = df_by_wer[df_by_wer.ref_tokens != 'to_delete']
        bleu_score, wer_score, meteor_score = compute_scores(final_data)
        print_automatic_eval(bleu_score, wer_score, meteor_score)
        id += 1


def evaluate_by_wer(file_asr, file_asr_grammar, file_ref):
    data_asr = read_csv_files(file_asr)
    data_asr_grammar = read_csv_files(file_asr_grammar)
    data_refs = read_csv_files(file_ref)
    clean_data_asr(data_asr_grammar)
    read_data_and_select_from_wer_scores(data_asr, data_asr_grammar, data_refs)


if __name__ == '__main__':
    # evaluate("/data/macairec/PhD/Grammaire/exps_speech/whisper/out_grammar/whisper_results_medium_grammar.csv",
    #          "/data/macairec/PhD/Grammaire/corpus/output_grammar/out_with_translation/corpus_all_grammar_pictos.csv")
    evaluate_by_wer("/data/macairec/PhD/Grammaire/exps_speech/whisper/large/whisper_results.csv",
                    "/data/macairec/PhD/Grammaire/exps_speech/whisper/out_grammar/whisper_results_large_grammar.csv",
                    "/data/macairec/PhD/Grammaire/corpus/output_grammar/out_with_translation/corpus_all_grammar_pictos.csv")

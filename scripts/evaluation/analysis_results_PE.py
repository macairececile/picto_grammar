#!/usr/bin/python

import math
import os
from os import listdir
from os.path import isfile, join
from grammar import *
import evaluate

bleu = evaluate.load("bleu")
meteor = evaluate.load('meteor')
wer = evaluate.load('wer')
ter = evaluate.load('ter')

from statistics import mean

from print_sentences_from_grammar import *


def get_json_from_post_edit(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f)) if '.json' in f]
    return files


def read_json_input(json_input_file):
    with open(json_input_file) as f:
        data = json.load(f)
        sentences = [item['sentence'] for item in data]
        pictos = [item['pictos'].split(',') for item in data]
        corpus_name = [item['corpus_name'] for item in data]
        return sentences, pictos, corpus_name


def read_json_output(json_output_file):
    with open(json_output_file) as f:
        data = json.load(f)
        sentence = data["document"]["text"]
        pictos = [p.strip() for p in data['document']['picto'].split(',')[:-1]]
        user = data['document']['user']
        return sentence, pictos, user


def get_input_picto_from_text(text, sentences, pictos, corpus_name, file_post_edit):
    try:
        index = sentences.index(text)
        if pictos[index] == ['']:
            return ["None"], ["None"]
        else:
            return pictos[index], corpus_name[index]
    except:
        print(file_post_edit)
        return []


def get_token_from_id_pictos(pictos, lexique):
    terms = []
    if not pictos == [''] and not pictos == []:
        for p in pictos:
            term = lexique.loc[lexique['id_picto'] == int(p)]["lemma"].tolist()
            if not term:
                terms.append('_')
            else:
                if isinstance(term[0], float):
                    if math.isnan(term[0]) and len(term) >= 2:
                        terms.append('_'.join(term[1].split(' #')[0].split(' ')))
                else:
                    terms.append('_'.join(term[0].split(' #')[0].split(' ')))
    else:
        return ['None']
    return terms


def create_data_for_analysis(file_post_edit, lexique, dataframe, sentences, pictos, corpus_name):
    text, pictos_annot, user = read_json_output(file_post_edit)
    pictos_grammar, name = get_input_picto_from_text(text, sentences, pictos, corpus_name, file_post_edit)
    pictos_grammar_tokens = get_token_from_id_pictos(pictos_grammar, lexique)
    pictos_annot_token = get_token_from_id_pictos(pictos_annot, lexique)
    add_info = {'file': file_post_edit, 'time': os.path.getmtime(file_post_edit), 'user': user, 'text': text,
                'pictos_grammar': pictos_grammar, 'pictos_annot': pictos_annot,
                'pictos_grammar_tokens': pictos_grammar_tokens, 'pictos_annot_token': pictos_annot_token,
                'corpus_name': name}
    dataframe.loc[len(dataframe), :] = add_info


def term_error_rate(dataframe, corpus_name):
    wer_scores = {}
    if corpus_name:
        by_corpus = dataframe.loc[dataframe['corpus_name'] == corpus_name]
        print(by_corpus)
        grouped = by_corpus.groupby('user').groups.values()
    else:
        by_corpus = dataframe
        grouped = by_corpus.groupby('user').groups.values()
    for user in grouped:
        references = []
        predictions = []
        username = by_corpus.loc[user[0]]["user"]
        for i in user:
            references.append(' '.join(by_corpus.loc[i]["pictos_grammar_tokens"]))
            predictions.append(' '.join(by_corpus.loc[i]["pictos_annot_token"]))
        results = wer.compute(predictions=predictions, references=references)
        wer_scores[username] = round(results, 3) * 100
    return wer_scores


def term_bleu(dataframe, corpus_name):
    bleu_scores = {}
    if corpus_name:
        by_corpus = dataframe.loc[dataframe['corpus_name'] == corpus_name]
        grouped = by_corpus.groupby('user').groups.values()
    else:
        by_corpus = dataframe
        grouped = by_corpus.groupby('user').groups.values()
    for user in grouped:
        references = []
        predictions = []
        username = by_corpus.loc[user[0]]["user"]
        for i in user:
            references.append([' '.join(by_corpus.loc[i]["pictos_grammar_tokens"])])
            predictions.append(' '.join(by_corpus.loc[i]["pictos_annot_token"]))
        results = bleu.compute(predictions=predictions, references=references)
        bleu_scores[username] = round(results["bleu"], 3)
    return bleu_scores


def term_meteor(dataframe, corpus_name):
    meteor_scores = {}
    if corpus_name:
        by_corpus = dataframe.loc[dataframe['corpus_name'] == corpus_name]
        grouped = by_corpus.groupby('user').groups.values()
    else:
        by_corpus = dataframe
        grouped = by_corpus.groupby('user').groups.values()
    for user in grouped:
        references = []
        predictions = []
        username = by_corpus.loc[user[0]]["user"]
        for i in user:
            references.append(' '.join(by_corpus.loc[i]["pictos_grammar_tokens"]))
            predictions.append(' '.join(by_corpus.loc[i]["pictos_annot_token"]))
        results = meteor.compute(predictions=predictions, references=references)
        meteor_scores[username] = round(results["meteor"], 3)
    return meteor_scores


def term_ter(dataframe, corpus_name):
    ter_scores = {}
    if corpus_name:
        by_corpus = dataframe.loc[dataframe['corpus_name'] == corpus_name]
        grouped = by_corpus.groupby('user').groups.values()
    else:
        by_corpus = dataframe
        grouped = by_corpus.groupby('user').groups.values()
    for user in grouped:
        references = []
        predictions = []
        username = by_corpus.loc[user[0]]["user"]
        for i in user:
            references.append([' '.join(by_corpus.loc[i]["pictos_grammar_tokens"])])
            predictions.append(' '.join(by_corpus.loc[i]["pictos_annot_token"]))
        results = ter.compute(predictions=predictions, references=references)
        ter_scores[username] = results["score"]
    return ter_scores


def print_automatic_eval(bleu_scores, term_error_rate_score, meteor_score, ter_score):
    print("-------------------")
    print("| User     | BLEU  | METEOR | WER  | TER |")
    print("|----------------------------------|")
    for k, v in bleu_scores.items():
        print("| {:<8s} | {:<5.3f} | {:<6.3f} | {:<4.1f} | {:<5.3f} |".format(k, v, meteor_score[k],
                                                                              term_error_rate_score[k], ter_score[k]))
    print("-------------------")


def score_between_annotators(df, corpus_name):
    references = []
    predictions = []
    if corpus_name:
        by_corpus = df.loc[df['corpus_name'] == corpus_name]
        grouped_df = by_corpus.groupby('text')
    else:
        by_corpus = df
        grouped_df = by_corpus.groupby('text')
    for text, group in grouped_df:
        if group['user'].nunique() > 1:
            add_cecile = False
            add_chloe = False
            for i, row in group.iterrows():
                if row['user'] == "cécile" and not add_cecile:
                    references.append(' '.join(row['pictos_annot_token']))
                    add_cecile = True
                if row['user'] == "chloé" and not add_chloe:
                    predictions.append(' '.join(row['pictos_annot_token']))
                    add_chloe = True
    wer_score = round(wer.compute(predictions=predictions, references=references), 3) * 100
    meteor_score = round(meteor.compute(predictions=predictions, references=references)["meteor"], 3)
    bleu_score = round(bleu.compute(predictions=predictions, references=[[i] for i in references])["bleu"], 3)
    ter_score = ter.compute(predictions=predictions, references=[[i] for i in references])["score"]
    print("-------------------")
    print("| BLEU  | METEOR | WER  | TER |")
    print("|----------------------------------|")
    print("| {:<5.3f} | {:<6.3f} | {:<4.1f} | {:<5.3f} |".format(bleu_score, meteor_score, wer_score, ter_score))
    print("-------------------")


def inter_annotator_aggrement(df, corpus_name):
    judges_agreed_to_include = 0
    judges_agreed_to_exclude = 0
    cecile_agreed_to_include = 0
    chloe_agreed_to_include = 0
    if corpus_name:
        by_corpus = df.loc[df['corpus_name'] == corpus_name]
        grouped_df = by_corpus.groupby('text')
    else:
        by_corpus = df
        grouped_df = by_corpus.groupby('text')
    for text, group in grouped_df:
        if group['user'].nunique() > 1:
            cecile = ''
            chloe = ''
            ref = ''
            for i, row in group.iterrows():
                ref = ' '.join(row['pictos_grammar_tokens'])
                if row['user'] == "cécile":
                    cecile = ' '.join(row['pictos_annot_token'])
                if row['user'] == "chloé":
                    chloe = ' '.join(row['pictos_annot_token'])
            if ref != '' and cecile != '' and chloe != '':
                if ref == cecile and ref == chloe:
                    judges_agreed_to_include += 1
                elif ref != cecile and ref != chloe:
                    judges_agreed_to_exclude += 1
                elif ref == cecile and ref != chloe:
                    cecile_agreed_to_include += 1
                elif ref == chloe and ref != cecile:
                    chloe_agreed_to_include += 1
    percentage_agreement = (judges_agreed_to_include + judges_agreed_to_exclude) / (
                judges_agreed_to_include + judges_agreed_to_exclude + chloe_agreed_to_include + cecile_agreed_to_include)
    percentage_yes = ((chloe_agreed_to_include + judges_agreed_to_include) / (
                judges_agreed_to_include + judges_agreed_to_exclude + chloe_agreed_to_include + cecile_agreed_to_include)) * (
                                 (cecile_agreed_to_include + judges_agreed_to_include) / (
                                     judges_agreed_to_include + judges_agreed_to_exclude + chloe_agreed_to_include + cecile_agreed_to_include))
    percentage_no = ((cecile_agreed_to_include + judges_agreed_to_exclude) / (
                judges_agreed_to_include + judges_agreed_to_exclude + chloe_agreed_to_include + cecile_agreed_to_include)) * (
                                (chloe_agreed_to_include + judges_agreed_to_exclude) / (
                                    judges_agreed_to_include + judges_agreed_to_exclude + chloe_agreed_to_include + cecile_agreed_to_include))
    p_e = percentage_yes + percentage_no
    p_o = percentage_agreement
    cohen_kappa = round((p_e - p_o) / (1 - p_e),2)
    # print("Include : ", str(judges_agreed_to_include))
    # print("Exclude : ", str(judges_agreed_to_exclude))
    # print("Include cecile : ", str(cecile_agreed_to_include))
    # print("Include chloe : ", str(chloe_agreed_to_include))
    print("Percentage of agreement : ", str(percentage_agreement))
    print("Cohen's Kappa : ", str(cohen_kappa))


def get_different_annotation_html_file(dataframe, html_file):
    grouped_df = dataframe.groupby('text')
    for text, group in grouped_df:
        if group['user'].nunique() > 1:
            group['str'] = group['pictos_annot'].apply(lambda x: str(x))
            if group['str'].nunique() > 1:
                html_file.write("<div class=\"shadow p-3 mb-5 bg-white rounded\">")
                write_header_info_per_sentence(html_file, "Text : " + group["text"].iloc[0])
                html_file.write("<div class=\"container-fluid\">")
                write_reference_picto(html_file, group.iloc[0])
                html_file.write("</div>")
                for i, row in group.iterrows():
                    write_header_info_per_sentence(html_file, "User : " + row["user"])
                    html_file.write("<div class=\"container-fluid\">")
                    write_differences_to_html(html_file, row)
                    html_file.write("</div>")
                html_file.write("</div>")


def get_same_annotation_html_file_no_PE(dataframe, html_file):
    grouped_df = dataframe.groupby('text')
    for text, group in grouped_df:
        if group['user'].nunique() > 1:
            group['str'] = group['pictos_annot'].apply(lambda x: str(x))
            pictos = [group["pictos_grammar"].iloc[0]]
            for i, row in group.iterrows():
                pictos.append(row['pictos_annot'])
            if all(pictos[0] == sublist for sublist in pictos[1:]):
                html_file.write("<div class=\"shadow p-3 mb-5 bg-white rounded\">")
                write_header_info_per_sentence(html_file, "Text : " + group["text"].iloc[0])
                html_file.write("<div class=\"container-fluid\">")
                write_reference_picto(html_file, group.iloc[0])
                html_file.write("</div>")
            html_file.write("</div>")


def create_html_with_differences(dataframe, html_file):
    html = create_html_file(html_file)
    html.write("<div class = \"container-fluid\">")
    get_different_annotation_html_file(dataframe, html)
    html.write("</div></body></html>")
    html.close()


def create_html_with_correct_sentences(df, html_file):
    html = create_html_file(html_file)
    html.write("<div class = \"container-fluid\">")
    get_same_annotation_html_file_no_PE(df, html)
    html.write("</div></body></html>")
    html.close()


def write_differences_to_html(html_file, row):
    for i, p in enumerate(row['pictos_annot']):
        html_file.write(
            "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
            "<img src=\"https://static.arasaac.org/pictograms/" + p + "/" + p + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                                                                              "<figcaption class=\"figure-caption text-center\">Token : " +
            row['pictos_annot_token'][i] + "</figcaption></figure>")


def write_reference_picto(html_file, row):
    for i, p in enumerate(row['pictos_grammar']):
        html_file.write(
            "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
            "<img src=\"https://static.arasaac.org/pictograms/" + p + "/" + p + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                                                                              "<figcaption class=\"figure-caption text-center\">Token : " +
            row['pictos_grammar_tokens'][i] + "</figcaption></figure>")


def analysis(json_folder, json_input, lexique, corpus_name=None):
    html_file = "/data/macairec/PhD/Grammaire/corpus/analysis_PE/orfeo/dif.html"
    html_file_2 = "/data/macairec/PhD/Grammaire/corpus/analysis_PE/orfeo/same.html"
    lexique = read_lexique(lexique)
    json_post_edit_files = get_json_from_post_edit(json_folder)
    df = pd.DataFrame(
        columns=['file', 'time', 'user', 'text', 'pictos_grammar', 'pictos_annot', 'pictos_grammar_tokens',
                 'pictos_annot_token', 'corpus_name'])
    sentences, pictos, names = read_json_input(json_input)
    for f in json_post_edit_files:
        create_data_for_analysis(json_folder + f, lexique, df, sentences, pictos, names)
    df.to_csv("orfeo_PE_data.csv", index=False, header=True, sep='\t')
    # wer_scores = term_error_rate(df, corpus_name)
    # bleu_scores = term_bleu(df, corpus_name)
    # meteor_scores = term_meteor(df, corpus_name)
    # ter_scores = term_ter(df, corpus_name)
    # score_between_annotators(df, corpus_name)
    # print_automatic_eval(bleu_scores, wer_scores, meteor_scores, ter_scores)
    inter_annotator_aggrement(df, corpus_name)
    # create_html_with_differences(df, html_file)
    # create_html_with_correct_sentences(df, html_file_2)


if __name__ == '__main__':
    names = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative", "ofrom", "reunions", "tcof", "tufs", "valibel"]
    # analysis(
    #     "/data/macairec/PhD/Grammaire/corpus/output_jsonPE/orfeo/"+corpus_name+"/",
    #     "/data/macairec/PhD/Grammaire/corpus/json_PE/orfeo/sentences_"+corpus_name+".json",
    #     "/data/macairec/PhD/Grammaire/dico/lexique.csv",
    #     corpus_name)
    for i in range(0,12):
        corpus_name = names[i]
        print("Corpus name : ", corpus_name)
        analysis(
            "/data/macairec/PhD/Grammaire/corpus/output_jsonPE/orfeo/all_output_orfeo/",
            "/data/macairec/PhD/Grammaire/corpus/json_PE/orfeo/sentences_orfeo.json",
            "/data/macairec/PhD/Grammaire/dico/lexique.csv",
            corpus_name)
        print("--------")


"""
Script to analysis the Post-edition results.

Example of use: python analysis_results_PE.py --json_PE_dir '/outputPE/' --json_init 'sentences_to_PE.json'
--lexicon 'lexique.csv'
"""

from print_sentences_from_grammar import *
import math
import os
from os import listdir
from os.path import isfile, join
import evaluate
import json
from argparse import ArgumentParser, RawTextHelpFormatter

bleu = evaluate.load("bleu")
meteor = evaluate.load('meteor')
wer = evaluate.load('wer')
ter = evaluate.load('ter')
from jiwer import compute_measures
from sacrebleu.metrics import lib_ter


def get_json_from_post_edit(folder):
    """
        Get json files from a directory.

        Arguments
        ---------
        folder: str

        Returns
        -------
        The list of json files.
    """
    files = [f for f in listdir(folder) if isfile(join(folder, f)) if '.json' in f]
    return files


def read_json_input(json_input_file):
    """
        Read the content of the json file.

        Arguments
        ---------
        json_input_file: str

        Returns
        -------
        The sentences, pictos, and corpus_name.
    """
    with open(json_input_file) as f:
        data = json.load(f)
        sentences = [item['sentence'] for item in data]
        pictos = [item['pictos'].split(',') for item in data]
        corpus_name = [item['corpus_name'] for item in data]
        return sentences, pictos, corpus_name


def read_json_output(json_output_file):
    """
        Read the content of the json file generated after the post-edition.

        Arguments
        ---------
        json_output_file: str

        Returns
        -------
        The sentences, pictos, and user name.
    """
    with open(json_output_file) as f:
        data = json.load(f)
        sentence = data["document"]["text"]
        pictos = [p.strip() for p in data['document']['picto'].split(',')[:-1]]
        user = data['document']['user']
        return sentence, pictos, user


def get_input_picto_from_text(text, sentences, pictos, corpus_name, file_post_edit):
    """
        Get the reference picto sequence of the post-edit sentence.

        Arguments
        ---------
        text: str
        sentences: list
        pictos: list
        corpus_name: list
        file_post_edit: str

        Returns
        -------
        The sequence of pictos, and the name of the corpus
    """
    try:
        index = sentences.index(text)
        if pictos[index] == ['']:
            return ["None"], ["None"]
        else:
            return pictos[index], corpus_name[index]
    except:
        print(file_post_edit)
        return []


def get_token_from_id_pictos(pictos, lexicon):
    """
        Get the token associated to each pictogram from the lexicon.

        Arguments
        ---------
        pictos: list
        lexicon: dataframe

        Returns
        -------
        A list with the associated tokens.
    """
    terms = []
    if not pictos == [''] and not pictos == []:
        for p in pictos:
            term = lexicon.loc[lexicon['id_picto'] == int(p)]["lemma"].tolist()
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


def create_data_for_analysis(file_post_edit, lexicon, dataframe, sentences, pictos, corpus_name):
    """
        Gather all the information to evaluate the post-edition.

        Arguments
        ---------
        file_post_edit: str
        lexicon: dataframe
        dataframe: dataframe
        sentences: list
        pictos: list
        corpus_name: list
    """
    text, pictos_annot, user = read_json_output(file_post_edit)
    pictos_grammar, name = get_input_picto_from_text(text, sentences, pictos, corpus_name, file_post_edit)
    pictos_grammar_tokens = get_token_from_id_pictos(pictos_grammar, lexicon)
    pictos_annot_token = get_token_from_id_pictos(pictos_annot, lexicon)
    add_info = {'file': file_post_edit, 'time': os.path.getmtime(file_post_edit), 'user': user, 'text': text,
                'pictos_grammar': pictos_grammar, 'pictos_annot': pictos_annot,
                'pictos_grammar_tokens': pictos_grammar_tokens, 'pictos_annot_token': pictos_annot_token,
                'corpus_name': name}
    dataframe.loc[len(dataframe), :] = add_info


def term_ter(dataframe, corpus_name):
    """
        Calculate the Picto Error Rate (PER) between the two annotators.

        Arguments
        ---------
        dataframe: dataframe
        corpus_name: str

        Returns
        -------
        The list of PER scores for each post-edit sentence.
    """
    ter_scores = {}
    edits = {}
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
        subs, delet, inser, shift = 0, 0, 0, 0
        for i in user:
            references.append([' '.join(by_corpus.loc[i]["pictos_grammar_tokens"])])
            predictions.append(' '.join(by_corpus.loc[i]["pictos_annot_token"]))
        results = ter.compute(predictions=predictions, references=references)
        for prediction, reference in zip(predictions, references):
            measures = compute_measures(reference, prediction)
            subs += measures["substitutions"]
            delet += measures["deletions"]
            inser += measures["insertions"]
            shift += lib_ter.translation_edit_rate2(prediction[0].split(" "), reference[0].split(" "))
        if username in edits.keys():
            edits[username][0] += subs
            edits[username][1] += delet
            edits[username][2] += inser
            edits[username][3] += shift
        else:
            edits[username] = [subs, delet, inser, shift]
        ter_scores[username] = results["score"]
    print(edits)
    return ter_scores


def print_automatic_eval(ter_score):
    """
        Print the evaluation.

        Arguments
        ---------
        ter_score: dict
    """
    print("-------------------")
    print("| User     | TER |")
    print("|----------------------------------|")
    for k, v in ter_score.items():
        print("| {:<8s} | {:<5.3f} |".format(k, v, ter_score[k]))
    print("-------------------")


def score_between_annotators(df, corpus_name):
    """
        Print the evaluation between annotators.

        Arguments
        ---------
        df: dataframe
        corpus_name: str
    """
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
            add_expert1 = False
            add_expert2 = False
            for i, row in group.iterrows():
                if row['user'] == "1" and not add_expert1:
                    references.append(' '.join(row['pictos_annot_token']))
                    add_expert1 = True
                if row['user'] == "2" and not add_expert2:
                    predictions.append(' '.join(row['pictos_annot_token']))
                    add_expert2 = True
    ter_score = ter.compute(predictions=predictions, references=[[i] for i in references])["score"]
    print("-------------------")
    print("| TER |")
    print("|-----|")
    print("| {:<5.3f} |".format(ter_score))
    print("-------------------")


def inter_annotator_aggrement(df, corpus_name):
    """
        Calculate the inter-annotator agreement.

        Arguments
        ---------
        df: dataframe
        corpus_name: str
    """
    judges_agreed_to_include = 0
    judges_agreed_to_exclude = 0
    expert1_agreed_to_include = 0
    expert2_agreed_to_include = 0
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
                    expert1_agreed_to_include += 1
                elif ref == chloe and ref != cecile:
                    expert2_agreed_to_include += 1
    percentage_agreement = (judges_agreed_to_include + judges_agreed_to_exclude) / (
            judges_agreed_to_include + judges_agreed_to_exclude + expert2_agreed_to_include + expert1_agreed_to_include)
    percentage_yes = ((expert2_agreed_to_include + judges_agreed_to_include) / (
            judges_agreed_to_include + judges_agreed_to_exclude + expert2_agreed_to_include + expert1_agreed_to_include)) * (
                             (expert1_agreed_to_include + judges_agreed_to_include) / (
                             judges_agreed_to_include + judges_agreed_to_exclude + expert2_agreed_to_include + expert1_agreed_to_include))
    percentage_no = ((expert1_agreed_to_include + judges_agreed_to_exclude) / (
            judges_agreed_to_include + judges_agreed_to_exclude + expert2_agreed_to_include + expert1_agreed_to_include)) * (
                            (expert2_agreed_to_include + judges_agreed_to_exclude) / (
                            judges_agreed_to_include + judges_agreed_to_exclude + expert2_agreed_to_include + expert1_agreed_to_include))
    p_e = percentage_yes + percentage_no
    p_o = percentage_agreement
    cohen_kappa = round((p_e - p_o) / (1 - p_e), 2)
    print("Percentage of agreement : ", str(percentage_agreement))
    print("Cohen's Kappa : ", str(cohen_kappa))


def get_different_annotation_html_file(dataframe, html_file):
    """
        Add to the html file post-edition that are different between annotators.

        Arguments
        ---------
        dataframe: dataframe
        html_file: file
    """
    grouped_df = dataframe.groupby('text')
    for text, group in grouped_df:
        if group['user'].nunique() > 1:
            group['str'] = group['pictos_annot'].apply(lambda x: str(x))
            if group['str'].nunique() > 1:
                html_file.write("<div class=\"shadow p-3 mb-5 bg-white rounded\">")
                write_header_info_per_sentence(html_file, "Text: " + group["text"].iloc[0])
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
    """
        Add to the html file post-edition that are the same between annotators.

        Arguments
        ---------
        dataframe: dataframe
        html_file: file
    """
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
    """
        Create the html file with different post-edition.

        Arguments
        ---------
        dataframe: dataframe
        html_file: file
    """
    html = create_html_file(html_file)
    html.write("<div class = \"container-fluid\">")
    get_different_annotation_html_file(dataframe, html)
    html.write("</div></body></html>")
    html.close()


def create_html_with_correct_sentences(df, html_file):
    """
        Create the html file with same post-edition.

        Arguments
        ---------
        df: dataframe
        html_file: file
    """
    html = create_html_file(html_file)
    html.write("<div class = \"container-fluid\">")
    get_same_annotation_html_file_no_PE(df, html)
    html.write("</div></body></html>")
    html.close()


def write_differences_to_html(html_file, row):
    """
        Add to the html file the row with different annotations.

        Arguments
        ---------
        html_file: file
        row: dataframe
    """
    for i, p in enumerate(row['pictos_annot']):
        html_file.write(
            "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
            "<img src=\"https://static.arasaac.org/pictograms/" + p + "/" + p + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                                                                              "<figcaption class=\"figure-caption text-center\">Token : " +
            row['pictos_annot_token'][i] + "</figcaption></figure>")


def write_reference_picto(html_file, row):
    """
        Add to the html file the reference sentence.

        Arguments
        ---------
        html_file: file
        row: dataframe
    """
    for i, p in enumerate(row['pictos_grammar']):
        html_file.write(
            "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
            "<img src=\"https://static.arasaac.org/pictograms/" + p + "/" + p + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                                                                              "<figcaption class=\"figure-caption text-center\">Token : " +
            row['pictos_grammar_tokens'][i] + "</figcaption></figure>")


def analysis(json_folder, json_input, lexicon, corpus_name=None):
    """
        Analysis of the post-edition.

        Arguments
        ---------
        json_folder: str
        json_input: str
        lexicon: str
        corpus_name: str
    """
    html_same = "same.html"
    html_dif = "dif.html"
    lexique = read_lexique(lexicon)
    json_post_edit_files = get_json_from_post_edit(json_folder)
    df = pd.DataFrame(
        columns=['file', 'time', 'user', 'text', 'pictos_grammar', 'pictos_annot', 'pictos_grammar_tokens',
                 'pictos_annot_token', 'corpus_name'])
    sentences, pictos, names = read_json_input(json_input)
    for f in json_post_edit_files:
        create_data_for_analysis(json_folder + f, lexique, df, sentences, pictos, names)
    df.to_csv("orfeo_PE_data.csv", index=False, header=True, sep='\t')
    ter_scores = term_ter(df, corpus_name)
    # score_between_annotators(df, corpus_name)
    # print_automatic_eval(ter_scores)
    # inter_annotator_aggrement(df, corpus_name)
    # create_html_with_correct_sentences(df, html_same)
    # create_html_with_differences(df, html_dif)

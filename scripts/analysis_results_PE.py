import math
import os
from os import listdir
from os.path import isfile, join

from grammar import *
import evaluate

bleu = evaluate.load("bleu")
meteor = evaluate.load('meteor')
wer = evaluate.load('wer')

from print_sentences_from_grammar import *


def get_json_from_post_edit(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f)) if '.json' in f]
    return files


def read_json_input(json_input_file):
    with open(json_input_file) as f:
        data = json.load(f)
        sentences = [item['sentence'] for item in data]
        pictos = [item['pictos'].split(',') for item in data]
        return sentences, pictos


def read_json_output(json_output_file):
    with open(json_output_file) as f:
        data = json.load(f)
        sentence = data["document"]["text"]
        pictos = [p.strip() for p in data['document']['picto'].split(',')[:-1]]
        user = data['document']['user']
        return sentence, pictos, user


def get_input_picto_from_text(text, sentences, pictos, file_post_edit):
    index = sentences.index(text)
    return pictos[index]


def get_token_from_id_pictos(pictos, lexique):
    terms = []
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
    return terms


def create_data_for_analysis(file_post_edit, lexique, dataframe, sentences, pictos):
    text, pictos_annot, user = read_json_output(file_post_edit)
    pictos_grammar = get_input_picto_from_text(text, sentences, pictos, file_post_edit)
    pictos_grammar_tokens = get_token_from_id_pictos(pictos_grammar, lexique)
    pictos_annot_token = get_token_from_id_pictos(pictos_annot, lexique)
    add_info = {'file': file_post_edit, 'time': os.path.getmtime(file_post_edit), 'user': user, 'text': text, 'pictos_grammar': pictos_grammar, 'pictos_annot': pictos_annot,
                'pictos_grammar_tokens': pictos_grammar_tokens, 'pictos_annot_token': pictos_annot_token}
    dataframe.loc[len(dataframe), :] = add_info


def term_error_rate(dataframe):
    wer_scores = {}
    grouped = dataframe.groupby('user').groups.values()
    for user in grouped:
        references = []
        predictions = []
        username = dataframe.loc[user[0]]["user"]
        for i in user:
            references.append(' '.join(dataframe.loc[i]["pictos_grammar_tokens"]))
            predictions.append(' '.join(dataframe.loc[i]["pictos_annot_token"]))
        results = wer.compute(predictions=predictions, references=references)
        wer_scores[username] = round(results, 3) * 100
    return wer_scores


def term_bleu(dataframe):
    bleu_scores = {}
    grouped = dataframe.groupby('user').groups.values()
    for user in grouped:
        references = []
        predictions = []
        username = dataframe.loc[user[0]]["user"]
        for i in user:
            references.append([' '.join(dataframe.loc[i]["pictos_grammar_tokens"])])
            predictions.append(' '.join(dataframe.loc[i]["pictos_annot_token"]))
        results = bleu.compute(predictions=predictions, references=references)
        bleu_scores[username] = round(results["bleu"], 3)
    return bleu_scores


def term_meteor(dataframe):
    meteor_scores = {}
    grouped = dataframe.groupby('user').groups.values()
    for user in grouped:
        references = []
        predictions = []
        username = dataframe.loc[user[0]]["user"]
        for i in user:
            references.append(' '.join(dataframe.loc[i]["pictos_grammar_tokens"]))
            predictions.append(' '.join(dataframe.loc[i]["pictos_annot_token"]))
        results = meteor.compute(predictions=predictions, references=references)
        meteor_scores[username] = round(results["meteor"], 3)
    return meteor_scores


def print_automatic_eval(bleu_scores, term_error_rate_score, meteor_score):
    print("-------------------")
    print("| User     | BLEU  | METEOR | WER  |")
    print("|----------------------------------|")
    for k, v in bleu_scores.items():
        print("| {:<8s} | {:<5.3f} | {:<6.3f} | {:<4.1f} |".format(k, v, meteor_score[k], term_error_rate_score[k]))
    print("-------------------")


def get_different_annotation_html_file(dataframe, html_file):
    grouped_df = dataframe.groupby('text')
    for text, group in grouped_df:
        if group['user'].nunique() > 1:
            group['str'] = group['pictos_annot'].apply(lambda x: str(x))
            if group['str'].nunique() > 1:
                html_file.write("<div class=\"shadow p-3 mb-5 bg-white rounded\">")
                write_header_info_per_sentence(html_file, "Text : " + group["text"].iloc[0])
                for i, row in group.iterrows():
                    write_header_info_per_sentence(html_file, "User : " + row["user"])
                    html_file.write("<div class=\"container px-4\">")
                    write_differences_to_html(html_file, row)
                    html_file.write("</div>")
                html_file.write("</div>")


def create_html_with_differences(dataframe, html_file):
    html = create_html_file(html_file)
    html.write("<div class = \"container\">")
    get_different_annotation_html_file(dataframe, html)
    html.write("</div></body></html>")
    html.close()


def write_differences_to_html(html_file, row):
    for i, p in enumerate(row['pictos_annot']):
        html_file.write(
            "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
            "<img src=\"/data/macairec/Cloud/PROPICTO_RESSOURCES/ARASAAC/ARASAAC_Pictos_All/" + p + ".png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                                                                                             "<figcaption class=\"figure-caption text-center\">Token : " +
            row['pictos_annot_token'][i] + "</figcaption></figure>")


def analysis(json_folder, json_input, lexique):
    html_file = "/data/macairec/PhD/Grammaire/picto_grammar/scripts/differences_PE.html"
    lexique = read_lexique(lexique)
    json_post_edit_files = get_json_from_post_edit(json_folder)
    df = pd.DataFrame(
        columns=['file', 'time', 'user', 'text', 'pictos_grammar', 'pictos_annot', 'pictos_grammar_tokens', 'pictos_annot_token'])
    sentences, pictos = read_json_input(json_input)
    for f in json_post_edit_files:
        create_data_for_analysis(json_folder + f, lexique, df, sentences, pictos)
    bleu_scores = term_bleu(df)
    meteor_scores = term_meteor(df)
    wer_scores = term_error_rate(df)
    print_automatic_eval(bleu_scores, wer_scores, meteor_scores)
    create_html_with_differences(df, html_file)



if __name__ == '__main__':
    analysis("/data/macairec/PhD/Grammaire/corpus/output_jsonPE/requests_all/",
             "/data/macairec/PhD/Grammaire/corpus/json_PE/sentences.json", "/data/macairec/PhD/Grammaire/dico/lexique.csv")

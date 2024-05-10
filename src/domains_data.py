import ast
import numpy as np

from collections import Counter
from print_sentences_from_grammar import *


def read_data(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t')
    return df


def group_by_category(df):
    return df.groupby('category')


def html_file(df, ids, html_file):
    """
        Create the html file with same post-edition.

        Arguments
        ---------
        df: dataframe
        html_file: file
    """
    html = create_html_file(html_file)
    html.write("<div class = \"container-fluid\">")
    write_html_file(df, ids, html)
    html.write("</div></body></html>")
    html.close()


def write_html_file(df, segments, html):
    """
        Add to the html file post-edition that are different between annotators.

        Arguments
        ---------
        df: dataframe
        html_file: file
    """
    for s in segments:
        info_seg = df.iloc[s]
        html.write("<div class=\"shadow p-3 mb-5 bg-white rounded\">")
        write_header_info_per_sentence(html, "Texte: " + info_seg["text"])
        write_header_info_per_sentence(html, "Terme à évaluer: <strong>" + info_seg["terme"] + "</strong>")
        html.write("<div class=\"container-fluid\">")
        for a, p in enumerate(ast.literal_eval(info_seg["pictos"])):
            html.write(
                "<span style=\"color: #000080;\"><figure class=\"figure\">"
                "<img src=\"https://static.arasaac.org/pictograms/" + str(p) + "/" + str(
                    p) + "_2500.png" + "\"alt=\"\" width=\"110\" height=\"110\" />"
                                       "<figcaption class=\"figure-caption text-center\">" +
                info_seg["tokens"].split(' ')[a] + "</figcaption></figure>")
        html.write("</div>")
        html.write("</div>")


def generate_html(data, categories):
    for k, v in categories.groups.items():
        html_file(data, v, k + ".html")


def get_words_to_check_per_category(categories):
    for c in categories.groups.keys():
        keywords = categories.get_group(c)["terme"]
        # print(keywords.tolist())
        print(c)
        print([item for item, count in Counter(keywords).items() if count > 1])


##### EVAL ANNOT DOMAIN #####
def get_data_eval_per_category(data_cecile, data_chloe):
    cats = group_by_category(data_cecile)
    for c in cats.groups.keys():
        print(c)
        evals = []
        terms_with_non = []
        data_chloe_cat = data_chloe.loc[data_chloe['category'] == c]
        data_cecile_cat = data_cecile.loc[data_cecile['category'] == c]
        for i, el in enumerate(data_cecile_cat["eval"]):
            el_chloe = data_chloe_cat["eval"].tolist()[i]
            evals.append([el, el_chloe])
            if 3 in (el, el_chloe):
                terms_with_non.append(data_chloe_cat["terme"].tolist()[i])
        matrix = create_matrice(evals)
        calculate_and_print_results(matrix)
        calculate_inter_annotator_agreement(matrix)
        print("Termes avec non : ", terms_with_non)
        print("----------------------------")


def create_matrice(evals):
    matrix = np.zeros((3, 3), dtype=int)

    # Compter les occurrences
    for item in evals:
        exp1, exp2 = item
        matrix[exp1 - 1][exp2 - 1] += 1
    print(matrix)

    # Afficher la matrice
    return matrix


def calculate_and_print_results(matrix):
    termes_oui_oui_mais = matrix[0, 0] + matrix[0, 1] + matrix[1, 0] + matrix[1, 1]
    print("Nombre de termes oui / oui mais:", str(termes_oui_oui_mais) + "%")
    print("Nombre de termes non:", str(100 - termes_oui_oui_mais) + "%")


def calculate_inter_annotator_agreement(matrix):
    total_annotations = np.sum(matrix)

    # Calcul de la probabilité d'accord observée (po)
    po = np.trace(matrix) / total_annotations

    # Calcul de la probabilité d'accord attendue par le hasard (pe)
    row_totals = np.sum(matrix, axis=0)
    column_totals = np.sum(matrix, axis=1)
    pe = np.sum(row_totals * column_totals) / (total_annotations ** 2)

    # Calcul de l'indice de Cohen's Kappa
    kappa = (po - pe) / (1 - pe)

    print("L'indice de Cohen's Kappa est :", kappa)


def main():
    # data = read_data("/data/macairec/PhD/Grammaire/corpus/corpus_v2/domaines/domains_with_pictos.tsv")
    # categories = group_by_category(data)
    # generate_html(data, categories)
    # get_words_to_check_per_category(categories)
    eval_cecile = read_data("/data/macairec/PhD/Grammaire/corpus/corpus_v2/domaines/eval_domain_cecile.tsv")
    eval_chloe = read_data("/data/macairec/PhD/Grammaire/corpus/corpus_v2/domaines/eval_domain_chloé.tsv")
    get_data_eval_per_category(eval_cecile, eval_chloe)


if __name__ == '__main__':
    main()

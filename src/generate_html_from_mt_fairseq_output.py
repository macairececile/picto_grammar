"""
Script to evaluate the grammar translation in pictograms from the ASR predictions.

Example of use:
    python evaluate_grammar_from_asr.py --eval_choice "all" --file_ref "whisper_results_small_grammar.csv"
    --file_asr_grammar "corpus_all_grammar_pictos.csv"
    python evaluate_grammar_from_asr.py --eval_choice "all" --file_ref "whisper_results_small_grammar.csv"
    --file_asr_grammar "corpus_all_grammar_pictos.csv" --file_asr "whisper_results.csv"

"""
from print_sentences_from_grammar import *
import random

orfeo_segments = ["cefc-cfpb-1200-2-1323", "cefc-ofrom-unine08a14m-111", "cefc-tufs-fr13_2005_07_06-772",
                  "cefc-coralrom-ffammn16-17",
                  "cefc-valibel-accCV2r-336", "cefc-crfp-PRI-PRI-2-128", "cefc-tcof-Guadeloupe-159",
                  "cefc-reunions-de-travail-Immobilier_CoDir_GIS_9juin08-2330",
                  "cefc-clapi-montage_meuble-34", "cefc-cfpp-Louise_Liotard_F_85_et_Jeanne_Mallet_F_75_SO-1-1547",
                  "cefc-fleuron-V_Scol_endo_02_P11-6",
                  "cefc-frenchoralnarrative-Kiss_202i-11-12_TITETE_ET_TICORPS-81"]

commonvoice_segments = ["27044313", "17316124", "36373727", "20227204", "19728888", "19711506", "19643456", "25024324",
                        "27035745", "19965544", "20038744", "21351623"]

orfeo_indices = [208, 554, 2859, 9325, 13506, 16894, 23388, 25661, 27522, 18453, 18176, 10223]

commonvoice_indices = [49, 108, 184, 4150, 4286, 4335, 4361, 11254, 11292, 11509, 11770, 2655]

def read_lexique(lexicon):
    """
        Read the lexicon.

        Arguments
        ---------
        lexicon: str

        Returns
        -------
        The dataframe with the information.
    """
    df = pd.read_csv(lexicon, sep='\t')
    df.loc[:, 'keyword_no_cat'] = df['lemma'].apply(lambda a: "_".join(str(a).split(' #')[0].strip().split(" ")))
    return df


def read_out_mt_file_and_get_info(file):
    infos = {'S': [], 'T': [], 'H': []}

    with open(file, 'r', encoding='utf-8') as f:
        # Remplir le dictionnaire en fonction des données fournies
        for l in f:
            if l.startswith('S'):
                infos['S'].append(l.split('\t', 1)[1][:-1])
            elif l.startswith('T'):
                infos['T'].append(l.split('\t', 1)[1][:-1])
            elif l.startswith('H'):
                infos['H'].append(l.split('\t', 2)[2][:-1])

    if "orfeo" in file:
        selected_indices = orfeo_indices
    elif "commonvoice" in file:
        selected_indices = commonvoice_indices
    else:
        selected_indices = orfeo_indices
    new_infos = {'S': [infos['S'][i-1] for i in selected_indices],
                 'T': [infos['T'][i-1] for i in selected_indices],
                 'H': [infos['H'][i-1] for i in selected_indices]}
    return new_infos


def get_id_picto_from_predicted_lemma(df_lexicon, lemma):
    try:
        id_picto = df_lexicon.loc[df_lexicon['keyword_no_cat'] == lemma]["id_picto"].tolist()[0]
        return id_picto
    except:
        return 0


def get_id_picto_lexicon_from_mt(infos, df_lexicon):
    id_picto_ref = []
    id_picto_hyp = []
    for i, t in enumerate(infos['T']):
        id_picto_ref.append([get_id_picto_from_predicted_lemma(df_lexicon, l) for l in t.split(" ")])
        id_picto_hyp.append([get_id_picto_from_predicted_lemma(df_lexicon, l) for l in infos["H"][i].split(" ")])
    return id_picto_ref, id_picto_hyp


def html_file(infos, id_picto_ref, id_picto_hyp, html_file):
    """
        Create the html file with same post-edition.

        Arguments
        ---------
        df: dataframe
        html_file: file
    """
    html = create_html_file(html_file)
    html.write("<div class = \"container-fluid\">")
    write_html_file(infos, id_picto_ref, id_picto_hyp, html)
    html.write("</div></body></html>")
    html.close()


def write_html_file(infos, id_picto_ref, id_picto_hyp, html):
    """
        Add to the html file post-edition that are different between annotators.

        Arguments
        ---------
        df: dataframe
        html_file: file
    """
    for i, row in enumerate(infos["S"]):
        html.write("<div class=\"shadow p-3 mb-5 bg-white rounded\">")
        write_header_info_per_sentence(html, "Source: " + row)
        write_header_info_per_sentence(html, "Ref: " + infos["T"][i])
        write_header_info_per_sentence(html, "Hyp: " + infos["H"][i])
        html.write("<div class=\"container-fluid\">")
        for a, p in enumerate(id_picto_ref[i]):
            html.write(
                "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
                "<img src=\"https://static.arasaac.org/pictograms/" + str(p) + "/" + str(
                    p) + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                       "<figcaption class=\"figure-caption text-center\">Token : " +
                infos["T"][i].split(' ')[a] + "</figcaption></figure>")
        html.write("</div>")
        html.write("<div class=\"container-fluid\">")
        for a, p in enumerate(id_picto_hyp[i]):
            html.write(
                "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
                "<img src=\"https://static.arasaac.org/pictograms/" + str(p) + "/" + str(
                    p) + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                       "<figcaption class=\"figure-caption text-center\">Token: " +
                infos["H"][i].split(' ')[a] + "</figcaption></figure>")
        html.write("</div>")
        html.write("</div>")


def generate_html():
    lexicon = read_lexique("/data/macairec/PhD/Grammaire/dico/lexique.csv")
    infos = read_out_mt_file_and_get_info("/data/macairec/PhD/Grammaire/commonvoice.txt")
    id_picto_ref, id_picto_hyp = get_id_picto_lexicon_from_mt(infos, lexicon)
    html_file(infos, id_picto_ref, id_picto_hyp, "test_commonvoice.html")


if __name__ == '__main__':
    generate_html()

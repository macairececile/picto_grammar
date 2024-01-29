"""
Script to evaluate the grammar translation in pictograms from the ASR predictions.

Example of use:
    python evaluate_grammar_from_asr.py --eval_choice "all" --file_ref "whisper_results_small_grammar.csv"
    --file_asr_grammar "corpus_all_grammar_pictos.csv"
    python evaluate_grammar_from_asr.py --eval_choice "all" --file_ref "whisper_results_small_grammar.csv"
    --file_asr_grammar "corpus_all_grammar_pictos.csv" --file_asr "whisper_results.csv"

"""
import ast

from print_sentences_from_grammar import *
import random

orfeo_segments_2 = ["cefc-cfpb-1200-2-1185", "cefc-ofrom-unine11c05m-39", "cefc-tufs-14HCMJ110913-2764", "cefc-coralrom-fmedsc02-9",
                    "cefc-frenchoralnarrative-Bizouerne_039-8_CELUI_QUI_NE_VEUT_PAS_MOURIR_1-197", "cefc-valibel-ilcDA1r-246",
                    "cefc-crfp-PRI-BEL-2-223", "cefc-fleuron-V_Defle_endo_05_P4-11", "cefc-cfpp-Raphael_Lariviere_H_23_7e-767", "cefc-tcof-Etudesmedecine_sim-193",
                    "cefc-reunions-de-travail-OF1_SeanceTravail_4dec07-2270", "cefc-clapi-reunion_conception_mosaic_architecture-2036"]


commonvoice_segments_2 = ["common_voice_fr_30417468", "common_voice_fr_22783960", "common_voice_fr_27026911", "common_voice_fr_20042434",
                          "common_voice_fr_19751859", "common_voice_fr_19950489", "common_voice_fr_19718800", "common_voice_fr_22790625",
                          "common_voice_fr_19754281", "common_voice_fr_23067169", "common_voice_fr_37524250", "common_voice_fr_27216383"]

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


# def get_infos_from_s2p_data(file_s2p_ref):
#     infos_all = {"ID": [], "REF": [], "HYP_ASR": [], "REF_PICTO": [], "HYP_PICTO": []}
#
#     df = pd.read_csv(file_s2p_ref, sep='\t')
#
#     for i, row in df.iterrows():
#         infos_all["ID"].append(row["clips"])
#         infos_all["REF"].append(row["text"])
#         infos_all["REF_PICTO"].append(row["tgt"])
#     return infos_all


def read_out_mt_file_and_get_info(file, file_s2p_input, df_lexicon):
    infos = {"index": [], 'S': [], 'T': [], 'H': []}

    with open(file, 'r', encoding='utf-8') as f:
        # Remplir le dictionnaire en fonction des données fournies
        for l in f:
            if l.startswith('S'):
                infos['S'].append(l.split('\t', 1)[1][:-1])
            elif l.startswith('T'):
                infos['T'].append(l.split('\t', 1)[1][:-1])
            elif l.startswith('H'):
                infos['H'].append(l.split('\t', 2)[2][:-1])
                infos["index"].append(int(l.split("H-")[1].split("\t")[0]))
    data = pd.read_csv(file_s2p_input, sep="\t")
    if "orfeo" in file:
        subset_df = data[data['clips'].isin(orfeo_segments_2)]
    elif "commonvoice" in file:
        subset_df = data[data['clips'].isin(commonvoice_segments_2)]
    else:
        subset_df = data[data['clips'].isin(orfeo_segments_2)]
    # new_infos = {"ID": [], "REF": [], "REF_TOKENS": [], "REF_PICTOS": [], "HYP_TOKENS": [], "HYP_PICTOS": []}
    new_infos = []
    for i, row in subset_df.iterrows():
        index = infos['index'].index(i)
        inf = [row["clips"], row["text"], row["tokens"], ast.literal_eval(row["pictos"]), infos["H"][index],
               [get_id_picto_from_predicted_lemma(df_lexicon, l) for l in infos["H"][index].split(" ")]]
        new_infos.append(inf)
    return new_infos



def get_id_picto_from_predicted_lemma(df_lexicon, lemma):
    try:
        id_picto = df_lexicon.loc[df_lexicon['keyword_no_cat'] == lemma]["id_picto"].tolist()[0]
        return id_picto
    except:
        return 0


def html_file(segments, html_file):
    """
        Create the html file with same post-edition.

        Arguments
        ---------
        df: dataframe
        html_file: file
    """
    html = create_html_file(html_file)
    html.write("<div class = \"container-fluid\">")
    write_html_file(segments, html)
    html.write("</div></body></html>")
    html.close()

def write_html_file(segments, html):
    """
        Add to the html file post-edition that are different between annotators.

        Arguments
        ---------
        df: dataframe
        html_file: file
    """
    for s in segments:
        html.write("<div class=\"shadow p-3 mb-5 bg-white rounded\">")
        write_header_info_per_sentence(html, "Id: " + s[0])
        write_header_info_per_sentence(html, "Ref text: " + s[1])
        write_header_info_per_sentence(html, "Ref picto: " + s[2])
        write_header_info_per_sentence(html, "Hyp picto: " + s[4])
        html.write("<div class=\"container-fluid\">")
        for a, p in enumerate(s[3]):
            html.write(
                "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
                "<img src=\"https://static.arasaac.org/pictograms/" + str(p) + "/" + str(
                    p) + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                       "<figcaption class=\"figure-caption text-center\">Token : " +
                s[2].split(' ')[a] + "</figcaption></figure>")
        html.write("</div>")
        html.write("<div class=\"container-fluid\">")
        for a, p in enumerate(s[5]):
            html.write(
                "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
                "<img src=\"https://static.arasaac.org/pictograms/" + str(p) + "/" + str(
                    p) + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                       "<figcaption class=\"figure-caption text-center\">Token: " +
                s[4].split(' ')[a] + "</figcaption></figure>")
        html.write("</div>")
        html.write("</div>")


#################### MQM ###################
def create_MQM_file(segments, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["system", "doc", "doc_id", "seg_id", "rater", "source", "target"]
        writer.writerow(field)
        for i, j in enumerate(segments):
            writer.writerow(["grammar", j[0], j[0], j[0], "rater1", j[2], j[4]])
        for i, j in enumerate(segments):
            writer.writerow(["grammar", j[0], j[0], j[0], "rater2", j[2], j[4]])


def generate_html():
    lexicon = read_lexique("/data/macairec/PhD/Grammaire/dico/lexique.csv")
    infos = read_out_mt_file_and_get_info("/data/macairec/PhD/Grammaire/commonvoice.txt", "/data/macairec/PhD/Grammaire/S2P_eval_MQM/test_commonvoice_s2p.csv", lexicon)
    html_file(infos, "test_commonvoice.html")
    create_MQM_file(infos, "test_commonvoice.csv")


if __name__ == '__main__':
    generate_html()

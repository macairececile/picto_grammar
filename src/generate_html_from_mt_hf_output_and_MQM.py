"""
Script to evaluate the grammar translation in pictograms from the ASR predictions.

Example of use:
    python evaluate_grammar_from_asr.py --eval_choice "all" --file_ref "whisper_results_small_grammar.csv"
    --file_asr_grammar "corpus_all_grammar_pictos.csv"
    python evaluate_grammar_from_asr.py --eval_choice "all" --file_ref "whisper_results_small_grammar.csv"
    --file_asr_grammar "corpus_all_grammar_pictos.csv" --file_asr "whisper_results.csv"

"""
import pandas as pd

from print_sentences_from_grammar import *
import ast

orfeo_segments = ["cefc-cfpb-1000-5-705", "cefc-cfpp-Isabelle_Legrand_F_32_Anne-Lies_Simo-Groen_F_30_RO-425",
                  "cefc-clapi-repas_francais-15",
                  "cefc-coralrom-fnatco03-192", "cefc-crfp-PRI-BAY-1-160", "cefc-fleuron-V_Rint_03_P5-65",
                  "cefc-frenchoralnarrative-Bizouerne_062-2_BARBIER-32",
                  "cefc-ofrom-unine08a16m-208", "cefc-reunions-de-travail-OF1_Reunion22Nov07-2449",
                  "cefc-tcof-guitariste-2",
                  "cefc-tufs-14HCMJ110913-2331", "cefc-valibel-ileBH1r-55"]

orfeo_segments_2 = ["cefc-cfpb-1200-2-1185", "cefc-ofrom-unine11c05m-39", "cefc-tufs-14HCMJ110913-2764", "cefc-coralrom-fmedsc02-9",
                    "cefc-frenchoralnarrative-Bizouerne_039-8_CELUI_QUI_NE_VEUT_PAS_MOURIR_1-197", "cefc-valibel-ilcDA1r-246",
                    "cefc-crfp-PRI-BEL-2-223", "cefc-fleuron-V_Defle_endo_05_P4-11", "cefc-cfpp-Raphael_Lariviere_H_23_7e-767", "cefc-tcof-Etudesmedecine_sim-193",
                    "cefc-reunions-de-travail-OF1_SeanceTravail_4dec07-2270", "cefc-clapi-reunion_conception_mosaic_architecture-2036"]

commonvoice_segments = ["common_voice_fr_17863422", "common_voice_fr_18058692", "common_voice_fr_17394327",
                        "common_voice_fr_37437270", "common_voice_fr_17945155",
                        "common_voice_fr_25960260", "common_voice_fr_27415274", "common_voice_fr_19738135",
                        "common_voice_fr_27041566", "common_voice_fr_19685495",
                        "common_voice_fr_28739512", "common_voice_fr_25327246"]

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


def read_out_mt_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        return [line.split('\t', 1)[0].rstrip() for line in f]


def get_infos_from_test_data(file_s2p_ref, file_asr_input):
    infos_all = {"ID": [], "REF": [], "HYP_ASR": [], "REF_PICTO": [], "HYP_PICTO": []}

    df = pd.read_csv(file_s2p_ref, sep='\t')
    df2 = pd.read_csv(file_asr_input, sep='\t')

    for i, row in df.iterrows():
        infos_all["ID"].append(row["clips"])
        infos_all["REF"].append(row["text"])
        infos_all["REF_PICTO"].append(row["tgt"])
        infos_all["HYP_ASR"].append(ast.literal_eval(df2.loc[i, 'translation'])["fr"])
    return infos_all


def add_mt_hyp_to_infos(infos_all, out_mt):
    for hyp_mt in out_mt:
        infos_all["HYP_PICTO"].append(hyp_mt)


def select_sentences_to_evaluate(infos_all, segments):
    return [
        [s, infos_all["REF"][infos_all["ID"].index(s)], infos_all["HYP_ASR"][infos_all["ID"].index(s)],
         infos_all["REF_PICTO"][infos_all["ID"].index(s)], infos_all["HYP_PICTO"][infos_all["ID"].index(s)]]
        for s in segments if s in infos_all["ID"]
    ]


def get_id_picto_from_lemma(df_lexicon, lemma):
    try:
        id_picto = df_lexicon.loc[df_lexicon['keyword_no_cat'] == lemma]["id_picto"].tolist()[0]
        return id_picto
    except:
        return 0


def get_id_picto_lexicon_from_mt(segments, df_lexicon):
    for i in segments:
        i.append([get_id_picto_from_lemma(df_lexicon, l) for l in i[3].split(" ")])
        i.append([get_id_picto_from_lemma(df_lexicon, l) for l in i[4].split(" ")])


def merge_ref_and_hyp_infos_and_select_sentences_to_eval(file_mt, file_s2p_ref, file_asr_input, segments, lexicon):
    out_mt = read_out_mt_file(file_mt)
    infos_all = get_infos_from_test_data(file_s2p_ref, file_asr_input)
    add_mt_hyp_to_infos(infos_all, out_mt)
    segments_to_eval = select_sentences_to_evaluate(infos_all, segments)
    get_id_picto_lexicon_from_mt(segments_to_eval, lexicon)
    return segments_to_eval



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
        write_header_info_per_sentence(html, "Hyp ASR: " + s[2])
        write_header_info_per_sentence(html, "Ref picto: " + s[3])
        write_header_info_per_sentence(html, "Hyp picto: " + s[4])
        html.write("<div class=\"container-fluid\">")
        for a, p in enumerate(s[5]):
            html.write(
                "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
                "<img src=\"https://static.arasaac.org/pictograms/" + str(p) + "/" + str(
                    p) + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                       "<figcaption class=\"figure-caption text-center\">Token : " +
                s[3].split(' ')[a] + "</figcaption></figure>")
        html.write("</div>")
        html.write("<div class=\"container-fluid\">")
        for a, p in enumerate(s[6]):
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
            writer.writerow(["grammar", j[0], j[0], j[0], "rater1", j[3], j[4]])
        for i, j in enumerate(segments):
            writer.writerow(["grammar", j[0], j[0], j[0], "rater2", j[3], j[4]])


def generate_html():
    lexicon = read_lexique("/data/macairec/PhD/Grammaire/dico/lexique.csv")
    infos_to_eval = merge_ref_and_hyp_infos_and_select_sentences_to_eval(
        "/data/macairec/PhD/Grammaire/S2P_eval_MQM/wav2vec2_t5_orfeo.txt",
        "/data/macairec/PhD/Grammaire/S2P_eval_MQM/test_s2p_orfeo.csv",
        "/data/macairec/PhD/Grammaire/S2P_eval_MQM/data_asr/wav2vec2/test_orfeo.csv",
        orfeo_segments_2, lexicon)

    create_MQM_file(infos_to_eval, "/data/macairec/PhD/Grammaire/S2P_eval_MQM/MQM_2/MQM_wav2vec2_t5_orfeo.csv")

    html_file(infos_to_eval, "/data/macairec/PhD/Grammaire/S2P_eval_MQM/MQM_2/wav2vec2_t5_orfeo.html")


def generate_MQM_file_from_out_grammar_asr_merge(file_csv):
    data = pd.read_csv(file_csv, sep="\t")
    subset_df = data[data['clips'].isin(orfeo_segments_2)]
    infos = []
    for i, row in subset_df.iterrows():
        infos.append([row["clips"], row["text_ref"], row["text"], row["ref_tokens"], row["tokens"], ast.literal_eval(row["ref_pictos"]), ast.literal_eval(row["hyp_picto"])])
    create_MQM_file(infos, "/data/macairec/PhD/Grammaire/End2End/MQM/MQM_wav2vec2_grammar_orfeo.csv")

    html_file(infos, "/data/macairec/PhD/Grammaire/End2End/MQM/MQM_wav2vec2_grammar_orfeo.html")



if __name__ == '__main__':
    # generate_html()
    generate_MQM_file_from_out_grammar_asr_merge("/data/macairec/PhD/Grammaire/End2End/grammar_out/test_orfeo_wav2vec2_grammar_merge.csv")

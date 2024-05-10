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

orfeo_segments_final = ['cefc-crfp-PUB-BAY-1-95', 'cefc-reunions-de-travail-CCI_ReunionInterne_15jan08-1318', 'cefc-valibel-jtaBJ1r-156', 'cefc-valibel-chaBP1r-508', 'cefc-cfpp-Youcef_Zerari_H_29_Abdel_Hachim_H_25_SO-3614', 'cefc-ofrom-unine11b03m-70', 'cefc-valibel-ilrLD3r-88', 'cefc-crfp-PRI-POI-2-291', 'cefc-ofrom-unine08a07m-8', 'cefc-frenchoralnarrative-Bizouerne_061-7_LE_VAMPIRE-234', 'cefc-cfpp-Isabelle_Legrand_F_32_Anne-Lies_Simo-Groen_F_30_RO-375', 'cefc-coralrom-fmedrp01-265', 'cefc-valibel-styBM1s-10', 'cefc-clapi-reunion_conception_mosaic_architecture-1245', 'cefc-cfpp-Ozgur_Kilic_H_32_alii_3e-1-1126', 'cefc-crfp-PUB-ORL-1-174', 'cefc-ofrom-unine08a04m-223', 'cefc-tcof-Hen_sai_vin_reunion_08-190', 'cefc-tufs-15_LW_MG_100224-984', 'cefc-tufs-03IAGJ110912-359', 'cefc-clapi-repas_kiwi-1162', 'cefc-crfp-PRI-PNE-3-56', 'cefc-crfp-PRI-GRE-1-125', 'cefc-cfpp-Laurence_Leblond_F_43_Stephanie_Zanotti_F_49_7e-1-781', 'cefc-tufs-Kathy-2011', 'cefc-ofrom-unine11c08m-213', 'cefc-clapi-reunion_conception_mosaic_architecture-165', 'cefc-valibel-ilpBM1r-181', 'cefc-valibel-ilcDA1r-1032', 'cefc-cfpp-Paul_Simo_20_Pierre_Marie-Simo_M_34_18e-184', 'cefc-tcof-groupe_musique-162', 'cefc-valibel-styHC1r-121', 'cefc-valibel-ilrSP1r-994', 'cefc-valibel-ilcBC1r-250', 'cefc-coralrom-ftelpv26-107', 'cefc-coralrom-fnatpd01-116', 'cefc-clapi-commerce_boulangerie_rurale_C21_C40-332', 'cefc-crfp-PRI-BOR-1-164', 'cefc-valibel-accBF1r-527', 'cefc-tcof-Cadeaux_bon_08-228', 'cefc-frenchoralnarrative-Calandry_042-6_LES_TROIS_CHEVEUX_DOR-55', 'cefc-cfpp-Gary_Collard_H_24_20e-2395', 'cefc-reunions-de-travail-OF1_SeanceTravail_4dec07-1397', 'cefc-crfp-PUB-AMI-1-190', 'cefc-reunions-de-travail-OF2_ServiceTechReg_14mars08-519', 'cefc-clapi-commerce_fromagerie-654', 'cefc-crfp-PRI-AMI-2-63', 'cefc-valibel-norBB1r-416', 'cefc-tufs-12_JG_AI_100224-1710', 'cefc-tufs-04_CA_NV_100223-1674', 'cefc-clapi-commerce_boulangerie_rurale_C21_C40-349_2', 'cefc-cfpp-Isabelle_Legrand_F_32_Anne-Lies_Simo-Groen_F_30_RO-1122', 'cefc-crfp-PRI-AUX-1-176', 'cefc-tufs-31_SN_LL_100228-390', 'cefc-coralrom-fnatpd02-37', 'cefc-tufs-fr04_2005_07_04-576', 'cefc-crfp-PRI-ROU-3-237', 'cefc-fleuron-V_Defle_endo_05_P4-18', 'cefc-tufs-05_SB_LZ_100223-967', 'cefc-cfpp-Louise_Liotard_F_85_et_Jeanne_Mallet_F_75_SO-1-134', 'cefc-valibel-styCC1s-464', 'cefc-valibel-ilcDA1r-763', 'cefc-tufs-27_JD_CP_100226-109', 'cefc-crfp-PRI-NCY-2-567', 'cefc-valibel-accBF1r-475', 'cefc-valibel-famRM1r-556', 'cefc-tufs-05_SB_LZ_100223-712', 'cefc-tufs-03_MW_CD_100222-1697_1', 'cefc-crfp-PRI-LAR-2-219', 'cefc-cfpp-Yvette_Audin_F_70_7e-1970', 'cefc-fleuron-V_Rint_exo_05_P8-53', 'cefc-crfp-PRI-BES-1-175', 'cefc-cfpp-Mira_F_88_14e-347', 'cefc-crfp-PRI-NCY-2-472', 'cefc-tufs-16_FB_EL_100224-1874_2', 'cefc-tcof-foot_musique-174', 'cefc-ofrom-unine11b10m-53', 'cefc-ofrom-unine11b18m-238', 'cefc-clapi-aperitif_rupture-150', 'cefc-crfp-PRI-NIC-2-86', 'cefc-frenchoralnarrative-Guillemin_044-3_LE_ROI_LYCAON-18', 'cefc-valibel-debLJ1r-209', 'cefc-tufs-Martha-2271', 'cefc-frenchoralnarrative-Boyer_100-3_TAVERNE_DE_GALWAY-366', 'cefc-ofrom-unine11c13m-70', 'cefc-cfpp-Youcef_Zerari_H_29_Abdel_Hachim_H_25_SO-4289', 'cefc-clapi-aperitif_glasgow-305', 'cefc-crfp-PRI-AMI-1-174', 'cefc-tufs-03IAGJ110912-40', 'cefc-crfp-PRI-POI-1-248', 'cefc-valibel-ilrDM2r-1003', 'cefc-cfpp-Killian_Belamy_H_22_Lucas_Hermano_H_21_KB-524', 'cefc-clapi-reunion_conception_mosaic_architecture-2331', 'cefc-tcof-Sousse_bur-193', 'cefc-tufs-17DCBC110914-14', 'cefc-coralrom-fmedin02-273', 'cefc-crfp-PRI-STR-1-350', 'cefc-valibel-ilcBM1r-160_2', 'cefc-cfpp-Raphael_Lariviere_H_23_7e-560', 'cefc-tufs-fr13_2005_07_06-300']

commonvoice_segments_final = ['common_voice_fr_20709845', 'common_voice_fr_23711269', 'common_voice_fr_19689561', 'common_voice_fr_22982593', 'common_voice_fr_27742252', 'common_voice_fr_37066272', 'common_voice_fr_36390220', 'common_voice_fr_19757856', 'common_voice_fr_19740312', 'common_voice_fr_20064062', 'common_voice_fr_19861273', 'common_voice_fr_17341254', 'common_voice_fr_19646034', 'common_voice_fr_19607909', 'common_voice_fr_20306552', 'common_voice_fr_25024547', 'common_voice_fr_20968624', 'common_voice_fr_20228203', 'common_voice_fr_19650734', 'common_voice_fr_19971746', 'common_voice_fr_19737450', 'common_voice_fr_19715148', 'common_voice_fr_33360922', 'common_voice_fr_36301542', 'common_voice_fr_27079085', 'common_voice_fr_19705952', 'common_voice_fr_21158027', 'common_voice_fr_28799743', 'common_voice_fr_19709504', 'common_voice_fr_27338603', 'common_voice_fr_36887397', 'common_voice_fr_18362890', 'common_voice_fr_19763452', 'common_voice_fr_18640596', 'common_voice_fr_20305284', 'common_voice_fr_18866750', 'common_voice_fr_37977423', 'common_voice_fr_17740394', 'common_voice_fr_32659493', 'common_voice_fr_27519312', 'common_voice_fr_20055927', 'common_voice_fr_25179679', 'common_voice_fr_19740173', 'common_voice_fr_19673680', 'common_voice_fr_19619020', 'common_voice_fr_27326501', 'common_voice_fr_17319950', 'common_voice_fr_27092137', 'common_voice_fr_37751355', 'common_voice_fr_22950166', 'common_voice_fr_19607344', 'common_voice_fr_19627017', 'common_voice_fr_19674336', 'common_voice_fr_18297347', 'common_voice_fr_18996159', 'common_voice_fr_32574004', 'common_voice_fr_20303324', 'common_voice_fr_25025945', 'common_voice_fr_27135026', 'common_voice_fr_28623752', 'common_voice_fr_35323552', 'common_voice_fr_20064844', 'common_voice_fr_24917502', 'common_voice_fr_28798712', 'common_voice_fr_19629318', 'common_voice_fr_19710528', 'common_voice_fr_19008192', 'common_voice_fr_27053658', 'common_voice_fr_17323552', 'common_voice_fr_19139525', 'common_voice_fr_20277418', 'common_voice_fr_19688281', 'common_voice_fr_20313322', 'common_voice_fr_17320280', 'common_voice_fr_18122643', 'common_voice_fr_34983597', 'common_voice_fr_31913481', 'common_voice_fr_24997354', 'common_voice_fr_19598237', 'common_voice_fr_19627983', 'common_voice_fr_27097785', 'common_voice_fr_20238006', 'common_voice_fr_21838667', 'common_voice_fr_17328000', 'common_voice_fr_20293825', 'common_voice_fr_27076965', 'common_voice_fr_25212996', 'common_voice_fr_17319882', 'common_voice_fr_19716816', 'common_voice_fr_26768656', 'common_voice_fr_18171946', 'common_voice_fr_22947349', 'common_voice_fr_19965440', 'common_voice_fr_36935393', 'common_voice_fr_26005144', 'common_voice_fr_26594779', 'common_voice_fr_19966434', 'common_voice_fr_23851166', 'common_voice_fr_19619410', 'common_voice_fr_19990737']

def select_random_segments_test(s2p_ref_ids, num_segments=100):
    import random
    return random.sample(s2p_ref_ids.tolist(), num_segments)


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
        infos_all["REF_PICTO"].append(row["tokens"]) # tokens or tgt
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

def select_sentences_to_evaluate_propicto_eval(infos_all):
    return [
        [j, infos_all["REF"][i], infos_all["HYP_ASR"][i],
         infos_all["REF_PICTO"][i], infos_all["HYP_PICTO"][i]]
        for i, j in enumerate(infos_all["ID"])
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
    # segments = select_random_segments_test(infos_all["ID"])
    # segments_to_eval = select_sentences_to_evaluate(infos_all, segments)
    segments_to_eval = select_sentences_to_evaluate_propicto_eval(infos_all)
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
            if j[3] == j[4]:
                writer.writerow(["grammar", j[0], j[0], j[0], "rater1", j[3], j[4], "No-error", "no-error"])
            else:
                writer.writerow(["grammar", j[0], j[0], j[0], "rater1", j[3], j[4]])


def generate_html():
    lexicon = read_lexique("/data/macairec/PhD/Grammaire/dico/lexique.csv")
    infos_to_eval = merge_ref_and_hyp_infos_and_select_sentences_to_eval(
        "/data/macairec/PhD/Grammaire/End2End/grammar_out/whisper_orfeo_t5_propicto_eval.txt",
        "/data/macairec/PhD/Grammaire/End2End/propicto_eval.csv",
        "/data/macairec/PhD/Grammaire/End2End/propicto_eval_whisper_for_mt.csv",
        commonvoice_segments_final, lexicon)

    create_MQM_file(infos_to_eval, "/data/macairec/PhD/Grammaire/End2End/MQM/MQM_whisper_t5_orfeo_propicto_eval.csv")

    html_file(infos_to_eval, "/data/macairec/PhD/Grammaire/End2End/MQM/MQM_whisper_t5_orfeo_propicto_eval.html")


def generate_MQM_file_from_out_grammar_asr_merge(file_csv):
    data = pd.read_csv(file_csv, sep="\t")
    # segments = select_random_segments_test(data["clips"])
    # print(segments)
    # subset_df = data[data['clips'].isin(commonvoice_segments_final)]
    subset_df = data
    infos = []
    for i, row in subset_df.iterrows():
        infos.append([row["clips"], row["text_ref"], row["text"], row["ref_tokens"], row["tokens"],
                      ast.literal_eval(row["ref_pictos"]), ast.literal_eval(row["hyp_picto"])])
    create_MQM_file(infos, "/data/macairec/PhD/Grammaire/End2End/MQM/MQM_whisper_grammar_propicto_eval.csv")

    html_file(infos, "/data/macairec/PhD/Grammaire/End2End/MQM/MQM_whisper_grammar_propicto_eval.html")


if __name__ == '__main__':
    generate_html()
    # generate_MQM_file_from_out_grammar_asr_merge(
    #     "/data/macairec/PhD/Grammaire/End2End/grammar_out/propicto_eval_whisper_grammar_merge.csv")

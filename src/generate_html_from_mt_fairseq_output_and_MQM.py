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

orfeo_segments_final = ['cefc-crfp-PUB-BAY-1-95', 'cefc-reunions-de-travail-CCI_ReunionInterne_15jan08-1318', 'cefc-valibel-jtaBJ1r-156', 'cefc-valibel-chaBP1r-508', 'cefc-cfpp-Youcef_Zerari_H_29_Abdel_Hachim_H_25_SO-3614', 'cefc-ofrom-unine11b03m-70', 'cefc-valibel-ilrLD3r-88', 'cefc-crfp-PRI-POI-2-291', 'cefc-ofrom-unine08a07m-8', 'cefc-frenchoralnarrative-Bizouerne_061-7_LE_VAMPIRE-234', 'cefc-cfpp-Isabelle_Legrand_F_32_Anne-Lies_Simo-Groen_F_30_RO-375', 'cefc-coralrom-fmedrp01-265', 'cefc-valibel-styBM1s-10', 'cefc-clapi-reunion_conception_mosaic_architecture-1245', 'cefc-cfpp-Ozgur_Kilic_H_32_alii_3e-1-1126', 'cefc-crfp-PUB-ORL-1-174', 'cefc-ofrom-unine08a04m-223', 'cefc-tcof-Hen_sai_vin_reunion_08-190', 'cefc-tufs-15_LW_MG_100224-984', 'cefc-tufs-03IAGJ110912-359', 'cefc-clapi-repas_kiwi-1162', 'cefc-crfp-PRI-PNE-3-56', 'cefc-crfp-PRI-GRE-1-125', 'cefc-cfpp-Laurence_Leblond_F_43_Stephanie_Zanotti_F_49_7e-1-781', 'cefc-tufs-Kathy-2011', 'cefc-ofrom-unine11c08m-213', 'cefc-clapi-reunion_conception_mosaic_architecture-165', 'cefc-valibel-ilpBM1r-181', 'cefc-valibel-ilcDA1r-1032', 'cefc-cfpp-Paul_Simo_20_Pierre_Marie-Simo_M_34_18e-184', 'cefc-tcof-groupe_musique-162', 'cefc-valibel-styHC1r-121', 'cefc-valibel-ilrSP1r-994', 'cefc-valibel-ilcBC1r-250', 'cefc-coralrom-ftelpv26-107', 'cefc-coralrom-fnatpd01-116', 'cefc-clapi-commerce_boulangerie_rurale_C21_C40-332', 'cefc-crfp-PRI-BOR-1-164', 'cefc-valibel-accBF1r-527', 'cefc-tcof-Cadeaux_bon_08-228', 'cefc-frenchoralnarrative-Calandry_042-6_LES_TROIS_CHEVEUX_DOR-55', 'cefc-cfpp-Gary_Collard_H_24_20e-2395', 'cefc-reunions-de-travail-OF1_SeanceTravail_4dec07-1397', 'cefc-crfp-PUB-AMI-1-190', 'cefc-reunions-de-travail-OF2_ServiceTechReg_14mars08-519', 'cefc-clapi-commerce_fromagerie-654', 'cefc-crfp-PRI-AMI-2-63', 'cefc-valibel-norBB1r-416', 'cefc-tufs-12_JG_AI_100224-1710', 'cefc-tufs-04_CA_NV_100223-1674', 'cefc-clapi-commerce_boulangerie_rurale_C21_C40-349_2', 'cefc-cfpp-Isabelle_Legrand_F_32_Anne-Lies_Simo-Groen_F_30_RO-1122', 'cefc-crfp-PRI-AUX-1-176', 'cefc-tufs-31_SN_LL_100228-390', 'cefc-coralrom-fnatpd02-37', 'cefc-tufs-fr04_2005_07_04-576', 'cefc-crfp-PRI-ROU-3-237', 'cefc-fleuron-V_Defle_endo_05_P4-18', 'cefc-tufs-05_SB_LZ_100223-967', 'cefc-cfpp-Louise_Liotard_F_85_et_Jeanne_Mallet_F_75_SO-1-134', 'cefc-valibel-styCC1s-464', 'cefc-valibel-ilcDA1r-763', 'cefc-tufs-27_JD_CP_100226-109', 'cefc-crfp-PRI-NCY-2-567', 'cefc-valibel-accBF1r-475', 'cefc-valibel-famRM1r-556', 'cefc-tufs-05_SB_LZ_100223-712', 'cefc-tufs-03_MW_CD_100222-1697_1', 'cefc-crfp-PRI-LAR-2-219', 'cefc-cfpp-Yvette_Audin_F_70_7e-1970', 'cefc-fleuron-V_Rint_exo_05_P8-53', 'cefc-crfp-PRI-BES-1-175', 'cefc-cfpp-Mira_F_88_14e-347', 'cefc-crfp-PRI-NCY-2-472', 'cefc-tufs-16_FB_EL_100224-1874_2', 'cefc-tcof-foot_musique-174', 'cefc-ofrom-unine11b10m-53', 'cefc-ofrom-unine11b18m-238', 'cefc-clapi-aperitif_rupture-150', 'cefc-crfp-PRI-NIC-2-86', 'cefc-frenchoralnarrative-Guillemin_044-3_LE_ROI_LYCAON-18', 'cefc-valibel-debLJ1r-209', 'cefc-tufs-Martha-2271', 'cefc-frenchoralnarrative-Boyer_100-3_TAVERNE_DE_GALWAY-366', 'cefc-ofrom-unine11c13m-70', 'cefc-cfpp-Youcef_Zerari_H_29_Abdel_Hachim_H_25_SO-4289', 'cefc-clapi-aperitif_glasgow-305', 'cefc-crfp-PRI-AMI-1-174', 'cefc-tufs-03IAGJ110912-40', 'cefc-crfp-PRI-POI-1-248', 'cefc-valibel-ilrDM2r-1003', 'cefc-cfpp-Killian_Belamy_H_22_Lucas_Hermano_H_21_KB-524', 'cefc-clapi-reunion_conception_mosaic_architecture-2331', 'cefc-tcof-Sousse_bur-193', 'cefc-tufs-17DCBC110914-14', 'cefc-coralrom-fmedin02-273', 'cefc-crfp-PRI-STR-1-350', 'cefc-valibel-ilcBM1r-160_2', 'cefc-cfpp-Raphael_Lariviere_H_23_7e-560', 'cefc-tufs-fr13_2005_07_06-300']

commonvoice_segments_final = ['common_voice_fr_20709845', 'common_voice_fr_23711269', 'common_voice_fr_19689561', 'common_voice_fr_22982593', 'common_voice_fr_27742252', 'common_voice_fr_37066272', 'common_voice_fr_36390220', 'common_voice_fr_19757856', 'common_voice_fr_19740312', 'common_voice_fr_20064062', 'common_voice_fr_19861273', 'common_voice_fr_17341254', 'common_voice_fr_19646034', 'common_voice_fr_19607909', 'common_voice_fr_20306552', 'common_voice_fr_25024547', 'common_voice_fr_20968624', 'common_voice_fr_20228203', 'common_voice_fr_19650734', 'common_voice_fr_19971746', 'common_voice_fr_19737450', 'common_voice_fr_19715148', 'common_voice_fr_33360922', 'common_voice_fr_36301542', 'common_voice_fr_27079085', 'common_voice_fr_19705952', 'common_voice_fr_21158027', 'common_voice_fr_28799743', 'common_voice_fr_19709504', 'common_voice_fr_27338603', 'common_voice_fr_36887397', 'common_voice_fr_18362890', 'common_voice_fr_19763452', 'common_voice_fr_18640596', 'common_voice_fr_20305284', 'common_voice_fr_18866750', 'common_voice_fr_37977423', 'common_voice_fr_17740394', 'common_voice_fr_32659493', 'common_voice_fr_27519312', 'common_voice_fr_20055927', 'common_voice_fr_25179679', 'common_voice_fr_19740173', 'common_voice_fr_19673680', 'common_voice_fr_19619020', 'common_voice_fr_27326501', 'common_voice_fr_17319950', 'common_voice_fr_27092137', 'common_voice_fr_37751355', 'common_voice_fr_22950166', 'common_voice_fr_19607344', 'common_voice_fr_19627017', 'common_voice_fr_19674336', 'common_voice_fr_18297347', 'common_voice_fr_18996159', 'common_voice_fr_32574004', 'common_voice_fr_20303324', 'common_voice_fr_25025945', 'common_voice_fr_27135026', 'common_voice_fr_28623752', 'common_voice_fr_35323552', 'common_voice_fr_20064844', 'common_voice_fr_24917502', 'common_voice_fr_28798712', 'common_voice_fr_19629318', 'common_voice_fr_19710528', 'common_voice_fr_19008192', 'common_voice_fr_27053658', 'common_voice_fr_17323552', 'common_voice_fr_19139525', 'common_voice_fr_20277418', 'common_voice_fr_19688281', 'common_voice_fr_20313322', 'common_voice_fr_17320280', 'common_voice_fr_18122643', 'common_voice_fr_34983597', 'common_voice_fr_31913481', 'common_voice_fr_24997354', 'common_voice_fr_19598237', 'common_voice_fr_19627983', 'common_voice_fr_27097785', 'common_voice_fr_20238006', 'common_voice_fr_21838667', 'common_voice_fr_17328000', 'common_voice_fr_20293825', 'common_voice_fr_27076965', 'common_voice_fr_25212996', 'common_voice_fr_17319882', 'common_voice_fr_19716816', 'common_voice_fr_26768656', 'common_voice_fr_18171946', 'common_voice_fr_22947349', 'common_voice_fr_19965440', 'common_voice_fr_36935393', 'common_voice_fr_26005144', 'common_voice_fr_26594779', 'common_voice_fr_19966434', 'common_voice_fr_23851166', 'common_voice_fr_19619410', 'common_voice_fr_19990737']

propicto_eval_segments = ["2671806_230406_094947653","2670269_230303_100231408","2670279_230303_100357584","2671542_230405_142431965","2670145_230307_133742181","2670179_230307_134250378","2673152_230425_085235140","2673173_230425_085543896","2676440_230814_162840025","2676437_230814_162727388","2673724_230626_162127985","2672284_230414_224030970","2670093_230307_194439458","2671205_230411_042135827","2671232_230411_042933206","2673066_230421_145518425","2671355_230421_135815401","2672922_230419_171655465","2674419_230512_115510092","2674401_230512_115246079","2673611_230427_165124923","2672512_230408_102620551","2672511_230408_102604535","2673501_230504_140935245","2673530_230504_142013667","2671029_230305_184923107","2671037_230305_185552596","2671490_230407_095059772","2671461_230407_094639558","2671114_230407_111958416","2671134_230407_112339546","2673801_230428_165753954","2673805_230428_165833106","2673249_230426_173948197","2673246_230426_173926537","2671077_230407_084842119","2673579_230427_164421387","2673554_230427_164153797","2672353_230627_091443865","2672349_230627_091352786","2670816_231005_152627610","2670798_231005_152317342","2671935_230407_101016561","2671903_230407_100516834","2671653_230411_151232852","2671168_230406_115000464","2671142_230406_114308065","2670985_230418_142322387","2670418_230306_110155207","2670396_230306_105720982","2670307_230306_104638651","2670318_230306_104853345","2670621_230307_211306452","2670640_230307_211552309","2670353_230306_220715629","2672723_230417_204550574","2671596_230405_171730403","2671599_230405_171754329","2674425_230510_172633509","2673351_230626_211830777","2673368_230626_212226667","2674544_230513_175822242","2674558_230513_180017992","2673421_230426_162140436","2673422_230426_162148152","2674300_230511_233902005","2674293_230511_233657374","2670202_230304_151701680","2670087_230305_170559233","2670503_230303_220353162","2670516_230303_220714476","2670488_230304_155655627","2674006_230502_185307138","2674024_230502_185524384","2671511_230421_150426413","2671537_230421_150803025","2673454_230504_151222504","2676315_230814_155453749","2676327_230814_155236109","2676367_230814_160738223","2676371_230814_160818149","2676214_230807_222836245","2676240_230807_223309865","2676173_230804_184904025","2676184_230804_185219383","2671987_230426_121057373","2672791_230501_001046523","2672742_230430_235838361","2674040_230508_113944654","2673950_230502_144827668","2676146_230728_160857078","2652099_221024_181211327","2652084_221024_180932435","2671312_230406_110844899","2671292_230406_110446059","2673312_230428_154723091","2673294_230428_154502799","2674502_230511_144535489","2671731_230411_180724972","2671737_230411_180931846"]

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
            elif l.startswith('D'):
                infos['H'].append(l.split('\t', 2)[2][:-1])
                infos["index"].append(int(l.split("D-")[1].split("\t")[0]))

    data = pd.read_csv(file_s2p_input, sep="\t")
    subset_df = data[data['clips'].isin(propicto_eval_segments)]
    # if "orfeo" in file:
    #     subset_df = data[data['clips'].isin(orfeo_segments_final)]
    # elif "commonvoice" in file:
    #     subset_df = data[data['clips'].isin(commonvoice_segments_final)]
    # elif "propicto" in file:
    #
    # else:
    #     subset_df = data[data['clips'].isin(orfeo_segments_final)]
    # new_infos = {"ID": [], "REF": [], "REF_TOKENS": [], "REF_PICTOS": [], "HYP_TOKENS": [], "HYP_PICTOS": []}
    new_infos = []
    for i, row in subset_df.iterrows():
        index = infos['index'].index(i)
        inf = [row["clips"], row["text"], row["tgt"], ast.literal_eval(row["pictos"]), infos["H"][index],
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
            if j[2] == j[4]:
                writer.writerow(["grammar", j[0], j[0], j[0], "rater1", j[2], j[4], "No-error", "no-error"])
            else:
                writer.writerow(["grammar", j[0], j[0], j[0], "rater1", j[2], j[4], "", ""])
            # writer.writerow(["grammar", j[0], j[0], j[0], "rater1", j[2], j[4]])
        # for i, j in enumerate(segments):
        #     writer.writerow(["grammar", j[0], j[0], j[0], "rater2", j[2], j[4]])


def generate_html():
    lexicon = read_lexique("/data/macairec/PhD/Grammaire/dico/lexique.csv")
    infos = read_out_mt_file_and_get_info("/data/macairec/PhD/Grammaire/S2P_eval_MQM/MQM_final_eval/Propicto_eval/const_orfeo_propicto_eval.txt", "/data/macairec/PhD/Grammaire/S2P_eval_MQM/MQM_final_eval/Propicto_eval/propicto_eval.csv", lexicon)
    html_file(infos, "const_propicto_eval.html")
    create_MQM_file(infos, "const_propicto_eval.csv")


if __name__ == '__main__':
    generate_html()

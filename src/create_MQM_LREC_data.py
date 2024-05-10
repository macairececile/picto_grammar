import pandas as pd

from print_sentences_from_grammar import *
import ast
import evaluate
corpus = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative", "ofrom", "reunions",
          "tufs", "valibel", "tcof"]

def select_random_segments_test(infos_all, num_segments=100):
    data_mqm = pd.read_csv("/data/macairec/PhD/Grammaire/MQM_eval_LREC/MQM_LREC_tcof.csv", sep=",")["doc"].tolist()
    import random
    segments = []
    refs = []
    hyps = []
    corpus_lines = []
    for v in infos_all["ID"]:
        if "tcof" in v:
            corpus_lines.append(v)
    # segments.extend(random.sample(corpus_lines, 100))
    it = 0
    wer_metric = evaluate.load("wer")
    lines_tcof_to_add = []
    for i, v in enumerate(infos_all["ID"]):
        if v in data_mqm:
            refs.append(infos_all["REF"][i].lower())
            hyps.append(infos_all["HYP_ASR"][i].lower())
        for s in corpus_lines:
            if s == v and s not in data_mqm and it < 55:
                ref = infos_all["REF"][i].lower()
                hyp = infos_all["HYP_ASR"][i].lower()
                wer_score = wer_metric.compute(predictions=[hyp], references=[ref])
                if wer_score > 0.5:
                    refs.append(infos_all["REF"][i].lower())
                    hyps.append(infos_all["HYP_ASR"][i].lower())
                    it += 1
                    lines_tcof_to_add.append(s)
    wer_score = wer_metric.compute(predictions=hyps, references=refs)
    print("WER: ", wer_score)
    return lines_tcof_to_add
    # for c in corpus:
    #     corpus_lines = []
    #     for v in infos_all["ID"]:
    #         if c in v:
    #             corpus_lines.append(v)
    #     segments.extend(random.sample(corpus_lines, num_segments))
    # return segments


def get_infos_from_test_data(file_out_grammar_from_asr, file_ref):
    infos_all = {"ID": [], "REF": [], "HYP_ASR": [], "REF_PICTO": [], "HYP_PICTO": [], "REF_TOKENS": [], "HYP_TOKENS": []}

    df = pd.read_csv(file_out_grammar_from_asr, sep='\t').dropna()
    df2 = pd.read_csv(file_ref, sep='\t')

    for i, row in df.iterrows():
        if row["clips"] in df2["clips"].tolist():
            infos_all["ID"].append(row["clips"])
            infos_all["REF"].append(df2.loc[df2["clips"] == row["clips"]]["text"].values[0])
            infos_all["HYP_ASR"].append(row["text_process"])
            infos_all["REF_TOKENS"].append(df2.loc[df2["clips"] == row["clips"]]["tgt"].values[0])
            infos_all["HYP_PICTO"].append(row["pictos"])
            infos_all["REF_PICTO"].append(df2.loc[df2["clips"] == row["clips"]]["pictos"].values[0])
            infos_all["HYP_TOKENS"].append(row["tokens"])
    return infos_all


def select_sentences_to_evaluate(infos_all, segments):
    return [
        [s, infos_all["REF"][infos_all["ID"].index(s)], infos_all["HYP_ASR"][infos_all["ID"].index(s)],
         infos_all["REF_TOKENS"][infos_all["ID"].index(s)], infos_all["HYP_TOKENS"][infos_all["ID"].index(s)],
         ast.literal_eval(infos_all["REF_PICTO"][infos_all["ID"].index(s)]), ast.literal_eval(infos_all["HYP_PICTO"][infos_all["ID"].index(s)])]
        for s in segments if s in infos_all["ID"]
    ]

def create_MQM_file(segments, csv_file):
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["system", "doc", "doc_id", "seg_id", "rater", "source", "target", "category", "severity"]
        writer.writerow(field)
        for i, j in enumerate(segments):
            if j[3] == j[4]:
                writer.writerow(["grammar", j[0], j[0], j[0], "rater1", j[3], j[4], "No-error", "no-error"])
            else:
                writer.writerow(["grammar", j[0], j[0], j[0], "rater1", j[3], j[4], "", ""])

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


def main(file_out_grammar_from_asr, file_ref):
    infos_all = get_infos_from_test_data(file_out_grammar_from_asr, file_ref)
    segments = select_random_segments_test(infos_all)
    segments_to_eval = select_sentences_to_evaluate(infos_all, segments)
    create_MQM_file(segments_to_eval, "MQM_LREC_modif.csv")
    html_file(segments_to_eval, "MQM_LREC_modif.html")


if __name__ == '__main__':
    main("/data/macairec/PhD/Grammaire/MQM_eval_LREC/whisper_results_large_grammar.csv", "/data/macairec/PhD/Grammaire/MQM_eval_LREC/test_s2p.csv")
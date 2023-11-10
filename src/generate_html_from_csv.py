from print_sentences_from_grammar import *
import ast


def read_csv_file(csv_file):
    data = pd.read_csv(csv_file, sep="\t")
    audios = data["clips"].tolist()
    sentences = data["text_process"].tolist()
    pictos = data["pictos"].tolist()
    pictos_tokens = data["tokens"].tolist()
    return audios, sentences, pictos, pictos_tokens


def html_file(html_file, audios, sentences, pictos, pictos_tokens):
    """
        Create the html file with same post-edition.

        Arguments
        ---------
        df: dataframe
        html_file: file
    """
    html = create_html_file(html_file)
    html.write("<div class = \"container-fluid\">")
    write_html_file(html, audios, sentences, pictos, pictos_tokens)
    html.write("</div></body></html>")
    html.close()


def write_html_file(html_file, audios, sentences, pictos, pictos_tokens):
    """
        Add to the html file post-edition that are different between annotators.

        Arguments
        ---------
        df: dataframe
        html_file: file
    """
    for i, j in enumerate(audios):
        html_file.write("<div class=\"shadow p-3 mb-5 bg-white rounded\">")
        write_header_info_per_sentence(html_file, "Audio: " + j)
        write_header_info_per_sentence(html_file, "Sentence: " + sentences[i])
        picto_sequence = ast.literal_eval(pictos[i])
        if len(pictos_tokens[i].split(' ')) == len(picto_sequence):
            html_file.write("<div class=\"container-fluid\">")
            for a, p in enumerate(picto_sequence):
                html_file.write(
                    "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
                    "<img src=\"https://static.arasaac.org/pictograms/" + str(p) + "/" + str(
                        p) + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                           "<figcaption class=\"figure-caption text-center\">Token : " +
                    pictos_tokens[i].split(' ')[a] + "</figcaption></figure>")
            html_file.write("</div>")
        html_file.write("</div>")


def generate_html_file_from_csv(csv_file):
    audios, sentences, pictos, pictos_tokens = read_csv_file(csv_file)
    name_html = csv_file.split("/")[-1].split('_grammar.csv')[0] + ".html"
    html_file(name_html, audios, sentences, pictos, pictos_tokens)


if __name__ == '__main__':
    generate_html_file_from_csv(
        "/data/macairec/PhD/Grammaire/corpus/output_grammar/commonvoice/test_commonvoice_grammar.csv")

from print_sentences_from_grammar import *
import json


def read_json_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    audios = [d['audio'] for d in data]
    sentences = [d['sentence'] for d in data]
    pictos = [d['pictos'] for d in data]
    pictos_tokens = [d['pictos_tokens'] for d in data]
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
        if len(pictos_tokens[i].split(' ')) == len(pictos[i]):
            html_file.write("<div class=\"container-fluid\">")
            for a, p in enumerate(pictos[i]):
                html_file.write(
                    "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
                    "<img src=\"https://static.arasaac.org/pictograms/" + str(p) + "/" + str(
                        p) + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                           "<figcaption class=\"figure-caption text-center\">Token : " +
                    pictos_tokens[i].split(' ')[a] + "</figcaption></figure>")
            html_file.write("</div>")
        html_file.write("</div>")


def generate_html_file_from_ortolang(json_file):
    audios, sentences, pictos, pictos_tokens = read_json_file(json_file)
    name_html = json_file.split("/")[-1].split('.json')[0] + ".html"
    html_file(name_html, audios, sentences, pictos, pictos_tokens)


if __name__ == '__main__':
    generate_html_file_from_ortolang("/home/getalp/macairec/Bureau/ex.json")
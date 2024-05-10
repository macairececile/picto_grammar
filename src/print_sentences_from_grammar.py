"""
Methods to show the translation in pictograms in a html file.
"""
import pandas as pd

def read_tags(tags):
    return pd.read_csv(tags, sep=",")


def check_tags(tags, ids_picto, html_file):
    violence = []
    sex = []
    for i in ids_picto:
        try:
            sex.append(tags.loc[tags['id_picto'] == i]["sex"].tolist()[0])
            violence.append(tags.loc[tags['id_picto'] == i]["violence"].tolist()[0])
        except:
            print("")
    if True in violence and True in sex:
        html_file.write(
            "<br/>WARNING: Cette séquence inclut des pictogrammes illustrant des scènes de violence et de contenu sexuel.<br/>")
    if True in violence:
        html_file.write(
            "<br/>WARNING: Cette séquence inclut des pictogrammes illustrant des scènes de violence.<br/>")
    if True in sex:
        html_file.write(
            "<br/>WARNING: Cette séquence inclut des pictogrammes illustrant du contenu sexuel.<br/>")


def create_html_file(html_file):
    """
        Create the header of the html file.

        Arguments
        ---------
        html_file: str

        Returns
        -------
        The html file.
    """
    header = "<!doctype html>" \
             "<html lang=\"fr\"><head>" \
             "<meta charset=\"utf-8\">" \
             "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, shrink-to-fit=no\">" \
             "<link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css\" integrity=\"sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T\" crossorigin=\"anonymous\">" \
             "</head>" \
             "<body>"
    body = "<div class=\"jumbotron text-center\">" \
           "<h1>Traduction de phrases en pictogrammes Arasaac &agrave; partir de la parole</h1>" \
           "</div>"
    f = open(html_file, 'w')
    f.write(header)
    f.write(body)
    return f


def write_header_info_per_sentence(html_file, utt_name):
    """
        Write the content of the header for each sentence.

        Arguments
        ---------
        html_file: File
        utt_name: str
    """
    html_file.write("<div class=\"container-fluid\">")
    html_file.write("<div class=\"row\">")
    html_file.write("<div class=\"col-12\"><div class=\"p-2\">")
    html_file.write(utt_name)
    html_file.write("</div></div></div></div>")


def write_translations_per_sentence(html_file, translations, tags):
    """
        Write the pictogram image in a div.

        Arguments
        ---------
        html_file: file
        translations: list
    """
    ids_picto = []
    if translations is not None:
        for id_picto in translations:
            if id_picto.to_picto:
                if id_picto.wsd != '':
                    info_wsd = id_picto.wsd.split(";")[0]
                else:
                    info_wsd = id_picto.wsd
                if id_picto.picto == [404]:
                    html_file.write(
                        "<span style=\"color: #000080;\"><figure class=\"figure\">"
                        "<img src=\"img/" + str(
                            id_picto.picto[0]) + ".png" + "\"alt=\"\" width=\"110\" height=\"110\" />"
                                                          "<figcaption class=\"figure-caption text-center\">Token : " + id_picto.token + "<br/>Lemma : " + str(
                            id_picto.lemma) + "<br/>Pos : " + id_picto.pos + "<br/>WSD : " + str(
                            info_wsd) + "<br/>Id picto : " + str(
                            id_picto.picto) + "</figcaption></figure>")
                else:
                    html_file.write(
                        "<span style=\"color: #000080;\"><figure class=\"figure\">"
                        "<img src=\"https://static.arasaac.org/pictograms/" + str(id_picto.picto[0]) + '/' + str(
                            id_picto.picto[0]) + "_2500.png" + "\"alt=\"\" width=\"110\" height=\"110\" />"
                                                               "<figcaption class=\"figure-caption text-center\">Token : " + id_picto.token + "<br/>Lemma : " + str(
                            id_picto.lemma) + "<br/>Pos : " + id_picto.pos + "<br/>WSD : " + str(
                            info_wsd) + "<br/>Id picto : " + str(
                            id_picto.picto) + "</figcaption></figure>")
                ids_picto.append(id_picto.picto[0])
                html_file.write("&nbsp;&nbsp;")
    check_tags(tags, ids_picto, html_file)


def generate_html(sentence, sentence_grammar, html_file, tags_file):
    """
        Create the html file.

        Arguments
        ---------
        sentence: list
        sentence_grammar: list
        html_file: str
    """
    tags = read_tags(tags_file)
    html = create_html_file(html_file)
    html.write("<div class = \"container-fluid\">")
    for i, s in enumerate(sentence):
        html.write("<div class=\"shadow p-3 mb-5 bg-white rounded\">")
        write_header_info_per_sentence(html, s)

        html.write("<div class=\"container-fluid\">")
        write_translations_per_sentence(html, sentence_grammar[i], tags)
        html.write("</div>")
        html.write("</div>")

    html.write("</div></body></html>")
    html.close()

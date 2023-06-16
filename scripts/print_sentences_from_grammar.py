from grammar import *

def create_html_file(html_file):
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
    html_file.write("<div class=\"container px-4\">")
    html_file.write("<div class=\"row\">")
    html_file.write("<div class=\"col-8\"><div class=\"p-3\">")
    html_file.write(utt_name)
    html_file.write("</div></div></div></div>")


def write_translations_per_sentence(html_file, translations):
    for id_picto in translations:
        if id_picto.to_picto:
            html_file.write(
                "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
                "<img src=\"/data/macairec/Cloud/PROPICTO_RESSOURCES/ARASAAC/ARASAAC_Pictos_All/" + str(
                    id_picto.picto[0]) + ".png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                                  "<figcaption class=\"figure-caption text-center\">" "</figcaption></figure>")
            html_file.write("&nbsp;&nbsp;")


# def write_bottom_info_per_sentence(html_file, ref, hyp, wsd, info_mot, picto_id_predicted):
#     html_file.write("<div class=\"container px-4\">")
#     html_file.write("<div class=\"row\">")
#     html_file.write("<div class=\"col-2\"><div class=\"p-2\">Référence : </div></div>")
#     html_file.write("<div class=\"col-10\"><div class=\"p-2\">" + ref + "</div></div>")
#     html_file.write("<div class=\"col-2\"><div class=\"p-2\"> Hypothèse : </div></div>")
#     html_file.write("<div class=\"col-10\"><div class=\"p-2\">" + hyp + "</div></div>")
#     html_file.write("<div class=\"col-2\"><div class=\"p-2\">WSD : </div></div>")
#     html_file.write("<div class=\"col-10\"><div class=\"p-2\">" + wsd + "</div></div>")
#     html_file.write("</div></div>")
#     html_file.write("<div class=\"container px-4\">")
#     html_file.write("<table class=\"table table-hover\">"
#                     "<thead class=\"table-success\">"
#                     "<tr>"
#                     "<th scope=\"col\">#</th>"
#                     "<th scope=\"col\">Mot lemmatisé / sens wordnet à traduire</th>"
#                     "<th scope=\"col\">ID lemma / ID wsd</th>"
#                     "</tr>"
#                     "</thead>"
#                     "<tbody>"
#                     )
#     for i, lemma_wn in enumerate(info_mot):
#         html_file.write("<tr>")
#         html_file.write("<th scope=\"row\">" + str(i) + "</th>")
#         if lemma_wn.wn and lemma_wn.lemma:
#             html_file.write("<td>" + lemma_wn.lemma + ' / ' + lemma_wn.wn + "</td>")
#         else:
#             html_file.write("<td>" + lemma_wn.lemma + ' / ' + '-' + "</td>")
#         html_file.write(
#             "<td>" + str(picto_id_predicted[i][1]) + ' / ' + str(picto_id_predicted[i][0]) + "</td>")
#         html_file.write("</td></tr>")
#     html_file.write("</tbody></table>")
#
#     html_file.write("</div>")


def print_pictograms(sentence, sentence_grammar):
    html_file = '/data/macairec/PhD/Grammaire/scripts/test.html'
    html = create_html_file(html_file)
    html.write("<div class = \"container\">")
    for i, s in enumerate(sentence):

        html.write("<div class=\"shadow p-3 mb-5 bg-white rounded\">")
        write_header_info_per_sentence(html, s)

        html.write("<div class=\"container px-4\">")
        write_translations_per_sentence(html, sentence_grammar[i])
        html.write("</div>")
        html.write("</div>")

    html.write("</div></body></html>")
    html.close()
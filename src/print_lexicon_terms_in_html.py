import pandas as pd


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
           "<h1>Lexicon terms</h1>" \
           "</div>"
    f = open(html_file, 'w')
    f.write(header)
    f.write(body)
    return f

def print_picto_per_term(html_file, id_picto, lemma, index):
    """
        Write the pictogram image in a div.

        Arguments
        ---------
        html_file: file
        translations: list
    """
    html_file.write(
        "<span style=\"color: #000080;\"><strong><figure class=\"figure\">"
        "<img src=\"https://static.arasaac.org/pictograms/" + str(id_picto) + '/' + str(
            id_picto) + "_2500.png" + "\"alt=\"\" width=\"150\" height=\"150\" />"
                                               "<figcaption class=\"figure-caption text-center\">Num : " + str(index) + "<br/>Lemma : " + lemma + "<br/>Id picto : " + str(
            id_picto) + "</figcaption></figure>")
    html_file.write("&nbsp;&nbsp;")

def print_pictograms(html_file):
    """
        Create the html file.

        Arguments
        ---------
        sentence: list
        sentence_grammar: list
        html_file: str
    """
    lexicon_file = pd.read_csv("/data/macairec/PhD/Grammaire/dico/wolf/wolf_with_words_not_in_lexicon.csv", sep='\t')
    html = create_html_file(html_file)
    html.write("<div class = \"container\">")
    for i, row in lexicon_file.iterrows():
        html.write("<div class=\"shadow p-3 mb-5 bg-white rounded\">")
        html.write("<div class=\"container px-4\">")
        print_picto_per_term(html, row["id_picto"], row["lemma"], int(i)+1)
        html.write("</div>")
        html.write("</div>")

    html.write("</div></body></html>")
    html.close()


if __name__ == '__main__':
    print_pictograms("lexicon_wolf.html")
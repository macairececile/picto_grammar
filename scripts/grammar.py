# python file to create the grammar and generate translation in pictos
import pandas as pd
import spacy
from text_to_num import text2num
from print_sentences_from_grammar import *

phatiques_onomatopees = ['ah', 'aïe', 'areu', 'atchoum', 'badaboum', 'baf', 'bah', 'bam', 'bang', 'bé', 'bêêê', 'beurk',
                         'ben',
                         'bing', 'bon', 'boum', 'broum', 'cataclop', 'clap clap', 'coa coa', 'cocorico', 'coin coin',
                         'crac',
                         'croa croa', 'cuicui', 'ding', 'ding deng dong', 'ding dong', 'dring', 'hé', 'hé ben',
                         'eh bien', 'euh',
                         'flic flac', 'flip flop', 'frou frou', 'glouglou', 'glou glou', 'groin groin', 'grr', 'hé',
                         'hep',
                         'hi han', 'hip hip hip hourra', 'houla', 'hourra', 'hum', 'mêêê', 'meuh', 'miam', 'miam miam',
                         'miaou',
                         'oh', 'O.K.', 'ouah', 'ouah ouah', 'ouf', 'ouh', 'paf', 'pan', 'patatras', 'pchhh', 'pchit',
                         'pff', 'pif-paf', 'pin pon', 'pioupiou', 'plouf', 'pof', 'pouet', 'pouet pouet', 'pouf',
                         'psst', 'ron ron',
                         'schlaf', 'snif', 'splaf', 'splatch', 'sss', 'tacatac', 'tagada', 'tchac', 'teuf teuf',
                         'tic tac', 'toc',
                         'tut tut', 'vlan', 'vroum', 'vrrr', 'wouah', 'zip']

special_characters = ['<', '>', '/', ',', '*', '"']

futur_anterieur = ["aurai", "auras", "aura", 'aurons', "aurez", "auront", "serai", "seras", "sera", "serons", "serez",
                   "seront"]

words_prefix = [
    ['dés', [24753]],
    ['intra', [5439]],
    ['inter', [7765]],
    ['demi', [19570]],
    ['més', [5504]],
    ['post', [32749]],
    ['semi', [19570]],
    ['anti', [7067]],
    ['col', [7064]],
    ['com', [7064]],
    ['con', [7064]],
    ['co', [7064]],
    ['dis', [32394]],
    ['il', [24753]],
    ['im', [24753]],
    ['in', [24753]],
    ['ir', [24753]],
    ['mi', [19570]],
    ['mé', [5504]],
    ['non', [5526]],
    ['per', [8274]],
    ['pré', [32745]],
    ['re', [37163]]
]


# -------------------------------- #
# -------- READ VOC FILES -------- #
# -------------------------------- #
def read_initial_voc_file(file_arasaac):
    df = pd.read_csv(file_arasaac)
    return df[['idpicto', 'lemma', 'lemma_plural']]


def read_dicoPicto(dicoPicto_file):
    df = pd.read_csv(dicoPicto_file, sep=',')
    df.loc[:, 'keyword_no_cat'] = df['keyword'].apply(lambda a: a.split(' #')[0])
    return df


def get_picto_dicoPicto(lemma, dico_picto):
    return list(set([int(a) for a in dico_picto.loc[dico_picto['keyword_no_cat'] == lemma]["id_picto"].tolist()]))


def get_picto_voc(lemma, voc1):
    return list(set([int(a) for a in voc1.loc[voc1['lemma'] == lemma]["idpicto"].tolist()]))


# ---------------------------- #
# -------- CLASS WORD -------- #
# ---------------------------- #
class Word():
    """Representation of a word and its associated information from applied shallow analysis pipeline.

    :param token: Lemma of the word
    :type lemma: list
    :param pos: POS Tag of the word
    :type pos: str
    :param ent_type: Type of the named entity
    :type ent_type: str, Optional
    """

    def __init__(self, token, lemma, pos, morph, dep, ent_type=None, tag=None, wn=None, id_picto=None, prefix=False,
                 plur=False):
        """
        Constructor method
        """
        self.token = token
        self.lemma = lemma
        self.pos = pos
        self.morph = morph
        self.tag = tag
        self.wn = wn
        self.picto = id_picto
        self.to_picto = True
        self.dep = dep
        self.verb_handle = False
        self.prefix = prefix
        self.plur = plur

        # Named Entity
        if (ent_type):
            self.ent_type = ent_type
        else:
            self.ent_type = None

        self.ent_text = None

    def __str__(self):
        """
        Print method
        """

        result = ""

        if (self.token):
            result = f"TOKEN: {self.token}  "

        if (self.lemma):
            result += f"LEMMA: {self.lemma}  "

        if (self.pos):
            result += f"POS: {self.pos}  "

        if (self.ent_type):
            result += f"ENT: {self.ent_type}  "

        result += f"TO_PICTO: {self.to_picto} "

        if self.picto:
            result += f"PICTO: {self.picto} "

        if self.morph:
            result += f"MORPH: {self.morph} "

        if self.dep:
            result += f"DEP: {self.dep} "

        result += f"VERB_HANDLE: {self.verb_handle} "

        if self.prefix:
            result += f"PREFIX: {self.prefix} "

        result += f"PLUR: {self.plur} "

        return result

    def add_wn(self, wn):
        self.wn = wn

    def add_picto(self, picto):
        self.picto = picto


def load_model(model_name):
    # Load the spacy model
    nlp = spacy.load(model_name)
    print("*** Spacy model ready to use : " + model_name + " ***\n")
    return nlp


def process_with_spacy(text, spacy_model):
    doc = spacy_model(text)
    text_spacy = []
    for token in doc:
        if token.lemma_ != " " and token.lemma_ != "'":
            text_spacy.append(
                Word(token.text, [token.lemma_], token.pos_, token.morph, token.dep_, ent_type=token.ent_type_,
                     tag=token.tag_))
    return text_spacy


def process_text(text):
    # remove onomatopées
    for o in phatiques_onomatopees:
        if o in text:
            text = text.replace(o, '')
    # remove special characters
    for c in special_characters:
        if c in text:
            text = text.replace(c, '')
    if not text.strip():
        return True
    else:
        return text


# ------------------ #
# ------ RULES ----- #
# ------------------ #
def imperative_sentence(text_spacy):
    # si le premier mot de la phrase est un verbe - phrase à l'impératif
    if text_spacy[0].pos == 'VERB':
        text_spacy.append(Word('!', ['!'], 'PUNCT', morph='', dep='', id_picto=[3417]))


def interr_excla(text_spacy):
    for w in text_spacy:
        if w.lemma == ["quel"]:
            w.add_picto([22620, 22624])


def add_tense_marker(text_spacy, add_tense_marker, tense):
    if add_tense_marker:
        for i in range(add_tense_marker[0][1], -1, -1):
            # cas ou nsubj précédé d'un déterminant
            if text_spacy[i].dep == "nsubj" and i - 1 >= 0:
                if text_spacy[i - 1].pos == "DET":
                    add_tense_marker[0].extend([i - 1])
                else:
                    add_tense_marker[0].extend([i])
            elif text_spacy[i].dep == "nsubj":
                add_tense_marker[0].extend([i])
        if len(add_tense_marker[0]) >= 3:
            if tense == "past":
                past_marqueur = Word('marqueur_passé', ['marqueur_passé'], 'marqueur_passé', morph='', dep='',
                                     id_picto=[9839])
                text_spacy.insert(add_tense_marker[0][-1], past_marqueur)
            else:
                futur_marqueur = Word('marqueur_futur', ['marqueur_futur'], 'marqueur_futur', morph='', dep='',
                                      id_picto=[9829])
                text_spacy.insert(add_tense_marker[0][-1], futur_marqueur)


def past_tense(text_spacy):
    num_words = len(text_spacy)
    add_past = []
    for i, w in enumerate(text_spacy):
        # case 1 : 3 words
        next = False
        if not w.verb_handle:
            if w.pos == 'AUX' and i + 2 < num_words:
                if text_spacy[i + 2].pos == 'VERB' and text_spacy[i + 2].morph.get("Tense"):
                    if text_spacy[i + 2].morph.get("Tense")[0] == 'Past':
                        w.to_picto = False
                        add_past.append([True, i])
                        next = True
            # case 2 : 2 words
            if w.pos in ['AUX', 'VERB'] and i + 1 < num_words and next == False:
                if text_spacy[i + 1].pos in ['VERB', 'AUX'] and text_spacy[i + 1].morph.get("Tense"):
                    if text_spacy[i + 1].morph.get("Tense")[0] == 'Past':
                        w.to_picto = False
                        add_past.append([True, i])
                        next = True
            # case 3 : 1 word
            if w.pos in ['AUX', 'VERB'] and w.morph.get("Tense") and next == False:
                if w.morph.get("Tense")[0] in ['Past', 'Imp']:
                    add_past.append([True, i])
    # ajout du marqueur du passé
    add_tense_marker(text_spacy, add_past, "past")


def futur_tense(text_spacy):
    add_fut = []
    num_words = len(text_spacy)
    for i, w in enumerate(text_spacy):
        if w.pos in ['AUX', 'VERB'] and w.morph.get("Tense"):
            if w.morph.get("Tense")[0] == "Fut":
                add_fut.append([True, i])
                w.verb_handle = True
        if w.pos in ['AUX', 'VERB'] and i + 1 < num_words:
            if w.token in futur_anterieur and text_spacy[i + 1].pos in ["AUX", "VERB"] and text_spacy[i + 1].morph.get(
                    "Tense"):
                if text_spacy[i + 1].morph.get("Tense")[0] == "Past":
                    w.to_picto = False
                    w.verb_handle = True
                    text_spacy[i + 1].verb_handle = True
                    add_fut.append([True, i])
            if w.lemma == ['aller'] and text_spacy[i + 1].pos in ["AUX", "VERB"]:
                w.to_picto = False
                w.verb_handle = True
                text_spacy[i + 1].verb_handle = True
                add_fut.append([True, i])
    add_tense_marker(text_spacy, add_fut, "futur")


def pronominale(text_spacy):
    pron = False
    for i, w in enumerate(text_spacy):
        if not type(w.morph) == str:
            if w.pos == "PRON" and w.morph.get("Reflex"):
                if w.morph.get("Reflex")[0] == "Yes":
                    w.to_picto = False
                    pron = True
            if w.pos == "VERB" and pron == True:
                w.lemma.insert(0, 'se ' + w.lemma[0])
                pron = False


def nombre(text_spacy):
    for w in text_spacy:
        if w.pos in ["NOUN", "PRON", "DET", "ADJ"]:
            if w.morph.get("Number"):
                if w.morph.get("Number")[0] == "Plur":
                    w.plur = True


def indic_temp(text_spacy):
    nums = []
    actual_group = []
    for i, w in enumerate(text_spacy):
        if w.pos == "NUM":
            actual_group.append(w.token)
            if i == len(text_spacy) - 1 or text_spacy[i + 1].pos != "NUM":
                nums.append([" ".join(actual_group), i])
                actual_group = []
    for el in nums:
        try:
            num_version = text2num(el[0], 'fr')
            text_spacy[el[1]].lemma = [str(num_version)]
            for i in range(el[1] - len(el[0].split()) - 1, el[1]):
                text_spacy[i].to_picto = False
        except:
            print("No conversion in digit for numbers")


def neg(text_spacy):
    for w in text_spacy:
        if 'ne' in w.lemma and w.pos == "ADV":
            w.to_picto = False
        if 'y' in w.lemma:
            w.to_picto = False
        if "pas" in w.lemma and w.pos == "ADV":
            w.add_picto([5526])


def prefix(text_spacy):
    prefix_infos = []
    for i, w in enumerate(text_spacy):
        for p in words_prefix:
            if w.lemma[0].startswith(p[0]):
                word_no_prefix = w.lemma[0][len(p[0]):]
                prefix_infos.append([p[0], p[1], i, word_no_prefix])
    num_pref = 0
    for pref in prefix_infos:
        prefix_pos = pref[2] + num_pref
        if text_spacy[prefix_pos].prefix == False and text_spacy[prefix_pos].pos not in ["DET", "PRON", "PROPN"]:
            text_spacy.insert(prefix_pos,
                              Word(pref[0], [pref[0]], "PREF", morph='', dep='', id_picto=pref[1], prefix=True))
            text_spacy[prefix_pos + 1].lemma.insert(0, pref[3])
            text_spacy[prefix_pos + 1].prefix = True
            num_pref += 1


def add_polylexical_picto(text_spacy, index, id_picto):
    text_spacy[index].picto = id_picto
    text_spacy[index - 1].to_picto = False
    text_spacy[index].prefix = True


def search_picto_for_poly(text_spacy, lemmas, voc1, dico_picto):
    for l in lemmas:
        id_picto = get_picto_dicoPicto(l[0], dico_picto)
        if not text_spacy[l[1]].prefix:
            if id_picto:
                add_polylexical_picto(text_spacy, l[1], id_picto)
            else:
                id_picto = get_picto_voc(l[0], voc1)
                if id_picto:
                    add_polylexical_picto(text_spacy, l[1], id_picto)


def polylexical(text_spacy, voc1, dico_picto):
    lemmas_with_prefix = [w.lemma[0] for w in text_spacy if w.to_picto == True]
    lemmas_no_prefix = []
    lemmas_plur = []
    for w in text_spacy:
        if w.to_picto == True:
            if len(w.lemma) > 1:
                lemmas_no_prefix.append(w.lemma[1])
            else:
                lemmas_no_prefix.append(w.lemma[0])
            if w.plur:
                lemmas_plur.append(w.token)
    lemmas_with_prefix_search = [[lemmas_with_prefix[i] + " " + lemmas_with_prefix[i + 1], i + 1] for i in
                                 range(len(lemmas_with_prefix) - 1)]
    lemmas_plur_search = [[lemmas_plur[i] + " " + lemmas_plur[i + 1], i + 1] for i in
                          range(len(lemmas_plur) - 1)]
    lemmas_no_prefix_search = [[lemmas_no_prefix[i] + " " + lemmas_no_prefix[i + 1], i + 1] for i in
                               range(len(lemmas_no_prefix) - 1)]
    search_picto_for_poly(text_spacy, lemmas_with_prefix_search, voc1, dico_picto)
    search_picto_for_poly(text_spacy, lemmas_plur_search, voc1, dico_picto)
    search_picto_for_poly(text_spacy, lemmas_no_prefix_search, voc1, dico_picto)


def mapping_text_to_picto(text_after_rules, voc1, dico_picto):
    for i, w in enumerate(text_after_rules):
        if w.to_picto and w.picto is None:
            if w.plur:
                lemma = w.token
                id_picto = get_picto_dicoPicto(lemma, dico_picto)
                if not id_picto:
                    id_picto = get_picto_voc(lemma, voc1)
                    if not id_picto:
                        lemma = w.lemma[1] if w.prefix else w.lemma[0]
                        id_picto = get_picto_dicoPicto(lemma, dico_picto)
                        if not id_picto:
                            id_picto = get_picto_voc(lemma, voc1)
            else:
                lemma = w.lemma[1] if w.prefix else w.lemma[0]
                id_picto = get_picto_dicoPicto(lemma, dico_picto)
                if not id_picto:
                    id_picto = get_picto_voc(lemma, voc1)
            if id_picto:
                w.add_picto(id_picto)
                if w.prefix:
                    text_after_rules[i - 1].to_picto = False
            else:
                if w.prefix:
                    text_after_rules[i - 1].to_picto = False
                w.add_picto([404])


# ------------------------- #
# -------- GRAMMAR -------- #
# ------------------------- #
def grammar(sentence, spacy_model):

    voc_magali = read_initial_voc_file("/data/macairec/PhD/Grammaire/dico/arasaac.fre30bis.csv")
    dicoPicto = read_dicoPicto("/data/macairec/PhD/Grammaire/dico/dicoPicto.csv")

    # apply rules
    s_process = process_text(sentence)
    s_spacy = process_with_spacy(s_process, spacy_model)
    imperative_sentence(s_spacy)
    futur_tense(s_spacy)
    past_tense(s_spacy)
    pronominale(s_spacy)
    nombre(s_spacy)
    indic_temp(s_spacy)
    neg(s_spacy)
    prefix(s_spacy)
    polylexical(s_spacy, voc_magali, dicoPicto)

    # mapping to ic_picto
    mapping_text_to_picto(s_spacy, voc_magali, dicoPicto)
    print("-----------------")
    for w in s_spacy:
        print(w.__str__())
    return s_spacy


if __name__ == '__main__':
    spacy_model = load_model("fr_core_news_md")
    sentences = ["écoute moi", "mange ton diner", "ferme la porte", "lève toi tôt", "donne moi cette clé", "quelle chanson tu préfères", "quel est ton nom", "une fois qu'il eut terminé", "il a été très sympa", "je l'avais fait", "je serai venu dans pas longtemps", "nous nous sommes perdus", "elles sont gentilles", "les chiens sont bruyants", "il va à la chasse", "le mardi vingt cinq juin deux mille", "elle n'y va pas", "tu ne veux pas dormir", "il n'est pas trop tard pour dormir", "La communication interpersonnelle est essentielle pour une relation saine.", "j'ai faim", "quel bel arc-en-ciel", "il n'y a pas"]
    phrases = [
        "Écoute moi",
        "Faites attention où vous marchez",
        "Quel est ton nom",
        "Quel beau pantalon",
        "L’étudiant a révisé",
        "Je suis en stage",
        "Je n'ai pas révisé je le ferai",
        "Elle part demain et reviens mercredi",
        "J'ai faim",
        "Quel bel arc-en-ciel",
        "Il se douche",
        "Je ne veux pas me réveiller",
        "Tu y es allé quand",
        "Nous sommes allés au marché",
        "Elle est gentille",
        "Il est au cinéma avec un ami",
        "Elles sont gentilles",
        "Vous pouvez venir demain",
        "Mardi 27 juin 2023",
        "Elle n'y va pas",
        "Tu ne veux pas dormir",
        "Je souhaite recommencer",
        "Ceci est illogique"
    ]
    s_rules = [grammar(s, spacy_model) for s in phrases]
    print_pictograms(phrases, s_rules)

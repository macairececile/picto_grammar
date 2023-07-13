# python file to create the grammar and generate translation in pictos
import math

import pandas as pd
import spacy
from text_to_num import text2num
from print_sentences_from_grammar import *
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from disambiguate import *
from create_dicoPicto import *
from nltk.corpus import wordnet as wn
from create_json_from_grammar_for_pictoEdit import *

phatiques_onomatopees = ['ah', 'aïe', 'areu', 'atchoum', 'badaboum', 'baf', 'bah', 'bam', 'bang', 'bé', 'bêêê', 'beurk',
                         'ben', 'beh', "bref",
                         'bing', 'boum', 'broum', 'cataclop', 'clap clap', 'coa coa', 'cocorico', 'coin coin',
                         'crac',
                         'croa croa', 'cuicui', 'ding', 'ding deng dong', 'ding dong', 'dring', 'hé', 'hé ben',
                         'eh bien', 'euh',
                         'flic flac', 'flip flop', 'frou frou', 'glouglou', 'glou glou', 'groin groin', 'grr', 'hé',
                         'hep', 'hein',
                         'hi han', 'hip hip hip hourra', 'hop', 'houla', 'hourra', 'hum', 'mêêê', 'meuh', 'miam',
                         'miam miam',
                         'miaou',
                         'oh', 'O.K.', 'ouah', 'ouah ouah', 'ouf', 'ouh', 'paf', 'pan', 'patatras', 'pchhh', 'pchit',
                         'pff', 'pif-paf', 'pin pon', 'pioupiou', 'plouf', 'pof', 'pouet', 'pouet pouet', 'pouf',
                         'psst', 'ron ron',
                         'schlaf', 'snif', 'splaf', 'splatch', 'sss', 'tacatac', 'tagada', 'tchac', 'teuf teuf',
                         'tic tac', 'toc',
                         'tut tut', 'vlan', 'vroum', 'vrrr', 'wouah', 'zip']

special_characters = ['<', '>', '/', ',', '*', '"', ',', '.', '…', '!', '?', ':', ';', '#']

futur_anterieur = ["aurai", "auras", "aura", 'aurons', "aurez", "auront", "serai", "seras", "sera", "serons", "serez",
                   "seront"]

words_to_replace = [["Monsieur", "monsieur"], ["Madame", "madame"], ["mademoiselle", "madame"], ["ça", "cela"],
                    ["nan", "non"], ["madame", "madame"], ["belle", "belle"]]
pronouns_to_replace = [["l'", "le"], ["j'", "je"], ["c'", "ceci"], ["ça", "cela"], ["t'", "tu"], ["on", "nous"],
                       ["m'", "me"], ["une", "une"], ["cette", "la"]]

expressions_no_translation_2words = ["du coup", "non plus", "en fait", "à part", "voili voilou", "au final",
                              "point barre", "en vérité", "entre guillemets"]

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
# -------- READ ELEMENTS -------- #
# -------------------------------- #
def read_lexique(dicoPicto_file):
    df = pd.read_csv(dicoPicto_file, sep='\t')
    df.loc[:, 'keyword_no_cat'] = df['lemma'].apply(lambda a: str(a).split(' #')[0].split('?')[0].split('!')[0].strip())
    return df


def get_picto_lexique(lemma, lexique):
    return list(set([int(a) for a in lexique.loc[lexique['keyword_no_cat'] == lemma]["id_picto"].tolist()]))


def get_picto_from_synset(sense_key, wn_data, lexique):
    pos_sense_key = get_pos_synset(int(sense_key.split('%')[1][0]))
    synsets = [str(s).zfill(8) + pos_sense_key for s in
               list(set(wn_data.loc[wn_data['sense_key'] == sense_key]["synset"].tolist()))]
    pictos = []
    for s in synsets:
        pictos.extend(list(set(lexique.loc[lexique['synsets'].apply(lambda x: s in x), "id_picto"].tolist())))
    return pictos


# -------------------------------- #
# -------- READ DATA FILES ------- #
# -------------------------------- #
def read_sentences(csv_file):
    df = pd.read_csv(csv_file, sep='\t')
    # df2 = df.dropna()
    return [a.lower() for a in df["text"].tolist()], df


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
                 plur=False, pron=False, ner='', wsd=''):
        """
        Constructor method
        """
        if id_picto is None:
            id_picto = []
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
        self.pron = pron
        self.ner = ner
        self.wsd = wsd

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

        result += f"PRONOMINAL: {self.pron} "

        result += f"NER: {self.ner} "

        result += f"WSD: {self.wsd}"

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
    # unique_words = dict.fromkeys(text.split())
    # text_unique = ' '.join(unique_words)
    pattern = r"/([^/]+)/"
    matches = re.findall(pattern, text)
    for match in matches:
        text = text.replace("/" + match + "/", match.split(",")[0])
    text = text.replace("ouais", "oui").replace("’", "'").replace("plait", "plaît")
    for c in special_characters:
        if c in text:
            text = text.replace(c, '')

    if not text.strip():
        return None
    else:
        return ' '.join(text.split())


def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
    ner_model = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_model


# ------------------ #
# ------ RULES ----- #
# ------------------ #
def no_translation_words(text_spacy):
    for i, w in enumerate(text_spacy):
        if len(text_spacy) > i + 1:
            to_search = w.token + ' ' + text_spacy[i+1].token
            if to_search in expressions_no_translation_2words:
                w.to_picto = False
                text_spacy[i + 1].to_picto = False


def handle_dash(text_spacy, spacy_model):
    index_to_add_dash = []
    for i, w in enumerate(text_spacy):
        if len(text_spacy) > i+2:
            if w.token == "est":
                if text_spacy[i+1].token == '-ce':
                    if text_spacy[i+2].token == "que" or text_spacy[i+2].token == "qu'":
                        index_to_add_dash.append(i+1)
                        text_spacy[i + 2].token = "que"
                        text_spacy[i + 2].lemma = ["que"]
        if w.token.startswith('-') and len(w.token) > 1:
            w.token = w.token[1:]
            index_to_add_dash.append(i)
            if w.lemma[0].startswith('-'):
                w.lemma = [w.lemma[0][1:]]
    num_elem = 0
    doc = spacy_model('-')
    for ind in list(set(index_to_add_dash)):
        dash = Word("-", ["-"], "PRON", morph=doc[0].morph, dep='')
        dash.to_picto = False
        text_spacy.insert(ind + num_elem, dash)
        num_elem += 1


def handle_onomatopoeia_and_others(text_spacy):
    for w in text_spacy:
        for t in words_to_replace:
            if w.lemma[0] == t[0]:
                w.lemma = [t[1]]
        if w.token in phatiques_onomatopees:
            w.to_picto = False
        if w.pos == "INTJ":
            w.to_picto = False


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
                if w.lemma[0].startswith(("a", "e", "i", "o", "u", "y", "h")):
                    w.lemma.insert(0, "s'" + w.lemma[0])
                else:
                    w.lemma.insert(0, "se " + w.lemma[0])
                w.pron = True
                pron = False


def nombre(text_spacy):
    for w in text_spacy:
        if w.pos in ["NOUN", "PRON", "DET", "ADJ"]:
            if w.morph.get("Number"):
                if w.morph.get("Number")[0] == "Plur":
                    w.plur = True


def pronouns(text_spacy):
    for w in text_spacy:
        if w.pos in ["PRON", "DET"]:
            if w.token in [item[0] for item in pronouns_to_replace]:
                for p in pronouns_to_replace:
                    if w.token == p[0]:
                        w.lemma = [p[1]]
            else:
                w.lemma = [w.token]


def indic_temp(text_spacy):
    nums = []
    actual_group = []
    position_nums = []
    for i, w in enumerate(text_spacy):
        if w.pos == "NUM":
            actual_group.append(w.token)
            position_nums.append(i)
            if i == len(text_spacy) - 1 or text_spacy[i + 1].pos != "NUM":
                nums.append([" ".join(actual_group), i, position_nums])
                position_nums = []
                actual_group = []
    for el in nums:
        try:
            el[0] = el[0].replace(' - ', '-')
            num_version = text2num(el[0], 'fr')
            text_spacy[el[1]].lemma = [str(num_version)]
            for i in range(el[1] - (len(el[0].split()) - 1), el[1]):
                text_spacy[i].to_picto = False
            for p in el[2]:
                if p != el[1]:
                    text_spacy[p].to_picto = False
        except:
            print("No conversion in digit for numbers")


def neg(text_spacy):
    neg_position = []
    for i, w in enumerate(text_spacy):
        if 'ne' in w.lemma and w.pos == "ADV":
            w.to_picto = False
        if 'y' in w.lemma:
            w.to_picto = False
        if "pas" in w.lemma and w.pos == "ADV":
            w.add_picto([5526])
            j = i
            while j + 1 < len(text_spacy):
                if text_spacy[j + 1].pos in ["VERB", "AUX"]:
                    pos = j + 1
                    if neg_position:
                        for el in neg_position:
                            if el[0] == i:
                                el[1] = pos
                    else:
                        neg_position.append([i, pos])
                    j += 1
                else:
                    break
    for el in neg_position:
        text_spacy[el[0]], text_spacy[el[1]] = text_spacy[el[1]], text_spacy[el[0]]


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
    for i, w in enumerate(text_spacy):
        if w.pos == 'PREF':
            if text_spacy[i+1].to_picto == False:
                w.to_picto = False


def add_polylexical_picto(text_spacy, index, id_picto, n_length):
    text_spacy[index].picto = id_picto
    for i in range(index - 1, index - n_length - 1, -1):
        if text_spacy[i].prefix is False:
            text_spacy[i].to_picto = False
    text_spacy[index].prefix = True


def search_picto_for_poly(text_spacy, lemmas, lexique, n_length):
    for l in lemmas:
        id_picto = get_picto_lexique(l[0], lexique)
        if not text_spacy[l[1]].prefix:
            if id_picto:
                add_polylexical_picto(text_spacy, l[1], id_picto, n_length)


def generate_possible_polylexical(lemmas, n_length):
    return [
        [" ".join(lemmas[i:i + n_length + 1]).replace("' ", "'").replace(" -", "-").replace("- ", "-"), i + n_length]
        for i in
        range(len(lemmas) - n_length)]


def apply_polylexical(lemmas, text_spacy, lexique, n_length):
    lemmas_search = generate_possible_polylexical(lemmas, n_length)
    search_picto_for_poly(text_spacy, lemmas_search, lexique, n_length)


def polylexical(text_spacy, lexique, length):
    lemmas_with_prefix = [w.lemma[0] if w.pos != "NUM" else w.token for w in text_spacy]
    lemmas_with_prefix_and_plur = []
    lemmas_no_prefix = []
    lemmas_plur = []
    tokens = []
    for w in text_spacy:
        if w.pos == "NUM":
            lemmas_no_prefix.append(w.token)
            lemmas_plur.append(w.token)
            lemmas_with_prefix_and_plur.append(w.token)
        else:
            if len(w.lemma) > 1:
                lemmas_no_prefix.append(w.lemma[1])
            else:
                lemmas_no_prefix.append(w.lemma[0])
            if w.plur:
                lemmas_plur.append(w.token)
                lemmas_with_prefix_and_plur.append(w.token)
            else:
                lemmas_plur.append(w.lemma[0])
                lemmas_with_prefix_and_plur.append(w.lemma[0])
        tokens.append(w.token)
    apply_polylexical(lemmas_with_prefix, text_spacy, lexique, length)
    apply_polylexical(lemmas_no_prefix, text_spacy, lexique, length)
    apply_polylexical(lemmas_plur, text_spacy, lexique, length)
    apply_polylexical(tokens, text_spacy, lexique, length)
    apply_polylexical(lemmas_with_prefix_and_plur, text_spacy, lexique, length)


def name_entities(text_spacy, ner_model):
    ner_res = ner_model(' '.join([w.token for w in text_spacy]))
    output_ner = [[item['entity_group'], item['word'], item['start'], item['end']] for item in ner_res]
    pos_w_ner_next = []
    ner = False
    for n in output_ner:
        i = 0
        for j, w in enumerate(text_spacy):
            if i >= n[2] + 1 and i + len(w.token) <= n[3] and n[0] in ["LOC", "ORG", "PER"]:
                if not pos_w_ner_next:
                    pos_w_ner_next.append(j)
                    ner = True
                else:
                    if pos_w_ner_next and ner:
                        pos_w_ner_next.append(j)
                # w.ner = n[0]
            else:
                if pos_w_ner_next and ner:
                    translations = [picto for p in pos_w_ner_next if text_spacy[p].picto for picto in
                                    text_spacy[p].picto]
                    if not translations:
                        for p in pos_w_ner_next:
                            text_spacy[p].to_picto = False
                        text_spacy[pos_w_ner_next[-1]].ner = n[0]
                        text_spacy[pos_w_ner_next[-1]].to_picto = True
                pos_w_ner_next = []
                ner = False
            i = i + len(w.token) + 1
        if pos_w_ner_next and ner:
            translations = [picto for p in pos_w_ner_next if text_spacy[p].picto for picto in text_spacy[p].picto]
            if not translations:
                for p in pos_w_ner_next:
                    text_spacy[p].to_picto = False
                text_spacy[pos_w_ner_next[-1]].ner = n[0]
                text_spacy[pos_w_ner_next[-1]].to_picto = True


def check_ner(w, lexique):
    if w.ner != '' and w.to_picto:
        id_picto = get_picto_lexique(w.token, lexique)
        if not id_picto:
            if w.ner == 'LOC':
                w.add_picto([2704])
            if w.ner == 'ORG':
                w.add_picto([12333])
            if w.ner == 'PER':
                w.add_picto([36935])


def mapping_text_to_picto(text_after_rules, lexique):
    # modifier cette horreur de code
    for i, w in enumerate(text_after_rules):
        id_picto = []
        check_ner(w, lexique)
        if w.pron and w.to_picto and not w.picto:
            id_picto = get_picto_lexique(w.lemma[0], lexique)
            if not id_picto:
                lemma = w.lemma[1]
                id_picto = get_picto_lexique(lemma, lexique)
        if w.to_picto and not w.picto and not id_picto:
            if w.plur:
                lemma = w.token
                id_picto = get_picto_lexique(lemma, lexique)
                if not id_picto:
                    lemma = w.lemma[1] if w.prefix else w.lemma[0]
                    id_picto = get_picto_lexique(lemma, lexique)
            if w.prefix:
                lemma = w.lemma[1]
                id_picto = get_picto_lexique(lemma, lexique)
                if not id_picto:
                    lemma = w.lemma[0]
                    id_picto = get_picto_lexique(lemma, lexique)
                    if id_picto:
                        text_after_rules[i - 1].to_picto = True
                    else:
                        text_after_rules[i - 1].to_picto = False
                else:
                    text_after_rules[i - 1].to_picto = False
            else:
                if not id_picto:
                    lemma = w.lemma[0]
                    id_picto = get_picto_lexique(lemma, lexique)
        if id_picto and not w.picto:
            w.add_picto(id_picto)
        else:
            if not w.picto and w.to_picto:
                w.add_picto([404])
                if w.prefix:
                    text_after_rules[i - 1].to_picto = False


def get_words_no_picto(texts_grammar):
    word_no_picto = []
    for t in texts_grammar:
        if t is not None:
            for w in t:
                if w.to_picto and w.picto == [404]:
                    word_no_picto.extend(w.lemma)
    return list(set(word_no_picto))


def apply_wsd(wsd_model, text_spacy, lexique, wn_data):
    apply = False
    for w in text_spacy:
        if w.to_picto and w.picto == [404] and w.pos in ["VERB", "ADJ", "NOUN", "PROPN"]:
            apply = True
    if apply:
        for w in text_spacy:
            print(w.token)
        disambiguate(text_spacy, wsd_model)
        for w in text_spacy:
            if w.wsd != '':
                picto = get_picto_from_synset(w.wsd, wn_data, lexique)
                if picto:
                    w.picto = picto
                else:
                    synonyms = get_synonyms(w.wsd)
                    if synonyms:
                        w.wsd += ';'.join(synonyms)
                        picto = list(
                            set([p for s in [get_picto_from_synset(syn, wn_data, lexique) for syn in synonyms]
                                 for
                                 p in s]))
                        if picto:
                            w.picto = picto


def remove_consecutive_picto(text_spacy):
    to_picto = [w.to_picto for w in text_spacy]
    for i, info in enumerate(to_picto):
        if not i == len(to_picto) - 1:
            if list(set(text_spacy[i].picto) & set(text_spacy[i + 1].picto)):
                text_spacy[i].to_picto = False


def get_synonyms(sense_key):
    try:
        synset = wn.synset_from_sense_key(sense_key)
        hypernyms = synset.hypernyms()
        hyponyms = synset.hyponyms()
        synonyms = [h.lemmas()[0].key() for h in hypernyms] + [h.lemmas()[0].key() for h in hyponyms]
        return synonyms
    except:
        return []


# ------------------------- #
# -------- GRAMMAR -------- #
# ------------------------- #
def grammar(sentence, spacy_model, words_not_in_dico_picto, ner_model, wsd_model, sentences_proc):
    lexique = read_lexique(
        "/data/macairec/PhD/Grammaire/dico/lexique.csv")
    wn_data = parse_wn31_file("/data/macairec/PhD/Grammaire/dico/index.sense")
    # apply rules
    s_process = process_text(sentence)
    if s_process is not None:
        sentences_proc.append(s_process)
        s_spacy = process_with_spacy(s_process, spacy_model)
        handle_dash(s_spacy, spacy_model)
        no_translation_words(s_spacy)
        handle_onomatopoeia_and_others(s_spacy)
        imperative_sentence(s_spacy)
        futur_tense(s_spacy)
        past_tense(s_spacy)
        pronominale(s_spacy)
        nombre(s_spacy)
        pronouns(s_spacy)
        indic_temp(s_spacy)
        neg(s_spacy)
        for i in range(8, 0, -1):
            polylexical(s_spacy, lexique, i)
        prefix(s_spacy)
        name_entities(s_spacy, ner_model)

        # mapping to ic_picto
        mapping_text_to_picto(s_spacy, lexique)
        # apply_wsd(wsd_model, s_spacy, lexique, wn_data)
        remove_consecutive_picto(s_spacy)
        print("-----------------")
        for w in s_spacy:
            # if w.picto == [404]:
            #     words_not_in_dico_picto.extend(w.lemma)
            print(w.__str__())

        return s_spacy
    else:
        return None



if __name__ == '__main__':
    spacy_model = load_model("fr_dep_news_trf")
    ner_model = load_ner_model()
    # wsd_model = load_wsd_model("/data/macairec/PhD/pictodemo/model_wsd/")
    # sentences, dataframe = read_sentences("/data/macairec/PhD/Grammaire/corpus/csv/corpus_grammar_selected_sentences_from_audio_PE2.csv")
    sentences = ["et qu'est-ce qu'il veut faire avec la voiture", "oui heu où est-ce que tu es allé en vacance", "comment vas-tu",
                               "est-ce qu'on lance le débat", "genre qui est-ce qui nettoie la douche", "où allons-nous", "est-ce qu'il pleut", "quel âge as-tu", "combien y en a-t-il", "où vont-ils", "où va-t-il"]
    sentences_2 = ["enfin voili voilou je suis pas sûr"]
    words_not_in_dico_picto = []
    sentences_proc = []
    html_file = "/data/macairec/PhD/Grammaire/scripts/test.html"
    s_rules = [grammar(s, spacy_model, words_not_in_dico_picto, ner_model, "", sentences_proc) for s in sentences_2]
    # create_json(sentences_proc, s_rules, "/data/macairec/PhD/Grammaire/corpus/json_PE/PE2.json")
    # print("\nWords with no picto : ", get_words_no_picto(s_rules))
    # print("\nWords not in dico_picto : ", list(set(words_not_in_dico_picto)))
    print_pictograms(sentences_2, s_rules, html_file)

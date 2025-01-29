"""
Script to apply the grammar and retrieve the sequence of pictograms.

"""

from print_sentences_from_grammar import *
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from disambiguate import *
from create_final_lexicon import *
from nltk.corpus import wordnet as wn
from analysis_results_PE import *
import csv
import spacy
from text_to_num import text2num
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# -------------------------------- #
# ---------- CONSTANTS ----------- #
# -------------------------------- #
phatiques_onomatopees = ['ah', 'aïe', 'areu', 'atchoum', 'badaboum', 'baf', 'bah', 'bam', 'bang', 'bé', 'bè', 'bêêê',
                         'beurk',
                         'ben', 'beh', "bref", "brrrrr", "brrrrrr",
                         'bing', 'boum', 'broum', 'cataclop', 'clap clap', 'coa coa', 'cocorico', 'coin coin',
                         'crac',
                         'croa croa', 'cuicui', 'ding', 'ding deng dong', 'ding dong', 'dring', 'hé', 'eh', 'hé ben',
                         'eh bien', 'euh',
                         'flic flac', 'flip flop', 'frou frou', 'glouglou', 'glou glou', 'groin groin', 'grr', 'hé',
                         'hep', 'hein', 'hm',
                         'hi han', 'hip hip hip hourra', 'hop', 'houla', 'hourra', 'hum', 'mêêê', 'meuh', 'miam',
                         'miam miam', 'mh', 'mm',
                         'miaou',
                         'oh', 'oh là là', 'oh là', 'ohé', 'O.K.', 'ouah', 'ouah ouah', 'ouf', 'ouh', 'paf', 'pan',
                         'patatras',
                         'pchhh', 'pchit', 'pf',
                         'pff', 'pif-paf', 'pin pon', 'pioupiou', 'plouf', 'pof', 'pouet', 'pouet pouet', 'pouf',
                         'psst', 'rah', 'ron ron',
                         'schlaf', 'snif', 'splaf', 'splatch', 'sss', 'tacatac', 'tagada', 'tchac', 'teuf teuf',
                         'tic tac', 'toc',
                         'tut tut', 'vlan', 'vroum', 'vrrr', 'wouah', 'zip']

single_letters = ['b', 'c', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
                  'e']

special_characters = ['<', '>', '/', ',', '*', '"', ',', '.', '…', '!', '?', ':', ';', '#', "²", "]", "»", "-~", "~",
                      "«", "(", ")"]

futur_anterieur = ["aurai", "auras", "aura", 'aurons', "aurez", "auront", "serai", "seras", "sera", "serons", "serez",
                   "seront"]

words_to_replace = [["Monsieur", "monsieur"], ["Madame", "madame"], ["mademoiselle", "madame"], ["ça", "cela"],
                    ["nan", "non"], ["madame", "madame"], ["belle", "belle"], ['moui', 'oui'], ["bel", "belle"]]

pronouns_to_replace = [["l'", "le"], ["j'", "je"], ["c'", "ceci"], ["ça", "cela"], ["t'", "tu"], ["on", "nous"],
                       ["m'", "me"], ["une", "une"], ["cette", "la"], ['te', 'toi'], ["ce", "celui"], ["qu'", "que"],
                       ["d'", "de"]]

words_no_translations = ["puis", "sinon", "voilà", "enfin", "donc", "complètement", "énormément", "chez", "vraiment",
                         "ainsi", "absolument", "assez", "essentiellement", "peut-être", "certainement",
                         "certes", "apparemment", "dont", "déjà", "envers", "façon", "sûrement", "rarement",
                         "surtout", "éventuellement", "éventuel", "alors"]

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


# ------------------------------------ #
# --------- READ/GET ELEMENTS -------- #
# ------------------------------------ #
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
    df.loc[:, 'keyword_no_cat'] = df['lemma'].apply(lambda a: str(a).split(' #')[0].split('?')[0].split('!')[0].strip())
    return df


def read_sentences(csv_file):
    """
        Get the transcriptions to apply to grammar.

        Arguments
        ---------
        csv_file: str

        Returns
        -------
        The list of sentences.
    """
    df = pd.read_csv(csv_file, sep='\t')
    # df2 = df.dropna()
    df2 = df.drop_duplicates()
    return [a.lower() for a in df2["text"].tolist()], df2


def get_picto_lemma(lemma, lexicon):
    """
        Get the pictograms associated to a lemma.

        Arguments
        ---------
        lemma: str
        lexicon: dataframe

        Returns
        -------
        The list of pictograms.
    """
    return list(set([int(a) for a in lexicon.loc[lexicon['keyword_no_cat'] == lemma]["id_picto"].tolist()]))


def get_picto_from_sensekey(sense_key, wn_data, lexicon):
    """
        Get the pictogram associated to a sense key.

        Arguments
        ---------
        sense_key: str
        wn_data: dataframe
        lexicon: dataframe

        Returns
        -------
        The list of pictograms.
    """
    pos_sense_key = get_pos_synset(int(sense_key.split('%')[1][0]))
    synsets = [str(s).zfill(8) + pos_sense_key for s in
               list(set(wn_data.loc[wn_data['sense_key'] == sense_key]["synset"].tolist()))]
    pictos = []
    for s in synsets:
        pictos.extend(list(set(lexicon.loc[lexicon['synsets'].apply(lambda x: s in x), "id_picto"].tolist())))
    return pictos


def read_no_translation_words(no_translation_file):
    """
        Get the list of expressions to not translate (disfluencies).

        Arguments
        ---------
        no_translation_file: str

        Returns
        -------
        The list of expressions.
    """
    no_translation = []
    with open(no_translation_file, newline='') as csvfile:
        csv_lines = csv.reader(csvfile)
        for ligne in csv_lines:
            element = ligne[0]
            no_translation.append(element)
    return no_translation


def load_ner_model():
    """
        Load the NER model.

        Returns
        -------
        The NER model.
    """
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
    ner_model = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    return ner_model


# ---------------------------- #
# -------- CLASS WORD -------- #
# ---------------------------- #
class Word():
    """
        Representation of a word and its associated information to apply the grammar.

        Returns
        -------
        A Word object.
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
        if ent_type:
            self.ent_type = ent_type
        else:
            self.ent_type = None

        self.ent_text = None

    def __str__(self):
        """
        Print method
        """
        result = ""

        if self.token:
            result = f"TOKEN: {self.token}  "

        if self.lemma:
            result += f"LEMMA: {self.lemma}  "

        if self.pos:
            result += f"POS: {self.pos}  "

        if self.ent_type:
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


# ---------------------------- #
# ----- PROCESS THE TEXT ----- #
# ---------------------------- #
def process_text(text):
    """
        Clean the text before applying spacy.

        Arguments
        -------
        text: str

        Returns
        -------
        The cleaned text.
    """
    # remove ce qui est entre parenthèses :
    text = re.sub(r'\([^)]*\)', '', text)
    pattern = r"/([^/]+)/"
    matches = re.findall(pattern, text)
    for match in matches:
        text = text.replace("/" + match + "/", match.split(",")[0])
    text = text.replace("aujourd' hui", "aujourd'hui").replace("quelqu' un", "quelqu'un").replace("c' est-à-dire",
                                                                                                  "c'est-à-dire")
    text = text.replace("ouais", "oui").replace("’", "'").replace("plait", "plaît").replace("là bas", "là-bas").replace(
        "qu' ", "que ")
    for c in special_characters:
        if c in text:
            text = text.replace(c, '')
    if not text.strip():
        return None
    else:
        return ' '.join(text.split())


def remove_phatiques(text, p):
    while ' ' + p + ' ' in text:
        text = text.replace(' ' + p + ' ', ' ').strip()
    return text


def remove_pathiques_begin(text, p):
    if text.startswith(p + ' '):
        return text[len(p):]
    else:
        return text


def remove_phatiques_end(text, p):
    if text.endswith(' ' + p):
        return text[:-len(p)]
    else:
        return text


def remove_useless_words(text, no_translation_words):
    """
        Remove disfluencies from the text.

        Arguments
        -------
        text: str
        no_translation_words: list

        Returns
        -------
        The cleaned text.
    """
    text_to_proc = text
    for p in phatiques_onomatopees:
        text_1 = remove_phatiques(text_to_proc, p)
        text_2 = remove_pathiques_begin(text_1, p)
        text_3 = remove_phatiques_end(text_2, p)
        text_to_proc = text_3
    text_no_pathiques_onom = ' '.join(text_to_proc.split())
    # handle words not to translate
    for w in no_translation_words:
        text_no_pathiques_onom = text_no_pathiques_onom.replace(w, '')
    final_text = ' '.join(text_no_pathiques_onom.split())
    for s in single_letters:
        final_text = final_text.replace(' ' + s + ' ', ' ')
        if final_text.startswith(s + ' '):
            final_text = final_text[1:]
        if final_text.endswith(' ' + s):
            final_text = final_text[:-1]
        final_text = ' '.join(final_text.split())
        if final_text == s:
            final_text = ""
    return final_text


# ---------------------------- #
# ------- SPACY MODULE ------- #
# ---------------------------- #
def load_model(model_name):
    """
        Load the spacy model.

        Arguments
        -------
        model_name: str

        Returns
        -------
        A spacy model.
    """
    nlp = spacy.load(model_name)
    print("*** Spacy model ready to use : " + model_name + " ***\n")
    return nlp


def process_with_spacy(text, spacy_model):
    """
        Apply the spacy model on a text.

        Arguments
        -------
        text: str
        spacy_model: Language

        Returns
        -------
        The text width, for each word, information tagged by spacy.
    """
    doc = spacy_model(text)
    text_spacy = []
    for token in doc:
        if token.lemma_ != " " and token.lemma_ != "'":
            text_spacy.append(
                Word(token.text, [token.lemma_], token.pos_, token.morph, token.dep_, ent_type=token.ent_type_,
                     tag=token.tag_))
    return text_spacy


# ------------------- #
# ------ RULES ------ #
# ------------------- #
def handle_anonym(text_spacy):
    for w in text_spacy:
        if w.lemma == ["nnaammee"]:
            w.picto = [36935]


def handle_dash(text_spacy, spacy_model):
    """
        Handle dash for specific multi-word-expressions.

        Arguments
        ---------
        text_spacy: list
        spacy_model: Language
    """
    index_to_add_dash = []
    for i, w in enumerate(text_spacy):
        if len(text_spacy) > i + 2:
            if w.token == "est":
                if text_spacy[i + 1].token == '-ce':
                    if text_spacy[i + 2].token == "que" or text_spacy[i + 2].token == "qu'":
                        index_to_add_dash.append(i + 1)
                        text_spacy[i + 2].token = "que"
                        text_spacy[i + 2].lemma = ["que"]
        if w.token.startswith('-') and len(w.token) > 1:
            w.token = w.token[1:]
            index_to_add_dash.append(i)
            if w.lemma[0].startswith('-'):
                w.lemma = [w.lemma[0][1:]]
        if w.token == '-':
            w.to_picto = False
    num_elem = 0
    doc = spacy_model('-')
    for ind in list(set(index_to_add_dash)):
        dash = Word("-", ["-"], "PRON", morph=doc[0].morph, dep='')
        dash.to_picto = False
        text_spacy.insert(ind + num_elem, dash)
        num_elem += 1


def handle_intj(text_spacy):
    """
        Handle interjection, i.e. do not translate in pictograms.

        Arguments
        ---------
        text_spacy: list
    """
    for w in text_spacy:
        for t in words_to_replace:
            if w.lemma[0] == t[0]:
                w.lemma = [t[1]]
        if w.pos == "INTJ":
            w.to_picto = False


def imperative_sentence(text_spacy):
    """
        Add a specific marker for imperative sentences.

        Arguments
        ---------
        text_spacy: list
    """
    if text_spacy[0].pos == 'VERB':
        text_spacy.append(Word('!', ['!'], 'PUNCT', morph='', dep='', id_picto=[3417]))


def interr_excla(text_spacy):
    """
        Handle interrogation and exclamative sentences.

        Arguments
        ---------
        text_spacy: list
    """
    for w in text_spacy:
        if w.lemma == ["quel"]:
            w.add_picto([22620, 22624])


def add_tense_marker(text_spacy, add_tense_marker, tense):
    """
        Add a tense marker for futur or past tense.

        Arguments
        ---------
        text_spacy: list
        add_tense_marker: list
        tense: str
    """
    if add_tense_marker:
        for i in range(add_tense_marker[0][1], -1, -1):
            # case where nsubj preceded with a determiner
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
    """
        Detect the past tense and process the information.

        Arguments
        ---------
        text_spacy: list
    """
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
            if w.pos in ['AUX', 'VERB'] and i + 1 < num_words and next is False:
                if text_spacy[i + 1].pos in ['VERB', 'AUX'] and text_spacy[i + 1].morph.get("Tense"):
                    if text_spacy[i + 1].morph.get("Tense")[0] == 'Past':
                        w.to_picto = False
                        add_past.append([True, i])
                        next = True
            # case 3 : 1 word
            if w.pos in ['AUX', 'VERB'] and w.morph.get("Tense") and next is False:
                if w.morph.get("Tense")[0] in ['Past', 'Imp']:
                    add_past.append([True, i])
    # ajout du marqueur du passé
    add_tense_marker(text_spacy, add_past, "past")


def futur_tense(text_spacy):
    """
        Detect the futur tense and process the information.

        Arguments
        ---------
        text_spacy: list
    """
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
    """
        Handle pronominal form of verbs.

        Arguments
        ---------
        text_spacy: list
    """
    pron = False
    for i, w in enumerate(text_spacy):
        if not type(w.morph) == str:
            if w.pos == "PRON" and w.morph.get("Reflex"):
                if w.morph.get("Reflex")[0] == "Yes":
                    w.to_picto = False
                    pron = True
            if w.pos == "VERB" and pron:
                if w.lemma[0].startswith(("a", "e", "i", "o", "u", "y", "h")):
                    w.lemma.insert(0, "s'" + w.lemma[0])
                else:
                    w.lemma.insert(0, "se " + w.lemma[0])
                w.pron = True
                pron = False


def nombre(text_spacy):
    """
        Handle the singular/plural form.

        Arguments
        ---------
        text_spacy: list
    """
    for w in text_spacy:
        if w.pos in ["NOUN", "PRON", "DET", "ADJ"]:
            if w.morph.get("Number"):
                if w.morph.get("Number")[0] == "Plur":
                    w.plur = True


def pronouns(text_spacy):
    """
        Handle the pronouns (if relative or not).

        Arguments
        ---------
        text_spacy: list
    """
    for w in text_spacy:
        if w.pos in ["PRON", "DET"]:
            if w.token in [item[0] for item in pronouns_to_replace]:
                for p in pronouns_to_replace:
                    if w.token == p[0]:
                        w.lemma = [p[1]]
            else:
                w.lemma = [w.token]
            if w.token == "qui" and w.morph.get("PronType"):
                if w.morph.get("PronType")[0] == "Rel":
                    w.picto = [11351]


def indic_temp(text_spacy):
    """
        Detect a number and convert in a unique form.

        Arguments
        ---------
        text_spacy: list
    """
    nums = []
    actual_group = []
    position_nums = []
    for i, w in enumerate(text_spacy):
        if w.pos == "NUM":
            actual_group.append(w.token)
            position_nums.append(i)
            if i < len(text_spacy) - 1:
                if text_spacy[i + 1].pos != "NUM":
                    if text_spacy[i + 1].token.endswith("ième") or text_spacy[i + 1].token.endswith("ièmes"):
                        for pos in position_nums:
                            text_spacy[pos].to_picto = False
                        text_spacy[i + 1].to_picto = True
                        text_spacy[i + 1].picto = [9877]
                else:
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
    """
        Handle the negation form.

        Arguments
        ---------
        text_spacy: list
    """
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
    """
        Detect and handle the prefix.

        Arguments
        ---------
        text_spacy: list
    """
    prefix_infos = []
    for i, w in enumerate(text_spacy):
        for p in words_prefix:
            if w.lemma[0].startswith(p[0]):
                word_no_prefix = w.lemma[0][len(p[0]):]
                prefix_infos.append([p[0], p[1], i, word_no_prefix])
    num_pref = 0
    for pref in prefix_infos:
        prefix_pos = pref[2] + num_pref
        if text_spacy[prefix_pos].prefix is False and text_spacy[prefix_pos].pos not in ["DET", "PRON", "PROPN"]:
            text_spacy.insert(prefix_pos,
                              Word(pref[0], [pref[0]], "PREF", morph='', dep='', id_picto=pref[1], prefix=True))
            text_spacy[prefix_pos + 1].lemma.insert(0, pref[3])
            text_spacy[prefix_pos + 1].prefix = True
            num_pref += 1
    for i, w in enumerate(text_spacy):
        if w.pos == 'PREF':
            if text_spacy[i + 1].to_picto is False:
                w.to_picto = False


def search_picto_for_mwe(text_spacy, lemmas, lexicon, n_length):
    """
        Search the pictogram linked to the mwe.

        Arguments
        ---------
        text_spacy: list
        lemmas: list
        lexicon: dataframe
        n_length: int
    """
    for l in lemmas:
        id_picto = get_picto_lemma(l[0], lexicon)
        if not text_spacy[l[1]].prefix:
            if id_picto:
                add_mwe_picto(text_spacy, l[1], id_picto, n_length)


def add_mwe_picto(text_spacy, index, id_picto, n_length):
    """
        Add the MWE picto.

        Arguments
        ---------
        text_spacy: list
        index: int
        id_picto: list
        n_length: int
    """
    text_spacy[index].picto = id_picto
    for i in range(index - 1, index - n_length - 1, -1):
        if text_spacy[i].prefix is False:
            text_spacy[i].to_picto = False
    text_spacy[index].prefix = True


def generate_possible_polylexical(lemmas, n_length):
    """
        Generate all the possible mwe from lemmas.

        Arguments
        ---------
        lemmas: list
        n_length: int

        Returns
        ---------
        The possibilities for MWE.
    """
    return [
        [" ".join(lemmas[i:i + n_length + 1]).replace("' ", "'").replace(" -", "-").replace("- ", "-"), i + n_length]
        for i in
        range(len(lemmas) - n_length)]


def apply_mwe_with_in_between_words(lemmas, text_spacy, lexicon, length):
    """
        Generate all the possible mwe from lemmas with in between words.

        Arguments
        ---------
        lemmas: list
        text_spacy: list
        lexicon: dataframe
        length: int
    """
    lemmas_search = [[" ".join(lemmas[i:i + length + 1]), i + length] for i in range(len(lemmas) - length)]
    for i in range(1, length):
        for lem in lemmas_search:
            index_to_keep = lem[1] - length + i
            to_search_split = lem[0].split(" ")
            to_search = to_search_split[:i] + to_search_split[i + 1:]
            to_search = " ".join(to_search).replace("' ", "'").replace(" -", "-").replace("- ", "-")
            id_picto = get_picto_lemma(to_search, lexicon)
            if not text_spacy[lem[1]].prefix:
                if id_picto:
                    add_mwe_picto(text_spacy, lem[1], id_picto, length)
                    text_spacy[index_to_keep].to_picto = True


def apply_mwe(lemmas, text_spacy, lexicon, n_length):
    """
        Apply MWE.

        Arguments
        ---------
        lemmas: list
        text_spacy: list
        lexicon: dataframe
        n_length: int
    """
    lemmas_search = generate_possible_polylexical(lemmas, n_length)
    search_picto_for_mwe(text_spacy, lemmas_search, lexicon, n_length)


def mwe(text_spacy, lexicon, length):
    """
        Handle the MWE from 8 to 2 words with or without in between words.

        Arguments
        ---------
        text_spacy: list
        lexicon: dataframe
        length: int
    """
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
    apply_mwe(lemmas_with_prefix, text_spacy, lexicon, length)
    apply_mwe(lemmas_no_prefix, text_spacy, lexicon, length)
    apply_mwe(lemmas_plur, text_spacy, lexicon, length)
    apply_mwe(tokens, text_spacy, lexicon, length)
    apply_mwe(lemmas_with_prefix_and_plur, text_spacy, lexicon, length)
    apply_mwe_with_in_between_words(lemmas_with_prefix, text_spacy, lexicon, length)
    apply_mwe_with_in_between_words(lemmas_no_prefix, text_spacy, lexicon, length)
    apply_mwe_with_in_between_words(lemmas_plur, text_spacy, lexicon, length)
    apply_mwe_with_in_between_words(tokens, text_spacy, lexicon, length)


def name_entities(text_spacy, ner_model):
    """
        Recognize and handle the named entities.

        Arguments
        ---------
        text_spacy: list
        ner_model: Pipeline
    """
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


def check_ner(w, lexicon):
    """
        Translate a Named Entity with a default pictogram.

        Arguments
        ---------
        w: Word
        lexicon: dataframe
    """
    if w.ner != '' and w.to_picto:
        id_picto = get_picto_lemma(w.token, lexicon)
        if not id_picto:
            if w.ner == 'LOC':
                w.add_picto([2704])
            if w.ner == 'ORG':
                w.add_picto([12333])
            if w.ner == 'PER':
                w.add_picto([36935])


def special_cases(text_spacy):
    """
        Handle special cases (e.g. Paris the city; Wrong tokenization, words to not translate)

        Arguments
        ---------
        text_spacy: list
    """
    for a, w in enumerate(text_spacy):
        if w.token == 'paris' and w.pos == 'PROPN':
            w.picto = [10271]
        if w.token in words_no_translations:
            pref = [i[0] for i in words_prefix]
            if any(w.token.startswith(s) for s in pref):
                text_spacy[a - 1].to_picto = False
            w.to_picto = False
        if w.pos == "VERB":
            if len(w.lemma) == 2:
                idx = 1
            else:
                idx = 0
            if not any(w.lemma[idx].endswith(s) for s in ["ir", "oir", "re", "aller"]):
                if w.lemma[idx].endswith("e"):
                    w.lemma[idx] = w.lemma[idx] + "r"
        if w.token == "car" and w.pos == "CCONJ":
            w.picto = [11348]


def adverbial_group(text_spacy):
    for i, w in enumerate(text_spacy):
        if w.token in ["tout", "fort", "très", "assez"] and w.pos == "ADV":
            if i < len(text_spacy) - 1:
                if text_spacy[i + 1].pos == "ADV":
                    w.to_picto = False
                if text_spacy[i + 1].pos == "ADJ" and w.token in ["tout", "fort", "assez"]:
                    w.to_picto = False


def mapping_text_to_picto(text_after_rules, lexicon):
    """
        Mapping words to pictograms depending on the annotated information.

        Arguments
        ---------
        text_after_rules: list
        lexicon: dataframe
    """
    # Modify this horror...
    for i, w in enumerate(text_after_rules):
        id_picto = []
        check_ner(w, lexicon)
        if w.pron and w.to_picto and not w.picto:
            id_picto = get_picto_lemma(w.lemma[0], lexicon)
            if not id_picto:
                lemma = w.lemma[1]
                id_picto = get_picto_lemma(lemma, lexicon)
        if w.to_picto and not w.picto and not id_picto:
            if w.plur:
                lemma = w.token
                id_picto = get_picto_lemma(lemma, lexicon)
                if not id_picto:
                    lemma = w.lemma[1] if w.prefix else w.lemma[0]
                    id_picto = get_picto_lemma(lemma, lexicon)
            if w.prefix:
                lemma = w.lemma[1]
                id_picto = get_picto_lemma(lemma, lexicon)
                if not id_picto:
                    lemma = w.lemma[0]
                    id_picto = get_picto_lemma(lemma, lexicon)
                    if id_picto:
                        text_after_rules[i - 1].to_picto = True
                    else:
                        text_after_rules[i - 1].to_picto = False
                else:
                    text_after_rules[i - 1].to_picto = False
            else:
                if not id_picto:
                    lemma = w.lemma[0]
                    id_picto = get_picto_lemma(lemma, lexicon)
        if id_picto and not w.picto:
            w.add_picto(id_picto)
        else:
            if not w.picto and w.to_picto:
                w.add_picto([404])
                if w.prefix:
                    text_after_rules[i - 1].to_picto = False


def apply_wsd(wsd_model, text_spacy, lexicon, wn_data):
    """
        Apply thr wsd pipeline.

        Arguments
        ---------
        wsd_model: NeuralDisambiguator
        text_spacy: list
        lexicon: dataframe
        wn_data: dataframe
    """
    apply = False
    for w in text_spacy:
        if w.to_picto and w.picto == [404] and w.pos in ["VERB", "ADJ", "NOUN", "PROPN"]:
            apply = True
    if apply:
        disambiguate(text_spacy, wsd_model)
        for w in text_spacy:
            if w.wsd != '':
                picto = get_picto_from_sensekey(w.wsd, wn_data, lexicon)
                if picto:
                    w.picto = picto
                else:
                    synonyms = get_synonyms(w.wsd)
                    if synonyms:
                        w.wsd += ';'.join(synonyms)
                        picto = list(
                            set([p for s in [get_picto_from_sensekey(syn, wn_data, lexicon) for syn in synonyms]
                                 for
                                 p in s]))
                        if picto:
                            w.picto = picto


def get_synonyms(sense_key):
    """
        Get the synonyms from a sense_key in WordNet.

        Arguments
        ---------
        sense_key: str

        Returns
        ---------
        A list of synonyms, if they exist
    """
    try:
        synset = wn.synset_from_sense_key(sense_key)
        hypernyms = synset.hypernyms()
        hyponyms = synset.hyponyms()
        synonyms = [h.lemmas()[0].key() for h in hypernyms] + [h.lemmas()[0].key() for h in hyponyms]
        return synonyms
    except:
        return []


def remove_consecutive_picto(text_spacy):
    """
        Remove the same consecutive pictograms.

        Arguments
        ---------
        text_spacy: list
    """
    to_picto = []
    for a, w in enumerate(text_spacy):
        if w.to_picto and len(w.picto) != 0:
            to_picto.append([a, w.picto])
    for i, info in enumerate(to_picto):
        if not i == len(to_picto) - 1:
            if list(set(info[1]) & set(to_picto[i + 1][1])):
                text_spacy[info[0]].to_picto = False


# ------------------- #
# ------ INFOS ------ #
# ------------------- #
def get_words_no_picto(texts_grammar):
    """
        Get the words with no associated pictograms.

        Arguments
        ---------
        texts_grammar: list

        Returns
        ---------
        The list of words with no pictogram.
    """
    word_no_picto = []
    for t in texts_grammar:
        if t is not None:
            for w in t:
                if w.to_picto and w.picto == [404]:
                    word_no_picto.extend(w.lemma)
    return list(set(word_no_picto))


# --------------------------------------- #
# -------- APPLY RULES - GRAMMAR -------- #
# --------------------------------------- #
def grammar(wn_data, no_transl, sentence, spacy_model, ner_model, wsd_model, sentences_proc,
            lexicon):
    """
        Apply the rules on the text.

        Arguments
        ---------
        wn_data: dataframe
        no_transl: list
        sentence: str
        spacy_model: Language
        ner_model: Pipeline
        wsd_model: NeuralDisambiguator
        sentences_proc: list
        lexicon: dataframe

        Returns
        ---------
        The list of words with no pictogram.
    """
    s_process = process_text(sentence)
    sentences_proc.append(s_process)
    if s_process is not None:
        s_new = remove_useless_words(s_process, no_transl)
        if len(s_new) > 0:
            s_spacy = process_with_spacy(s_new, spacy_model)
            handle_anonym(s_spacy)
            handle_dash(s_spacy, spacy_model)
            handle_intj(s_spacy)
            imperative_sentence(s_spacy)
            futur_tense(s_spacy)
            past_tense(s_spacy)
            pronominale(s_spacy)
            nombre(s_spacy)
            pronouns(s_spacy)
            indic_temp(s_spacy)
            for i in range(8, 0, -1):
                mwe(s_spacy, lexicon, i)
            neg(s_spacy)
            prefix(s_spacy)
            name_entities(s_spacy, ner_model)
            special_cases(s_spacy)
            adverbial_group(s_spacy)
            mapping_text_to_picto(s_spacy, lexicon)
            apply_wsd(wsd_model, s_spacy, lexicon, wn_data)
            remove_consecutive_picto(s_spacy)
            return s_spacy
        else:
            return None
    else:
        return None


def save_data_grammar_to_csv(data_init, sentences, s_rules, sentences_proc, lexicon, output_file):
    """
        Save the data from the grammar in a csv file.

        Arguments
        ---------
        data_init: dataframe
        sentences: list
        s_rules: list
        sentences_proc: list
        lexicon: dataframe
        output_file: str
    """
    out_data = pd.DataFrame(columns=['clips', 'text', 'text_process', 'pictos', 'tokens'])
    for i, j in enumerate(s_rules):
        if j is None:
            add_info = {'clips': data_init.iloc[i]["clips"], 'text': sentences[i], 'text_process': "",
                        'pictos': "",
                        'tokens': ""}
            out_data.loc[len(out_data), :] = add_info
        else:
            pictos = []
            for w in j:
                if w.to_picto:
                    if w.picto != [404]:
                        pictos.append(w.picto[0])
            tokens = get_token_from_id_pictos(pictos, lexicon)
            add_info = {'clips': data_init.iloc[i]["clips"], 'text': sentences[i],
                        'text_process': sentences_proc[i], 'pictos': pictos,
                        'tokens': " ".join(tokens)}
            out_data.loc[len(out_data), :] = add_info
    out_data.to_csv(output_file, sep='\t', index=False)


def main(args):
    wn_data = parse_wn31_file(args.wn_file)  # "index.sense"
    no_transl = read_no_translation_words(args.no_transl)  # "no_translation.csv"
    spacy_model = load_model("fr_dep_news_trf")
    ner_model = load_ner_model()
    wsd_model = load_wsd_model(args.wsd)
    lexicon = read_lexique(args.lexicon)
    sentences, data_init = read_sentences(args.data)
    sentences_proc = []
    s_rules = [
        grammar(wn_data, no_transl, s, spacy_model, ner_model, wsd_model, sentences_proc, lexicon) for s in
        sentences]
    save_data_grammar_to_csv(data_init, sentences, s_rules, sentences_proc, lexicon, args.out)
    generate_html(sentences, s_rules, "out.html", args.tags)


parser = ArgumentParser(description="Generate the translation in pictograms with the grammar.",
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('--wn_file', type=str, required=True,
                    help="")
parser.add_argument('--no_transl', type=str, required=True,
                    help="")
parser.add_argument('--wsd', type=str, required=True,
                    help="")
parser.add_argument('--lexicon', type=str, required=True,
                    help="")
parser.add_argument('--data', type=str, required=True,
                    help="")
parser.add_argument('--out', type=str, required=True,
                    help="")
parser.add_argument('--tags', type=str, required=True,
                    help="")
parser.set_defaults(func=main)
args = parser.parse_args()
args.func(args)

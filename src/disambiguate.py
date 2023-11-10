"""
Script to disambiguate a text.
You need to import a specific library to do so.

"""

from method.neural.NeuralDisambiguator import NeuralDisambiguator
from ufsac.ufsac.core.Word import Word
from ufsac.ufsac.core.Sentence import Sentence
from ufsac.common.WordnetHelper import WordnetHelper
from ufsac.common.XMLHelper import XMLHelper


def load_wsd_model(path_model):
    """
        Load a WSD model

        Arguments
        ---------
        path_model: str
            Path of the trained model.

        Returns
        -------
        A neural disambiguator.
    """
    lowercase = True
    clear_text = False
    batch_size = 1
    filter_lemma = False
    sense_compression_clusters = None
    wn = WordnetHelper.wn30()
    neural_disambiguator = NeuralDisambiguator(path_model + "data_wsd",
                                               [path_model + "model_weights_wsd0_camembert_base"],
                                               clear_text,
                                               batch_size, wn=wn, hf_model="camembert-base")
    neural_disambiguator.lowercase_words = lowercase
    neural_disambiguator.filter_lemma = filter_lemma
    neural_disambiguator.reduced_output_vocabulary = sense_compression_clusters
    return neural_disambiguator


def processing(text):
    """
        Process the text before the disambiguation.

        Arguments
        ---------
        text: list

        Returns
        -------
        The processed sentence.
    """
    sent = Sentence()

    for word in text:
        w_proc = Word(XMLHelper.from_valid_xml_entity(word.token.lower()))
        sent.add_word(w_proc)
    return [sent]


def disambiguate(text, neural_disambiguator):
    """
        Function to disambiguate predicted hypothesis from asr model and store in dict

        Arguments
        ---------
        neural_disambiguator: NeuralDisambiguator
        text: list
            Sentence to disambiguate

        Returns
        -------
        The disambiguate sentence.
    """
    sentence = processing(text)
    neural_disambiguator.disambiguate_dynamic_sentence_batch(sentence, "wsd_test")
    sense_keys_words = []
    for word in sentence[0].get_words():
        if word.has_annotation("wsd_test"):
            sense_keys_words.append(word.get_annotation_value("wsd_test"))
    for i, w in enumerate(text):
        if w.to_picto and w.picto == [404] and w.pos in ["VERB", "ADJ", "NOUN", "PROPN"]:
            w.wsd = sense_keys_words[i]

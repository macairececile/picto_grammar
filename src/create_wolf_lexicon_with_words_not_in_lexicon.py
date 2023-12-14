import pandas as pd


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
    return df


def read_wolf_lexicon(wolf_lexicon):
    df = pd.read_csv(wolf_lexicon, sep='\t')
    df["lemma"] = df["lemma"].apply(lambda x: x.lower())
    return df


def remove_common_lemma(data_lexicon, data_wolf):
    result = data_wolf[~data_wolf['lemma'].isin(data_lexicon['lemma'])]
    return result


def save_the_new_dataframe(result):
    result.to_csv("wolf_with_words_not_in_lexicon.csv", sep="\t", index=False)


def main():
    data_lexicon = read_lexique("/data/macairec/PhD/Grammaire/dico/lexique_5_12_2023_11h.csv")
    data_wolf = read_wolf_lexicon("/data/macairec/PhD/Grammaire/dico/wolf/wolf_merge_with_lexicon.csv")
    result = remove_common_lemma(data_lexicon, data_wolf)
    save_the_new_dataframe(result)


if __name__ == '__main__':
    main()

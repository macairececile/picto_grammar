"""
Script to process the data from orfeo, and create train, dev, test sets.

Example of use: python create_train_dev_test_split_for_asr.py --files '/.../' --save 'corpus/'
"""

from sklearn.model_selection import train_test_split
from os import listdir
from os.path import join, isfile
import pandas as pd
from argparse import ArgumentParser, RawTextHelpFormatter


def get_files_csv_from_directory(path):
    """
        Get the files in a directory.

        Arguments
        ---------
        path: str

        Returns
        -------
        A list of files.
    """
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files


def read_corpus_file(csv_file):
    """
        Read the csv file in a dataframe.

        Arguments
        ---------
        csv_file: str

        Returns
        -------
        A dataframe.
    """
    corpus = pd.read_csv(csv_file, sep='\t')
    corpus = corpus[corpus['text'] != '']
    return corpus


def shuffle_and_split(corpus_name, dataframe):
    """
        Shuffle the dataframe, and split equally the corpus

        Arguments
        ---------
        corpus_name: name of the corpus
        dataframe: dataframe

        Returns
        -------
        Train, Valid, and Test sets.
    """
    df_shuffle = dataframe.sample(frac=1).reset_index(drop=True)
    df_shuffle['corpus_name'] = df_shuffle.shape[0] * [corpus_name]
    train, test = train_test_split(df_shuffle, test_size=0.2, random_state=42)
    validation, test = train_test_split(test, test_size=0.5, random_state=42)
    return train, validation, test


def create_train_dev_test(args):
    """
        Full pipeline to create train, dev, test sets from csv files with data.
    """
    files = get_files_csv_from_directory(args.files)
    splits_train, splits_valid, splits_test = [], [], []
    for f in files:
        data = read_corpus_file(args.files + f)
        corpus_name = f[f.index('corpus_') + len('corpus_'):f.index('.csv')]
        train, valid, test = shuffle_and_split(corpus_name, data)
        splits_train.append(train)
        splits_valid.append(valid)
        splits_test.append(test)
    train_data = pd.concat(splits_train, axis=0)
    valid_data = pd.concat(splits_valid, axis=0)
    test_data = pd.concat(splits_test, axis=0)
    train_data.to_csv(
        args.save + "train_grammar.csv",
        sep='\t',
        index=False)
    valid_data.to_csv(
        args.save + "valid_grammar.csv",
        sep='\t',
        index=False)
    test_data.to_csv(
        args.save + "test_grammar.csv",
        sep='\t',
        index=False)


parser = ArgumentParser(description="Create splits from corpus.",
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('--files', type=str, required=True,
                    help="Data files in csv format.")
parser.add_argument('--save', type=str, required=True,
                    help="Directory to save the splits.")
parser.set_defaults(func=create_train_dev_test)
args = parser.parse_args()
args.func(args)

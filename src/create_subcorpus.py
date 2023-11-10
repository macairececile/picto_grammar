"""
Retrieve data belonging to the same sub-corpus and save them in a new .csv file.
The data file has to have a corpus_name column.

Example of use: python create_subcorpus.py --datafile 'test.csv' --save 'subcorpus/'
"""

import pandas as pd
from argparse import ArgumentParser, RawTextHelpFormatter


def read_tsv(csv_file):
    """
        Read the csv file in a dataframe.

        Arguments
        ---------
        csv_file: str

        Returns
        -------
        A dataframe.
    """
    return pd.read_csv(csv_file, sep='\t')


def create_subcorpus(args):
    """
        Create the data from data belonging to the same sub-corpus.
    """
    dataframe = read_tsv(args.datafile)
    dfs = {corpus: group for corpus, group in dataframe.groupby('corpus_name')}
    for corpus_name, df_corpus in dfs.items():
        df_corpus.to_csv(args.save + corpus_name + ".csv", index=False, header=True, sep='\t')


# parser = ArgumentParser(description="Create subcorpus",
#                         formatter_class=RawTextHelpFormatter)
# parser.add_argument('--datafile', type=str, required=True,
#                     help="Data files in csv format.")
# parser.add_argument('--save', type=str, required=True,
#                     help="Directory to save the subcorpus data.")
# parser.set_defaults(func=create_subcorpus)
# args = parser.parse_args()
# args.func(args)

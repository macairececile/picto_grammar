"""
Script to get the total duration in minutes of the corpus.

Example of use: python get_duration_data_from_tsv.py --csv 'corpus.csv' --path 'clips/'
"""

import pandas as pd
import librosa
from argparse import ArgumentParser, RawTextHelpFormatter


def get_duration(args):
    """
        Get the total duration in the corpus.
    """
    duration = 0
    data = pd.read_csv(args.csv, sep='\t')
    clips = data['clips'].values.tolist()
    for c in clips:
        t = librosa.get_duration(path=args.path + c + '.wav') / 60
        duration += t
    print("Duration associated to the corpus in minutes: ", duration)


if __name__ == "__main__":
    parser = ArgumentParser(description="Length of a data set in seconds.", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--csv", required=True,
                        help="Path of the .csv data file.")
    parser.add_argument("--path", required=True,
                        help="path")
    parser.set_defaults(func=get_duration)
    args = parser.parse_args()
    args.func(args)

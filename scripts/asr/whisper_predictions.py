"""
Script to run whisper and get predictions in a .txt file.
You need to install whisper library to run this script.

Example of use: python whisper_predictions --data 'corpus.csv' --save 'whisper_preds/' --clips 'corpus_clips/' --model 'large' --index 0
"""

import pandas as pd
from argparse import ArgumentParser, RawTextHelpFormatter
import whisper

punctuations = '!?\":,/.;()[]'
punc_table = str.maketrans({key: None for key in punctuations})


def read_tsv(csv_file):
    """
        Read csv file in a dataframe

        Arguments
        ---------
        csv_file: .csv file

        Returns
        -------
        A dataframe with the data.
    """
    return pd.read_csv(csv_file, sep='\t')


def run_whisper_and_get_prediction(args):
    """
        Run a specific whisper model and save predictions in a .txt file.

        Arguments
        ---------
        data: .csv file
        model_name: str
            Name of the whisper model ['small', 'base', 'medium', 'large']
        index: int
            Index to which sentence we start the predictions.
    """
    data = read_tsv(args.data)
    index = int(args.index)
    clips = data['clips'].values.tolist()[index:]
    model = whisper.load_model(args.model)
    f = open(args.save + args.model + "/out.txt", 'a')
    for i, c in enumerate(clips):
        path_clip = args.clips
        result = model.transcribe(path_clip + c + '.wav', language='fr')
        f.write(c + '\t' + result["text"] + '\n')
    f.close()


parser = ArgumentParser(description="Run whisper and get predictions in a .txt file.",
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('--data', type=str, required=True,
                    help="")
parser.add_argument('--save', type=str, required=True,
                    help="")
parser.add_argument('--clips', type=str, required=True,
                    help="")
parser.add_argument('--model', type=str, required=True, choices=['tiny', 'base', 'small', 'medium', 'large'],
                    help="")
parser.add_argument('--index', type=str, required=True,
                    help="")
parser.set_defaults(func=run_whisper_and_get_prediction)
args = parser.parse_args()
args.func(args)

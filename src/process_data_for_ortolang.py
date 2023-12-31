"""
Script to generate data in .json format for Ortolang from the one generated by the grammar.

Example of use: python process_data_for_ortolang.py --datafile 'corpus_grammar.csv' --path_clips 'clips/'
--outdir 'data_ortolang/'
"""

import pandas as pd
import shutil
from argparse import ArgumentParser, RawTextHelpFormatter


def create_json(csv_file):
    """
        Create .json file from the grammar output

        Arguments
        ---------
        csv_file: .csv file
    """
    dataframe = pd.read_csv(csv_file, sep='\t')
    data_for_json = dataframe[["clips", "text", 'pictos', 'tokens']]
    data_for_json = data_for_json.sort_values(by=['clips'], ignore_index=True)
    data_for_json['pictos'] = data_for_json['pictos'].apply(lambda x: eval(x))
    final_data = data_for_json.rename({'clips': 'audio', 'text': 'sentence', 'tokens': 'pictos_tokens'}, axis='columns')
    path_json = csv_file.split('/')[-1].split('_grammar_pictos.csv')[0] + '.json'
    final_data.to_json(path_json, orient='records')


def copy_clips_corpus(csv_file, path_clips, outdir):
    """
        Copy the clips in another directory

        Arguments
        ---------
        csv_file: .csv file
        path_clips: str
        outdir: str
    """
    dataframe = pd.read_csv(csv_file, sep='\t')
    clips = dataframe["clips"].values.tolist()
    for c in clips:
        name_clip = c + '.wav'
        shutil.copy(path_clips + name_clip, outdir + name_clip)


def create_data_for_ortolang(args):
    # corpus_name = ["cfpb", "cfpp", "clapi", "coralrom", "crfp", "fleuron", "frenchoralnarrative",
    # "ofrom", "reunions", "tcof", "tufs", "valibel"]
    # for i in corpus_name:
    copy_clips_corpus(
        args.datafile, args.path_clips, args.outdir)
    create_json(args.datafile)


# parser = ArgumentParser(description="Create data for Ortolang.",
#                         formatter_class=RawTextHelpFormatter)
# parser.add_argument('--datafile', type=str, required=True,
#                     help="")
# parser.add_argument('--path_clips', type=str, required=True,
#                     help="")
# parser.add_argument('--outdir', type=str, required=True,
#                     help="")
# parser.set_defaults(func=create_data_for_ortolang)
# args = parser.parse_args()
# args.func(args)

from sklearn.model_selection import train_test_split
from os import listdir
from os.path import join, isfile
import pandas as pd

def get_files_csv_from_directory(path):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files


def read_corpus_file(csv_file):
    corpus = pd.read_csv(csv_file, sep='\t')
    corpus = corpus[corpus['text'] != '']
    return corpus


def shuffle_and_split(corpus_name, dataframe):
    df_shuffle = dataframe.sample(frac=1).reset_index(drop=True)
    df_shuffle['corpus_name'] = df_shuffle.shape[0] * [corpus_name]
    train, test = train_test_split(df_shuffle, test_size=0.2, random_state=42)
    validation, test = train_test_split(test, test_size=0.5, random_state=42)
    return train, validation, test


def create_train_dev_test(path_files):
    files = get_files_csv_from_directory(path_files)
    splits_train, splits_valid, splits_test = [], [], []
    for f in files:
        data = read_corpus_file(path_files + f)
        corpus_name = f[f.index('corpus_') + len('corpus_'):f.index('.csv')]
        train, valid, test = shuffle_and_split(corpus_name, data)
        splits_train.append(train)
        splits_valid.append(valid)
        splits_test.append(test)
    train_data = pd.concat(splits_train, axis=0)
    valid_data = pd.concat(splits_valid, axis=0)
    test_data = pd.concat(splits_test, axis=0)
    train_data.to_csv(
        "train_grammar.csv",
        sep='\t',
        index=False)
    valid_data.to_csv(
        "valid_grammar.csv",
        sep='\t',
        index=False)
    test_data.to_csv(
        "test_grammar.csv",
        sep='\t',
        index=False)


if __name__ == '__main__':
    create_train_dev_test("/data/macairec/PhD/Grammaire/corpus/csv/orfeo/")
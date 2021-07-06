'''設定周りのファイル'''
import pandas as pd
pd.set_option('display.max_columns', 500)


class Config:
    def __init__(self):
        self.train_file = '../data/train.tsv'
        self.test_file = '../data/test.tsv'
        self.ans_file = '../data/answer.csv'

    def read_data(self):
        return pd.read_table(self.train_file), pd.read_table(self.test_file)

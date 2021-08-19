'''初期設定ファイル'''
import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/bank')
import pandas as pd
pd.set_option('display.max_columns', 50)


# 訓練データ
train_data = pd.read_csv('data/train.csv')
# テストデータ
test_data = pd.read_csv('data/test.csv')

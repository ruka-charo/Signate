import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/income/split_file')

import lightgbm as lgb

from config import Config
from preprocess_class import Preprocess
from train_run import Train
from test_run import Test


#%% データの読み込み
config = Config()
train_data, test_data = config.read_data()

#%% データ前処理
pre = Preprocess()
X_learn, X_val, y_learn, y_val = pre.preprocessing(train_data, test_data)

#%% 学習と評価
score = Train()
score.train(X_learn, X_val, y_learn, y_val)

#%% テストデータで予測、Submit
test = Test()
test.test_run(score.model, pre.X_test, test_data)

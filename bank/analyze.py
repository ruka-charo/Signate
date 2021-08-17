'''銀行の顧客ターゲティング'''
import os, sys
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/bank')
sys.path.append('../..')
import numpy as np
np.set_printoptions(precision=3)
import pandas as pd
pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import japanize_matplotlib
from IPython.display import display

import lightgbm as lgb

from Function.preprocess_func import *
from function import *
import config

# ----------------------------------------
# marital   : 未婚、既婚
# education : 教育水準
# default   : 債務不履行があるか
# balance   : 年間平均残高
# housing   : 住宅ローン
# loan      : 個人ローン
# contact   : 連絡方法
# day, month: 最終接触月日
# duration  : 最終接触時間(s)
# campaign  : 現キャンペーンにおける接触回数
# previous  : 現キャンペーン以前までに顧客に接触した回数
# poutcome  : 前回のキャンペーンの成果
# y         : 定期預金申し込み有無（1:有り、0:無し）
# ----------------------------------------

# 変数定義
category_features = config.category_features
train_data = config.train_data
test_data = config.test_data
display(train_data.head())


# データの準備
train = train_data.copy().drop(['id'], axis=1)
X_test = test_data.copy().drop(['id'], axis=1)

#%% データ前処理
# カテゴリ変数を変換する
train_en, X_test_en = encoding('label', train, X_test, category_features)

# 月を対応する数字に置き換える
train_en = month_replace(train_en)
X_test_en = month_replace(X_test_en).drop(['y'], axis=1)
display(train_en.head())


#%% ダウンサンプリングの準備
train_0 = train_en.query('y == 0')
train_1 = train_en.query('y == 1')

pred_score = []
#%% 学習と予測
for i in range(29):
    if i == 28: # 端数の処理
        train_batch_0 = train_0[50*i: 1450]
    else:
        train_batch_0 = train_0[50*i: 50*(i+1)]

    # 訓練データ
    X_learn, X_val, y_learn, y_val = make_batch_data(train_batch_0, train_1)


    '''モデル学習'''
    # LightGBMを用いる
    clf = lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=1000)
    clf.fit(X_learn, y_learn, eval_set=[(X_val, y_val)],
            early_stopping_rounds=100)

    clf.predict_proba(X_test_en, num_iteration=clf.best_iteration_)[:, 1].tolist()
    pred_score.append(y_pred)


# 予測値の平均を最終結果にする
y_pred = np.mean(np.array(pred_score).T, axis=1)


'''投稿用ファイルの作成'''
#%% csvファイルの作成
id = test_data['id']
ans = answer_csv(y_pred, id)
display(ans.head())

# ファイルの保存
ans.to_csv('data/answer.csv', header=False, index=False)

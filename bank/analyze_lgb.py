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



# 変数定義 ====================================
# Encoding用カテゴリ変数
train_data = config.train_data
test_data = config.test_data

category_features = ['job', 'marital', 'default', 'housing',
                    'education', 'loan', 'contact', 'poutcome']
drop_features = ['id'] # 使用しない特徴量

# ============================================


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
            early_stopping_rounds=100, verbose=False)

    # 変数重要度
    lgb_importance(train_0, clf)

    y_pred = clf.predict_proba(X_test_en, num_iteration=clf.best_iteration_)[:, 1].tolist()
    pred_score.append(y_pred)


# 予測値の平均を最終結果にする
y_pred = np.mean(np.array(pred_score).T, axis=1)


'''投稿用ファイルの作成'''
#%% csvファイルの作成
id = test_data['id']
ans = answer_csv(y_pred, id)
display(ans.head())

# ファイルの保存
ans.to_csv('data/answer_lgb.csv', header=False, index=False)

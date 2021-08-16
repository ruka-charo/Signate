'''銀行の顧客ターゲティング'''
import os, sys
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/bank')
sys.path.append('../..')
import pandas as pd
pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import japanize_matplotlib
from IPython.display import display

import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score

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


#%% データの準備
X_train = train_data.copy().drop(['id', 'y'], axis=1)
y_train = train_data['y']
X_test = test_data.copy().drop(['id'], axis=1)

#%% データ前処理
# カテゴリ変数を変換する
X_train_en, X_test_en = encoding('label', X_train, X_test, category_features)

# 月を対応する数字に置き換える
X_train_en = month_replace(X_train_en)
X_test_en = month_replace(X_test_en)
display(X_train_en.head())

#%% データを訓練用と検証用に分割する
X_learn, X_val, y_learn, y_val = train_test_split(X_train_en, y_train,
                stratify=y_train, test_size=0.2, shuffle=True, random_state=0)


'''モデル構築'''
#%% LightGBMを用いる
clf = lgb.LGBMClassifier(objective='binary', metric='auc')
# cross_val_scoreで精度のバラつきを確認する
print('交差検証:', cross_val_score(clf, X_train_en, y_train, cv=5, scoring='roc_auc'))

#%% 学習
clf.fit(X_learn, y_learn, eval_set=[(X_val, y_val)],
        early_stopping_rounds=10, verbose=10)

# 変数重要度
lgb_importance(X_learn, clf)

# 予測
y_pred = clf.predict_proba(X_test_en, num_iteration=clf.best_iteration_)[:, 1]


'''投稿用ファイルの作成'''
#%% csvファイルの作成
id = test_data['id']
ans = answer_csv(y_pred, id)
display(ans.head())

# ファイルの保存
ans.to_csv('data/answer.csv', header=False, index=False)

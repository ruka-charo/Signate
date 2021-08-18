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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from Function.preprocess_func import *
from function import *
import config



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
train_en, X_test_en = encoding('onehot', train, X_test, category_features)

# 月を対応する数字に置き換える
train_en = month_replace(train_en)
X_test_en = month_replace(X_test_en).drop(['y'], axis=1)
display(train_en.head())


#%% ダウンサンプリングの準備
train_0 = train_en.query('y == 0')
train_1 = train_en.query('y == 1')

auc_score = []
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
    # LogisticRegressionで学習
    pipe = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', C=0.001))
    pipe.fit(X_learn, y_learn)
    val_pred = pipe.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y_val, val_pred)
    auc_score.append(auc)
    print(f'{i+1}回目: {auc}')

    # 予測
    y_pred = pipe.predict_proba(X_test_en)[:, 1].tolist()
    pred_score.append(y_pred)

print(f'aucスコアの平均値: {np.mean(auc_score)}')


# 予測値の平均を最終結果にする
y_pred = np.mean(np.array(pred_score).T, axis=1)



'''投稿用ファイルの作成'''
#%% csvファイルの作成
id = test_data['id']
ans = answer_csv(y_pred, id)
display(ans.head())

# ファイルの保存
ans.to_csv('data/answer_log.csv', header=False, index=False)

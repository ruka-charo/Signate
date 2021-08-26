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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tensorflow.keras import models, layers, optimizers

from Function.preprocess_func import *
from function import *
import config



# 変数定義 ====================================
# Encoding用カテゴリ変数
train_data = config.train_data
test_data = config.test_data

category_features = ['job', 'marital', 'housing','default', 'balance',
                    'loan', 'contact', 'poutcome', 'education']
drop_features = ['id'] # 使用しない特徴量

replace_edu = 'tertiary' # 欠損値の補完
replace_con = 'cellular' # 欠損値の補完

# ============================================


# データの準備
train = train_data.copy().drop(drop_features, axis=1)
X_test = test_data.copy().drop(drop_features, axis=1)

#%% データ前処理
# 欠損値を最頻値で置換する
#train['education'].replace('unknown', replace_edu, inplace=True)
#X_test['education'].replace('unknown', replace_edu, inplace=True)

#train['contact'].replace('unknown', replace_con, inplace=True)
#X_test['contact'].replace('unknown', replace_con, inplace=True)

# カテゴリ変数を変換する
train_en, X_test_en = encoding('onehot', train, X_test, category_features)

# 月とeducationを対応する数字に置き換える(label encoder)
# educationのunknownは最頻値であるsecondaryに置換する
train_en = month_replace(train_en)
X_test_en = month_replace(X_test_en).drop(['y'], axis=1)

#train_en = education_replace(train_en)
#X_test_en = education_replace(X_test_en).drop(['y'], axis=1)
display(train_en.head())


#%% ダウンサンプリングの準備
train_0 = train_en.query('y == 0')
train_1 = train_en.query('y == 1')

auc_score = []
pred_score = []

epochs = 20

#%% 学習と予測
for i in range(29):
    if i == 28: # 端数の処理
        train_batch_0 = train_0[50*i: 1450]
    else:
        train_batch_0 = train_0[50*i: 50*(i+1)]

    # 訓練データ
    X_learn, X_val, y_learn, y_val = make_batch_data(train_batch_0, train_1)


    '''モデル学習'''
    # ニューラルネットワークで解いてみる
    model = models.Sequential()
    model.add(layers.Dense(X_learn.shape[0], activation='sigmoid')) # 隠れ層
    model.add(layers.Dense(1, activation='sigmoid')) # 出力層

    optimizer = optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[roc_auc])

    history = model.fit(X_learn, y_learn,
                        epochs=epochs,
                        batch_size=10,
                        verbose=1,
                        validation_data=(X_val, y_val))

    # 予測
    y_pred = model.predict(X_test_en).tolist()
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

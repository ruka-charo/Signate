%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/Forest_Fire
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt

import category_encoders as ce
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import r2_score


#%% データの読み込み
train_data = pd.read_table('./data/train.tsv')
test_data_origin = pd.read_table('./data/test.tsv')
train_data.head()


'''データ前処理'''
#%% 訓練データとテストデータで同じ処理を行うため、一旦結合する
# train: 0 <= index <= 257, test: 258 <= index <= 516
data = pd.concat([train_data, test_data_origin], ignore_index=True)
data.tail()

#%% 月と曜日を数値に変換
month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
month_num = [i for i in range(1, 13)]
day_num = [i for i in range(1, 8)]

data = data.copy().replace(month, month_num).replace(day, day_num)

#%% x, y座標を原点からの距離に変換して新しい特徴量にする
data['dist'] = np.sqrt(data['X']**2 + data['Y']**2)
data = data.drop(['X', 'Y'], axis=1)
data.head()

#%% 訓練データとテストデータに分ける
train_data = data[:258]
test_data = data[258:].drop(['id', 'area'], axis=1)

#%% 目的変数と説明変数を分ける
X = train_data.copy().drop(['id', 'area'], axis=1)
y = pd.DataFrame(train_data.copy()['area'], columns=['area'])

#%% 外れ値が一個ある(index=83)
# 外れ値は削除しておく
train_data.query('area > 1000')

X = X.drop(index=83)
y = y.drop(index=83)

#%% 訓練データと検証データに分類する
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
X_train.shape, X_val.shape


'''モデル作成'''
#%% 扱うデータをxgbのデータ構造にする
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(test_data)

#%% 学習の実行
params = {'objective': 'reg:squarederror', 'silent': 1}
watch_list = [(dtrain, 'train'), (dval, 'eval')]

model = xgb.train(params, dtrain, 100, evals=watch_list, early_stopping_rounds=10)

# バリデーションでスコアの確認
val_pred = model.predict(dval)
print('r2_score:', r2_score(y_val, val_pred))


#%% テストデータの予測を行い、csvファイルに出力
y_test_pred = pd.DataFrame(model.predict(dtest), columns=['fire area'])
ans = pd.DataFrame(test_data_origin['id'])
ans = ans.merge(y_test_pred, right_index=True, left_index=True)
ans.head()

ans.to_csv('./data/answer.csv', header=False, index=False)

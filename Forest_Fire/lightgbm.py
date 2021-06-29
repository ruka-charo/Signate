%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/Forest_Fire
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import r2_score


#%% データの読み込み
train_data = pd.read_table('./data/train.tsv')
test_data = pd.read_table('./data/test.tsv')


'''データ前処理'''
#%% 月と曜日、rainはいらなそうなので削除する
train_data = pd.read_table('./data/train.tsv')
test_data = pd.read_table('./data/test.tsv')
data = pd.concat([train_data, test_data], ignore_index=True).drop(
    ['id', 'month', 'day', 'rain'], axis=1)
data.head()

# x, y座標を原点からの距離に変換して新しい特徴量にする
data['dist'] = np.sqrt(data['X']**2 + data['Y']**2)
data = data.drop(['X', 'Y'], axis=1)

# 訓練データとテストデータに分ける
train_data_fix = data[:258]
test_data_fix = data[258:].drop(['area'], axis=1)

#%% 目的変数と説明変数を分ける
X = train_data_fix.copy().drop(['area'], axis=1)
y = pd.DataFrame(train_data_fix.copy()['area'], columns=['area'])


#%% 外れ値が一個ある(index=83) 削除しておく
train_data.query('area > 1000')

X = X.drop(index=83)
y = y.drop(index=83)

# 訓練データと検証データに分類する
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_train.shape, X_val.shape


'''モデル作成'''
#%% モデルを構築する
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print('r2_score:', r2_score(y_val, y_pred))

#%% 変数重要度
features = X_train.columns
importances = model.feature_importances_
indices = np.argsort(-importances)
plt.bar(np.array(features)[indices], np.array(model.feature_importances_[indices]))
plt.show()

#%% テストデータの予測を行い、csvファイルに出力
y_test_pred = pd.DataFrame(model.predict(test_data_fix))
ans = pd.DataFrame(test_data['id'])
ans = ans.merge(y_test_pred, right_index=True, left_index=True)

ans.to_csv('./data/answer2.csv', header=False, index=False)

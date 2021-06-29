%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/Forest_Fire
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


#%% データの読み込み
train_data = pd.read_table('./data/train.tsv')
test_data = pd.read_table('./data/test.tsv')

train_data.head()
train_data.shape, test_data.shape
train_data.describe()

# 欠損値の確認
train_data.isnull().any()
test_data.isnull().any()

#%% 月と曜日を数値に変換
month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
        'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
month_num = [i for i in range(1, 13)]
day_num = [i for i in range(1, 8)]

train_sample = train_data.copy().replace(month, month_num).replace(day, day_num)
train_sample.head()


#%% 座標と焼失面積を可視化する
x = train_sample['X'].values
y = train_sample['Y'].values
size = train_sample['area'].values

plt.scatter(x, y, s=size)
plt.show()


#%% 各説明変数と焼失面積の関係を可視化する。
for column in train_sample.columns[3:13]:
    plt.scatter(train_sample[column], train_sample['area'])
    plt.title('{}'.format(column))
    plt.ylim(0, 400)
    plt.show()

'''
わかったこと
・4月前後と8月前後が焼失面積が大きい
・曜日に偏りはあまりみられない(当然？)
・FFMC = 90 くらいで若干大きい
・DC は大きい方が焼失面積も大きい傾向
・ISI = 10 あたりでピークをとる
・気温による変化はあまりない
・RH = 40 くらいでピークをとる
・wind = 4 くらいでピークをとる
・rain = 0 で焼失面積大(当然)
'''


'''データ前処理'''
#%% 訓練データとテストデータで同じ処理を行うため、一旦結合する
# train: 0 <= index <= 257, test: 258 <= index <= 516
data = pd.concat([train_data, test_data], ignore_index=True)
data.tail()

# monthとdayをonehot化する
data = pd.get_dummies(data)
data.head()

# 訓練データとテストデータに分ける
train_data = data[:258]
test_data = data[258:].drop(['id', 'area'], axis=1)

#%% 目的変数と説明変数を分ける
X = train_data.copy().drop(['id', 'area'], axis=1)
y = pd.DataFrame(train_data.copy()['area'], columns=['area'])

# 説明変数の標準化
ss = StandardScaler()
X_std = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
X_std.head()

# テストデータの標準化
test_data_std = pd.DataFrame(ss.transform(test_data), columns=test_data.columns)
test_data_std.head()


#%% 外れ値が一個ある(index=83)
# 線形回帰は外れ値の影響を受けやすいので削除しておく
train_data.query('area > 1000')

X = X.drop(index=83)
y = y.drop(index=83)

# 訓練データと検証データに分類する
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_train.shape, X_val.shape


'''モデル作成'''
#%% 線形回帰モデルを構築する
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_val)

print('R2-score:', r2_score(y_val, y_pred)) # 相当ダメなモデルらしい

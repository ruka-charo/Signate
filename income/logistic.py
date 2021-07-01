%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/income
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


#%% データ読み込み
train_data = pd.read_table('./data/train.tsv')
test_data = pd.read_table('./data/test.tsv')

train_data.head()
train_data.shape

train_data['workclass'].unique()
train_data['education'].unique()
train_data['marital-status'].unique()
train_data['occupation'].unique()
train_data['relationship'].unique()
train_data['race'].unique()
train_data['native-country'].unique()

#%% 欠損値の確認
train_data.isnull().any() # なし
test_data.isnull().any() # なし

# 直感的に収入と関係がありそうな項目を調査
#%% 性別と収入
data = train_data[['sex', 'Y']]
data_male = data.query('sex == "Male"')
data_female = data.query('sex == "Female"')

plt.hist(data_male['Y'])
plt.hist(data_female['Y'])
plt.show()

#%% 労働時間と収入
data = train_data[['hours-per-week', 'Y']]
plt.hist(data['hours-per-week'], density=True)
plt.show()


'''データ前処理'''
#%% 訓練データの目的変数を0, 1に変換する
train_data = train_data.replace({'<=50K': 0, '>50K': 1})
# 職業不明は削除
train_data = train_data.drop(index=train_data.query('occupation == "?"').index)

#%% trainデータとtestデータを結合する
mix_data = pd.concat([train_data, test_data], ignore_index=True)
mix_data

#%% 関係なさそうなものをdropする
# 初めは「id」, 「marital-status」, 「relationship」, 「native-country」をdropする
mix_drop = mix_data.copy().drop(['id', 'marital-status', 'relationship', 'native-country'], axis=1)
mix_drop.head()

#%% カテゴリ変数をonehot化する
ohe_df = pd.get_dummies(mix_drop[['education', 'workclass', 'occupation', 'race', 'sex']],
                        drop_first=True)
mix_com = mix_drop.copy().drop(['education', 'workclass', 'occupation', 'race', 'sex'], axis=1).merge(
    ohe_df, right_index=True, left_index=True)

mix_com.head()
mix_com.shape

#%% 訓練データとテストデータを分ける
train = mix_com.query('Y != "NaN"')
test = mix_com.query('Y == "NaN"')


#%% 訓練データを説明変数と目的変数に分ける
X_train = train.copy().drop(['Y'], axis=1)
y_train = train['Y']

X_test = test.copy().drop(['Y'], axis=1)

#%% データの標準化
ss = StandardScaler()

# 訓練データの標準化
X_train_std = pd.DataFrame(ss.fit_transform(X_train),
                        columns=X_train.columns)
X_train_std.head()
# テストデータの標準化
X_test_std = pd.DataFrame(ss.transform(X_test),
                        columns=X_test.columns)
X_test_std.head()


'''モデルの学習と評価'''
#%% 訓練データと検証データに分ける
X_learn, X_val, y_learn, y_val = train_test_split(X_train_std, y_train, test_size=0.2)

#%% モデルの学習と評価
lr = LogisticRegression()
lr.fit(X_learn, y_learn)
y_pred = lr.predict(X_val)

print('Accuracy_score:', accuracy_score(y_val, y_pred))
print('confusion_matrix:\n', confusion_matrix(y_val, y_pred))


'''テストデータで予測, Submit'''
#%% テストデータで予測し、csvファイルで保存
y_pred_test = pd.DataFrame(lr.predict(X_test_std), columns=['income'])
ans = pd.DataFrame(test_data['id'], columns=['id'])
ans = ans.merge(y_pred_test, right_index=True, left_index=True)
ans['income'] = ans['income'].replace({0: '<=50K', 1: '>50K'})
ans.head()

ans.to_csv('./data/answer.csv', header=False, index=False)

%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/income
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt


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
# 初めは「marital-status」, 「relationship」, 「native-country」をdropする
mix_drop = mix_data.copy().drop(['marital-status', 'relationship', 'native-country'], axis=1)
mix_drop.head()

#%% カテゴリ変数をonehot化する
ohe_df = pd.get_dummies(mix_drop[['education', 'workclass', 'occupation', 'race', 'sex']],
                        drop_first=True)
mix_com = mix_drop.copy().drop(['education', 'workclass', 'occupation', 'race', 'sex'], axis=1).merge(
    ohe_df, right_index=True, left_index=True)

mix_com.head()
mix_com.shape

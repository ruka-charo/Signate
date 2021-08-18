'''データの把握'''
import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/bank')
import pandas as pd
pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import japanize_matplotlib
from IPython.display import display

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

# 訓練データの読み込み
train_data = config.train_data
test_data = config.test_data
display(train_data.head())
print(train_data.shape)
print(test_data.shape)


#%% 各特徴量のヒストグラムを作成
for column in train_data.columns:
    if column == 'id':
        continue
    plt.hist(train_data[column])
    plt.title(f'{column} (train)')
    plt.xlabel(column)
    plt.ylabel('counts')
    plt.show()

    if column == 'y':
        break
    plt.hist(test_data[column])
    plt.title(f'{column} (test)')
    plt.xlabel(column)
    plt.ylabel('counts')
    plt.show()

#%% クロス集計表を用いて影響力のある因子を把握
for column in train_data.columns:
    if column == 'id' or column == 'age' or column == 'balance' or column == 'duration':
        continue
    display(pd.crosstab(train_data['y'], train_data[column], normalize='columns'))


train_data.query('y == 1').shape

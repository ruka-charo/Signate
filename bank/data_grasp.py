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


# 訓練データの読み込み
train_data = config.train_data
display(train_data.head())
print(train_data.shape)

#%% 各特徴量のヒストグラムを作成
for column in train_data.columns:
    if column == id:
        continue

    else:
        plt.hist(train_data[column])
        plt.title(column)
        plt.xlabel(column)
        plt.ylabel('counts')
        plt.show()

#%% クロス集計表を用いて影響力のある因子を把握
for column in train_data.columns:
    if column == 'id' or column == 'age' or column == 'balance' or column == 'duration':
        continue
    display(pd.crosstab(train_data['y'], train_data[column], normalize='columns'))

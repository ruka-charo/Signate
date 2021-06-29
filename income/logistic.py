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

#%% 欠損値の確認
train_data.isnull().any() # なし
test_data.isnull().any() # なし

# 直感的に収入と関係がありそうな項目を調査
#%% 性別と収入
data = train_data[['sex', 'Y']]
data_male = data.query('sex == "Male"')
data_female = data.query('sex == "Female"')

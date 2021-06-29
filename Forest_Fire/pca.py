%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/Forest_Fire
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score


#%% データの読み込み
train_data = pd.read_table('./data/train.tsv')
test_data = pd.read_table('./data/test.tsv')


'''データ前処理'''
#%% 訓練データとテストデータで同じ処理を行うため、一旦結合する
# train: 0 <= index <= 257, test: 258 <= index <= 516
data = pd.concat([train_data, test_data], ignore_index=True)
data.tail()

# monthとdayをonehot化する
data = pd.get_dummies(data, drop_first=True)
data.head()



#%% 目的変数と説明変数を分ける
X = data.copy().drop(['id', 'area'], axis=1)
y = pd.DataFrame(data.copy()['area'], columns=['area'])
X.head()

# 説明変数の標準化
ss = StandardScaler()
X_std = pd.DataFrame(ss.fit_transform(X), columns=X.columns)
X_std.head()

#%% 主成分分析
pca = PCA()
pca.fit_transform(X_std)
# 主成分の確認
np.cumsum(pca.explained_variance_ratio_)

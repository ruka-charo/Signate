import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/income')
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline


#%% データ読み込み
train_data = pd.read_table('./data/train.tsv')
test_data = pd.read_table('./data/test.tsv')


'''データ前処理'''
#%% 訓練データの目的変数を0, 1に変換する
train_data = train_data.replace({'<=50K': 0, '>50K': 1})
# 職業不明は削除
train_data = train_data.drop(index=train_data.query('occupation == "?"').index)


#%% 関係なさそうなものをdropする
# 初めは「id」, 「marital-status」, 「relationship」, 「native-country」をdropする
train_drop = train_data.copy().drop(['id', 'marital-status', 'relationship', 'native-country'], axis=1)
test_drop = test_data.copy().drop(['id', 'marital-status', 'relationship', 'native-country'], axis=1)

X_train = train_drop.copy().drop(['Y'], axis=1)
y_train = train_drop['Y']
X_test = test_drop.copy()


#%% パイプラインの作成
label_features = ['education', 'workclass', 'occupation', 'race', 'sex']
pipe = make_pipeline(ce.OneHotEncoder(cols=label_features, handle_unknown='impute'),
                    StandardScaler(),
                    LogisticRegression())

'''モデルの学習と評価'''
#%% 訓練データと検証データに分ける(交差検証)
cross_val_score(pipe, X_train, y_train, cv=5)


ohe = ce.OneHotEncoder(cols=label_features, handle_unknown='impute')
ohe.fit(X_train)
X_train_2 = ohe.transform(X_train)
X_test_2 = ohe.transform(X_test)
X_train_2.shape, X_test_2.shape
X_train_2.head()

ohe.fit(X_test)
X_train_2 = ohe.transform(X_train)
X_test_2 = ohe.transform(X_test)
X_train_2.shape, X_test_2.shape
X_train_2.head()
X_test_2.head()


#%% モデルの訓練
pipe.fit(X_train, y_train)

pipe.predict(X_test)
'''テストデータで予測, Submit'''
#%% テストデータで予測し、csvファイルで保存
y_pred_test = pd.DataFrame(pipe.predict(X_test), columns=['income'])
ans = pd.DataFrame(test_data['id'], columns=['id'])
ans = ans.merge(y_pred_test, right_index=True, left_index=True)
ans['income'] = ans['income'].replace({0: '<=50K', 1: '>50K'})
ans.head()

#ans.to_csv('./data/answer.csv', header=False, index=False)

import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/income/lightgbm')
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

from preprocess import *
import config


# 変数の設定 ============================
label_features = config.label_features

# lightgbm
objective = 'binary'
metrics = 'binary_logloss'
# =====================================

#%% データ読み込み
train_data = pd.read_table('../data/train.tsv')
test_data = pd.read_table('../data/test.tsv')

#%% データ前処理
X_train, y_train, X_test = preprocessing(train_data, test_data)
# 訓練データと検証データに分ける
X_learn, X_val, y_learn, y_val = train_test_split(X_train, y_train, test_size=0.2)


'''lgbモデルの作成'''
#%% 特徴量と目的変数をlgm専用のデータ型にする
lgb_train = lgb.Dataset(X_learn, label=y_learn, free_raw_data=False)
lgb_eval = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

# ハイパーパラメータの設定
params = {'objective': objective, 'verbose': 0, 'metrics': metrics}

#%% 学習の実行
model = lgb.train(params, lgb_train, 100,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=['train', 'valid'],
                categorical_feature=label_features,
                early_stopping_rounds=10,
                verbose_eval=10)

# バリデーションデータでのスコアを確認
val_pred = model.predict(X_val, num_iteration=model.best_iteration)
print('log_loss値:', log_loss(y_val, val_pred))
print('accuracy_score:', accuracy_score(y_val, val_pred.round()))
print('confusion_matrix\n', confusion_matrix(y_val, val_pred.round()))

#%% 変数重要度
# split: 頻度、gain: 目的関数の現象寄与率
importance_type = 'gain'
features = X_learn.columns
importances = model.feature_importance(importance_type=importance_type)
indices = np.argsort(-importances)
plt.bar(np.array(features)[indices], np.array(importances[indices]))
plt.xticks(rotation=90)
plt.title(importance_type)
plt.show()


'''テストデータで予測, Submit'''
#%% テストデータで予測し、csvファイルで保存
test_pred = model.predict(X_test, num_iteration=model.best_iteration)
pred_df = pd.DataFrame(test_pred.round(), columns=['income'])
ans = pd.DataFrame(test_data['id'], columns=['id'])
ans = ans.merge(pred_df, right_index=True, left_index=True)
ans['income'] = ans['income'].replace({0: '<=50K', 1: '>50K'})
ans.head()

ans.to_csv('./data/answer.csv', header=False, index=False)

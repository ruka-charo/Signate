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


'''lgbモデルの作成(sklearnAPI)'''
#%% lightgbmモデルの作成
model = lgb.LGBMClassifier(objective=objective, metrics=metrics)
model.fit(X_learn, y_learn, eval_set=[(X_val, y_val)],
        categorical_feature=label_features,
        early_stopping_rounds=10,
        verbose=10)

model.get_params() # パラメータの表示
cross_val_score(model, X_train, y_train, cv=5) # 交差検証

# バリデーションデータでのスコアを確認
val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
val_proba = model.predict_proba(X_val, num_iteration=model.best_iteration_)


model.score(X_val, y_val)
log_loss(y_val, val_proba)# fit時に表示されているのと同じ
print('confusion_matrix\n', confusion_matrix(y_val, val_pred))


#%% 変数重要度
features = X_learn.columns
importances = model.feature_importances_
indices = np.argsort(-importances)
plt.bar(np.array(features)[indices], np.array(importances[indices]))
plt.xticks(rotation=90)
plt.show()


'''テストデータで予測, Submit'''
#%% テストデータで予測し、csvファイルで保存
test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
pred_df = pd.DataFrame(test_pred, columns=['income'])
ans = pd.DataFrame(test_data['id'], columns=['id'])
ans = ans.merge(pred_df, right_index=True, left_index=True)
ans['income'] = ans['income'].replace({0: '<=50K', 1: '>50K'})
ans.head()

ans.to_csv('./data/answer.csv', header=False, index=False)

%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/income
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt

import category_encoders as ce
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


#%% データ読み込み
train_data = pd.read_table('./data/train.tsv')
test_data = pd.read_table('./data/test.tsv')


'''データ前処理'''
#%% trainデータとtestデータを結合する
mix_data = pd.concat([train_data, test_data], ignore_index=True)

# 訓練データの目的変数を0, 1に変換する
# 「？」は欠損値として扱う
mix_data = mix_data.replace({'<=50K': 0, '>50K': 1, '?': np.nan})
mix_data.head(10)

#%% 関係なさそうなものをdropする
# 初めは「id」, 「marital-status」, 「relationship」, 「native-country」をdropする
drop_features = ['id']
mix_drop = mix_data.copy().drop(drop_features, axis=1)
mix_drop.head()

#%% カテゴリ変数をlabelencordingする
label_features = ['workclass', 'education', 'occupation', 'race', 'sex',
                'marital-status', 'relationship', 'native-country']
oe = ce.OrdinalEncoder(cols=label_features, handle_unknown='ignore')
mix_ec = oe.fit_transform(mix_drop)

mix_ec.head()
oe.category_mapping

#%% 訓練データとテストデータを分ける
train = mix_ec.query('Y != "NaN"')
test = mix_ec.query('Y == "NaN"')

# 訓練データを説明変数と目的変数に分ける
X_train = train.copy().drop(['Y'], axis=1)
y_train = train['Y']

X_test = test.copy().drop(['Y'], axis=1)

# 訓練データと検証データに分ける
X_learn, X_val, y_learn, y_val = train_test_split(X_train, y_train, test_size=0.2)

'''xgboostモデル作成'''
#%% 特徴量と目的変数をxgboost専用のデータ構造に変換する
dtrain = xgb.DMatrix(X_learn, label=y_learn)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

#%% ハイパーパラメータの設定
params = {'objective': 'binary:logistic', 'silent': 1}
num_round = 100

#%% 学習と評価
# バリデーションデータをモデルに渡し、スコアをモニタリングする
watch_list = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(params, dtrain, num_round,
                evals=watch_list, early_stopping_rounds=10)

# バリデーションでスコアを確認
val_pred = model.predict(dval, ntree_limit=model.best_ntree_limit)
score = log_loss(y_val, val_pred)
print('log-loss_score:', score)
print('accuracy_score:', accuracy_score(y_val.values, val_pred.round()))
print('confusion_matrix:\n', confusion_matrix(y_val.values, val_pred.round()))

#%% 変数重要度
xgb.plot_importance(model, importance_type='weight' , show_values=True)


'''テストデータで予測, Submit'''
#%% テストデータで予測し、csvファイルで保存
test_pred = model.predict(dtest, ntree_limit=model.best_ntree_limit)
pred_df = pd.DataFrame(test_pred.round(), columns=['income'])
ans = pd.DataFrame(test_data['id'], columns=['id'])
ans = ans.merge(pred_df, right_index=True, left_index=True)
ans['income'] = ans['income'].replace({0: '<=50K', 1: '>50K'})
ans.head()

ans.to_csv('./data/answer.csv', header=False, index=False)

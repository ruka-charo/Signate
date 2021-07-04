%cd /Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/income
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt

import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping


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
# lgmモデルの変数重要度から「education」「sex」「race」「native country」は削除する
drop_features = ['id', 'education', 'sex', 'race', 'native-country']
mix_drop = mix_data.copy().drop(drop_features, axis=1)
mix_drop.head()

#%% カテゴリ変数をlabelencordingする
label_features = ['workclass', 'occupation', 'marital-status', 'relationship']
ohe = ce.OneHotEncoder(cols=label_features, handle_unknown='impute')
mix_ec = ohe.fit_transform(mix_drop)

oe.category_mapping
mix_ec.head()

#%% 訓練データとテストデータを分ける
train = mix_ec.query('Y != "NaN"')
test = mix_ec.query('Y == "NaN"')

# 訓練データを説明変数と目的変数に分ける
X_train = train.copy().drop(['Y'], axis=1)
y_train = train['Y']

X_test = test.copy().drop(['Y'], axis=1)

# データの標準化
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.transform(X_test)

# 訓練データと検証データに分ける
X_learn, X_val, y_learn, y_val = train_test_split(X_train_std, y_train, test_size=0.2)
X_learn.shape


'''ニューラルネットワークの作成'''
#%% 入力層のユニット数=43, 出力層のユニット数=1
init = 'glorot_normal'
model = Sequential()
model.add(Dense(30, activation='relu', kernel_initializer=init,
                input_dim=(X_learn.shape[1])))
model.add(Dense(20, activation='relu', kernel_initializer=init))
model.add(Dense(1, activation='sigmoid', kernel_initializer=init))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#%% 学習と評価
batch_size = 700
epochs = 50
es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
cp = ModelCheckpoint(filepath='./data/best_model.h5', monitor='val_loss',
                    save_best_only=True, verbose=1)

history = model.fit(X_learn, y_learn,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_val, y_val),
                    callbacks=[es, cp], shuffle=True)

# バリデーションデータでスコアを確認する
val_pred = model.predict(X_val)
print('log_loss:', log_loss(y_val, val_pred))
print('accuracy_score:', accuracy_score(y_val, val_pred.round()))
print('confusion_matrix\n', confusion_matrix(y_val, val_pred.round()))

#%% テストデータの予測、Submit
best_model = keras.models.load_model('./data/best_model.h5')

test_pred = best_model.predict(X_test)
pred_df = pd.DataFrame(test_pred.round(), columns=['income'])
ans = pd.DataFrame(test_data['id'], columns=['id'])
ans = ans.merge(pred_df, right_index=True, left_index=True)
ans['income'] = ans['income'].replace({0: '<=50K', 1: '>50K'})
ans.head()

ans.to_csv('./data/answer.csv', header=False, index=False)

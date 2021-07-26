import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/income/lightgbm')
import numpy as np
import pandas as pd
import category_encoders as ce

import config


# 変数の定義
drop_features = config.drop_features
label_features = config.label_features

'''データ前処理'''
def preprocessing(train_data, test_data):
    # trainデータとtestデータを結合する
    mix_data = pd.concat([train_data, test_data], ignore_index=True)

    # 訓練データの目的変数を0, 1に変換する
    # 「？」は欠損値として扱う
    mix_data = mix_data.replace({'<=50K': 0, '>50K': 1, '?': np.nan})

    # 関係なさそうなものをdropする
    mix_drop = mix_data.copy().drop(drop_features, axis=1)

    # カテゴリ変数をlabelencordingする
    oe = ce.OrdinalEncoder(cols=label_features, handle_unknown='ignore')
    mix_ec = oe.fit_transform(mix_drop)

    # 訓練データとテストデータを分ける
    train = mix_ec.query('Y != "NaN"')
    test = mix_ec.query('Y == "NaN"')

    # 訓練データを説明変数と目的変数に分ける
    X_train = train.copy().drop(['Y'], axis=1)
    y_train = train['Y']

    X_test = test.copy().drop(['Y'], axis=1)

    return X_train, y_train, X_test

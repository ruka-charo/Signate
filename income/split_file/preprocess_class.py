'''前処理ファイル'''
import pandas as pd
import numpy as np
import category_encoders as ce

from sklearn.model_selection import train_test_split

class Preprocess:
    def __init__(self):
        self.drop_features = ['id']
        self.label_features = ['workclass', 'education', 'occupation', 'race',
                    'sex', 'marital-status', 'relationship', 'native-country']

    # 前処理の実行
    def preprocessing(self, train, test):
        mix_data = self._concat_data(train, test)
        mix_data = self._impute_nan(mix_data)
        mix_data = self._drop_columns(mix_data)
        mix_data_std = self._category_labeling(mix_data)
        return self._data_split(mix_data_std)


    # データの結合
    def _concat_data(self, train, test):
        return pd.concat([train ,test], ignore_index=True)

    # 欠損値と目的変数の置換
    def _impute_nan(self, data):
        return data.replace({'<=50K': 0, '>50K': 1, '?': np.nan})

    # 不必要なカラムの削除
    def _drop_columns(self, data):
        return data.copy().drop(self.drop_features, axis=1)

    # ラベルエンコードする
    def _category_labeling(self, data):
        oe = ce.OrdinalEncoder(cols=self.label_features, handle_unknown='impute')
        return oe.fit_transform(data)

    # train, val, testに分ける
    def _data_split(self, data):
        train = data.query('Y != "NaN"')
        test = data.query('Y == "NaN"')

        # 訓練データを説明変数と目的変数に分ける
        X_train = train.copy().drop(['Y'], axis=1)
        y_train = train['Y']

        self.X_test = test.copy().drop(['Y'], axis=1)

        # 訓練データと検証データに分ける
        return train_test_split(X_train, y_train, test_size=0.2)

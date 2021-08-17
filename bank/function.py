'''関数まとめ'''
import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/Signate/bank')
import pandas as pd
pd.set_option('display.max_columns', 50)
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import japanize_matplotlib
from IPython.display import display

from sklearn.model_selection import train_test_split

import config


def month_replace(df):
    df['month'].replace('jan', 1, inplace=True)
    df['month'].replace('feb', 2, inplace=True)
    df['month'].replace('mar', 3, inplace=True)
    df['month'].replace('apr', 4, inplace=True)
    df['month'].replace('may', 5, inplace=True)
    df['month'].replace('jun', 6, inplace=True)
    df['month'].replace('jul', 7, inplace=True)
    df['month'].replace('aug', 8, inplace=True)
    df['month'].replace('sep', 9, inplace=True)
    df['month'].replace('oct', 10, inplace=True)
    df['month'].replace('nov', 11, inplace=True)
    df['month'].replace('dec', 12, inplace=True)

    return df


def make_batch_data(df_1, df_2, test_size=0.2):
    train_batch = pd.concat([df_1, df_2])
    X_train_batch = train_batch.copy().drop(['y'], axis=1)
    y_train_batch = train_batch['y']
    # 検証用データの作成
    X_learn, X_val, y_learn, y_val = train_test_split(X_train_batch, y_train_batch,
        stratify=y_train_batch, test_size=test_size, shuffle=True, random_state=0)

    return X_learn, X_val, y_learn, y_val


# テストデータの予測
def answer_csv(y_pred, id):
    pred_df = pd.DataFrame(y_pred)
    ans = pd.DataFrame(id)
    ans = ans.merge(pred_df, right_index=True, left_index=True)

    return ans

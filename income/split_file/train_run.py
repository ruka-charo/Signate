'''学習ファイル'''
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from preprocess_class import Preprocess


class Train:
    def __init__(self):
        # ハイパーパラメータの設定
        self.params = {'objective': 'binary', 'verbose': 0, 'metrics': 'binary_logloss'}
        # カテゴリラベルの読み込み
        self.pre = Preprocess()
        self.label_features = self.pre.label_features

    # 訓練パートの実行
    def train(self, X_learn, X_val, y_learn, y_val):
        self._model_making(X_learn, X_val, y_learn, y_val)
        self._fit(self.lgb_train, self.lgb_eval)
        self._evaluation(X_val, y_val)


    # lgbモデルの作成
    def _model_making(self, X_learn, X_val, y_learn, y_val):
        # 特徴量と目的変数をlgm専用のデータ型にする
        self.lgb_train = lgb.Dataset(X_learn, label=y_learn, free_raw_data=False)
        self.lgb_eval = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    # 学習の実行
    def _fit(self, lgb_train, lgb_eval):
        self.model = lgb.train(self.params, lgb_train, 100,
                        valid_sets=[lgb_train, lgb_eval],
                        valid_names=['train', 'valid'],
                        categorical_feature=self.label_features,
                        early_stopping_rounds=10,
                        verbose_eval=10)

    # バリデーションで評価
    def _evaluation(self, X_val, y_val):
        # バリデーションデータでのスコアを確認
        val_pred = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        print('log_loss値:', log_loss(y_val, val_pred))
        print('accuracy_score:', accuracy_score(y_val, val_pred.round()))
        print('confusion_matrix\n', confusion_matrix(y_val, val_pred.round()))

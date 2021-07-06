'''テスト予測ファイル'''
from config import Config
import pandas as pd

class Test:
    def __init__(self):
        self.config = Config()
        self.ans_file = self.config.ans_file

    def test_run(self, model, X_test, test_data):
        self._test_predict(model, X_test)
        self._submit(test_data)


    # テストデータで予測
    def _test_predict(self, model, X_test):
        self.test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # csvファイルで保存
    def _submit(self, test_data):
        pred_df = pd.DataFrame(self.test_pred.round(), columns=['income'])
        ans = pd.DataFrame(test_data['id'], columns=['id'])
        ans = ans.merge(pred_df, right_index=True, left_index=True)
        ans['income'] = ans['income'].replace({0: '<=50K', 1: '>50K'})

        ans.to_csv(self.ans_file, header=False, index=False)

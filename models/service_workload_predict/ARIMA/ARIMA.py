import numpy as np

from statsmodels.tsa.arima.model import ARIMA


class ARIMAModle:
    @staticmethod
    def train(train_data, p=2, d=1, q=2):
        # 设置ARIMA模型参数
        # AR阶数
        # 差分阶数
        # MA阶数

        predict = []

        for item in train_data:
            model = ARIMA(item, order=(p, d, q))
            model_fit = model.fit()
            res = model_fit.forecast(steps=1)
            predict.append(res)

        return np.array(predict)

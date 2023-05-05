import sys
import torch

from itertools import product
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

sys.path.append('../')

from ARIMA.ARIMA import ARIMA
from N_Beats.N_Beats import NBeatsModel
from Bi_LSTM_Attention.BI_LSTM_Attention import BiLSTMAtteionModel
from Bi_LSTM.Bi_LSTM import BiLSTMModel
from LSTM.LSTM import LSTMModel
from train import *

train_workload, val_workload, test_workload = load_service_workload()


class PyTorchGridSearchCV(BaseEstimator, RegressorMixin):

    def __init__(self, model_name, input_dim=4, output_dim=4, step_num=5, hidden_dim=32, num_layers=2, epochs=10, batch_size=32, device=None):
        self.model_name = model_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.step_num = step_num
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def fit(self):
        if self.model_name == "LSTM":
            self.model = LSTMModel(self.input_dim, self.hidden_dim,
                                   self.num_layers, self.output_dim).to(self.device)
        elif self.model_name == "Bi-LSTM":
            self.model = BiLSTMModel(self.input_dim, self.hidden_dim,
                                     self.num_layers, self.output_dim).to(self.device)
        elif self.model_name == "Bi-LSTM-Attention":
            self.model = BiLSTMAtteionModel(
                self.input_dim, self.step_num, self.output_dim, self.hidden_dim, self.num_layers).to(self.device)
        elif self.model_name == "N-Beats":
            self.model = NBeatsModel(self.step_num, self.output_dim,
                                     self.hidden_dim).to(self.device)

        # 按照 msinstanceid 列分组
        train_grouped_df = train_workload.groupby('msinstanceid')
        val_grouped_df = val_workload.groupby('msinstanceid')

        train_http_rt_df = train_grouped_df.apply(
            lambda x: split_array_by_step(x, self.step_num + 1))

        val_http_rt_df = val_grouped_df.apply(
            lambda x: split_array_by_step(x, self.step_num + 1))

        train_http_rt_df = np.concatenate(train_http_rt_df, axis=0)
        val_http_rt_df = np.concatenate(val_http_rt_df, axis=0)

        x_train, y_train = create_sequences(train_http_rt_df, self.step_num)
        x_val, y_val = create_sequences(val_http_rt_df, self.step_num)

        train(x_train, y_train, x_val, y_val,
              self.model, self.epochs, self.batch_size, self.device)
        return self

    def predict(self):
        test_grouped_df = test_workload.groupby('msinstanceid')
        test_http_rt_df = test_grouped_df.apply(
            lambda x: split_array_by_step(x, self.step_num + 1))
        test_http_rt_df = np.concatenate(test_http_rt_df, axis=0)

        x_test, self.y_test = create_sequences(test_http_rt_df, self.step_num)

        x_pred = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x_pred)
        return y_pred.cpu().numpy()

    def set_params(self, **params):
        self.__dict__.update(params)
        return self


def custom_grid_search(model, param_grid):
    param_combinations = list(product(*param_grid.values()))
    best_score = float('inf')
    best_params = None
    best_model = None

    for param_values in param_combinations:
        params = dict(zip(param_grid.keys(), param_values))
        print(f"Training with parameters: {params}")

        # 创建模型实例并设置参数
        model_instance = model(**params)

        # 这里，您可以将train_data传递给fit方法
        model_instance.fit()

        # 计算验证集上的评分（例如，MSE）
        y_pred = model_instance.predict()
        score = mean_squared_error(model_instance.y_test, y_pred)

        if score < best_score:
            best_score = score
            best_params = params
            best_model = model_instance

    return best_model, best_score, best_params


if __name__ == "__main__":
    param_grid = {
        'model_name': ['LSTM'],
        'step_num': [2, 3, 5, 7],
        'num_layers': [1, 2, 4],
        'hidden_dim': [16, 32, 64],
        'epochs': [10, 30, 50, 80, 100],
        'batch_size': [16, 32, 64],
    }

    # 实例化PyTorchGridSearchCV对象
    input_dim = 4
    output_dim = 4

    # 执行自定义网格搜索
    best_model, best_score, best_params = custom_grid_search(
        PyTorchGridSearchCV, param_grid)

    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")

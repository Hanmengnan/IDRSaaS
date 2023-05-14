import sys
import torch

from itertools import product
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

sys.path.append('../../')

from Bi_LSTM_Attention.BI_LSTM_Attention import BiLSTMAttentionModel
from utils.train import *

train_workload, val_workload, test_workload = load_service_workload()


class PyTorchGridSearchCV(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=4, output_dim=4, step_num=5, hidden_dim=64, num_layers=1, epochs=80, batch_size=16, device=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.step_num = step_num
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self):
        self.model = BiLSTMAttentionModel(self.input_dim, self.output_dim, self.hidden_dim, self.num_layers).to(self.device)

        # 按照 msinstanceid 列分组
        train_grouped_df = train_workload.groupby('msinstanceid')
        val_grouped_df = val_workload.groupby('msinstanceid')

        train_http_rt_df = train_grouped_df.apply(lambda x: split_array_by_step(x, self.step_num + 1))
        val_http_rt_df = val_grouped_df.apply(lambda x: split_array_by_step(x, self.step_num + 1))

        train_http_rt_df = np.concatenate(train_http_rt_df, axis=0)
        val_http_rt_df = np.concatenate(val_http_rt_df, axis=0)

        x_train, y_train = create_sequences(train_http_rt_df, self.step_num)
        x_val, y_val = create_sequences(val_http_rt_df, self.step_num)

        attention_x_train = np.transpose(x_train, (0, 2, 1))
        attention_x_val = np.transpose(x_val, (0, 2, 1))

        train(attention_x_train, y_train, attention_x_val, y_val, self.model, self.epochs, self.batch_size, self.device)

        return self

    def predict(self):
        test_grouped_df = test_workload.groupby('msinstanceid')

        test_http_rt_df = test_grouped_df.apply(lambda x: split_array_by_step(x, self.step_num + 1))

        test_http_rt_df = np.concatenate(test_http_rt_df, axis=0)

        x_test, self.y_test = create_sequences(test_http_rt_df, self.step_num)

        attention_x_test = np.transpose(x_test, (0, 2, 1))

        predict_x = torch.tensor(attention_x_test, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(predict_x)

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

        print(f"Score with parameters: {score}")

        if score < best_score:
            best_score = score
            best_params = params
            best_model = model_instance

    return best_model, best_score, best_params


if __name__ == "__main__":
    param_grid = {
        'step_num': [2, 3, 5, 8, 10],
    }

    # 实例化PyTorchGridSearchCV对象
    input_dim = 4
    output_dim = 4

    # 执行自定义网格搜索
    best_model, best_score, best_params = custom_grid_search(PyTorchGridSearchCV, param_grid)

    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")

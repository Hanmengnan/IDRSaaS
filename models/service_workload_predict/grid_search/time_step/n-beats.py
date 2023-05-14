import sys
import torch

from itertools import product
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

sys.path.append('../../')

from N_Beats.N_Beats import NBeatsModel
from utils.train import *

train_workload, val_workload, test_workload = load_service_workload()


class PyTorchGridSearchCV(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=4, output_dim=1, step_num=5, hidden_dim=32, nb_blocks_per_stack=4, thetas_dim=4, epochs=80, batch_size=16, device=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.step_num = step_num
        self.hidden_dim = hidden_dim
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self):
        self.model = NBeatsModel.model(self.step_num, self.output_dim, self.nb_blocks_per_stack, self.thetas_dim, self.hidden_dim).to(self.device)

        # 按照 msinstanceid 列分组
        train_grouped_df = train_workload.groupby('msinstanceid')
        val_grouped_df = val_workload.groupby('msinstanceid')

        train_http_rt_df = train_grouped_df.apply(lambda x: split_array_by_step(x, self.step_num + 1))
        val_http_rt_df = val_grouped_df.apply(lambda x: split_array_by_step(x, self.step_num + 1))

        train_http_rt_df = np.concatenate(train_http_rt_df, axis=0)
        val_http_rt_df = np.concatenate(val_http_rt_df, axis=0)

        x_train, y_train = create_sequences(train_http_rt_df, self.step_num)
        x_val, y_val = create_sequences(val_http_rt_df, self.step_num)

        NBeatsModel.train(x_train[:, :, :1], y_train[:, :1], x_val[:, :, :1], y_val[:, :1], self.model, self.epochs, self.batch_size, self.device)

        return self

    def predict(self):
        test_grouped_df = test_workload.groupby('msinstanceid')
        test_http_rt_df = test_grouped_df.apply(lambda x: split_array_by_step(x, self.step_num + 1))
        test_http_rt_df = np.concatenate(test_http_rt_df, axis=0)

        x_test, self.y_test = create_sequences(test_http_rt_df, self.step_num)

        return NBeatsModel.predict(self.model, x_test[:, :, 0], self.device)

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
        score = mean_squared_error(model_instance.y_test[:, 0], y_pred)

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
    input_dim = 1
    output_dim = 1

    # 执行自定义网格搜索
    best_model, best_score, best_params = custom_grid_search(PyTorchGridSearchCV, param_grid)

    print(f"Best score: {best_score}")
    print(f"Best parameters: {best_params}")

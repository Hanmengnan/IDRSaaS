import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def predict(model, predict_x, device):
    predict_x = torch.tensor(predict_x, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(predict_x)
    return y_pred.cpu().numpy()


def MSE_Loss(pred_tensors, actual_tensor):
    criterion = nn.MSELoss()
    for item in pred_tensors:
        val_loss = criterion(item["tensor"], actual_tensor)
        print("{model_name} MSE Validation Loss: {loss:.6f}".format(model_name=item["model_name"], loss=val_loss.item()))


def MAPE_Loss(pred_tensors, actual_tensor):
    criterion = nn.L1Loss()
    for item in pred_tensors:
        val_loss = criterion(item["tensor"], actual_tensor)
        print("{model_name} MAPE Validation Loss: {loss:.6f}".format(model_name=item["model_name"], loss=val_loss.item()))


def MAE_Loss(pred_tensors, actual_tensor):
    for item in pred_tensors:
        pct_error = 100.0 * torch.abs((item["tensor"] - actual_tensor) / actual_tensor)
        isinf = torch.isinf(pct_error)
        pct_error = torch.masked_select(pct_error, ~isinf)
        print("{model_name} MAE Validation Loss: {loss:.6f}".format(model_name=item["model_name"], loss=torch.mean(pct_error)))


def RMSE_Loss(pred_tensors, actual_tensor):
    for item in pred_tensors:
        differences = item["tensor"] - actual_tensor
        squared_differences = torch.square(differences)
        mean_squared_error = torch.mean(squared_differences)
        root_mean_squared_error = torch.sqrt(mean_squared_error)

        print("{model_name} RMSE Validation Loss: {loss:.6f}".format(model_name=item["model_name"], loss=root_mean_squared_error))


def plot_predictions(pred_tensors, actual):
    color_palette = plt.get_cmap("tab10").colors

    plt.figure(figsize=(12, 6))
    plt.xlabel("Timestamp")
    plt.ylabel("Workload")
    plt.plot(actual, label="Actual Values", color='blue', linewidth=2)

    line_styles = ['--', '--', '--', '--', '--']

    for index, item in enumerate(pred_tensors):
        plt.plot(item["tensor"], label="{model_name} Predicted Values".format(
            model_name=item["model_name"]), color=color_palette[index], linestyle=line_styles[index], linewidth=1.5)

    plt.legend()
    plt.show()
    plt.savefig('./data/img/result.jpg', dpi=1200)


def save_result(pred_tensors, actual):
    df = actual.numpy()
    # 将向量与字典中的张量按列拼接
    for entry in pred_tensors:
        tensor = entry["tensor"]
        numpy_array = tensor.numpy()
        df = np.hstack((df, numpy_array))

    # 准备列标题
    column_names = ['Original'] + [entry["model_name"] for entry in pred_tensors]

    # 将 NumPy 数组保存到 CSV 文件中
    df = pd.DataFrame(df, columns=column_names)
    df.to_csv("./data/output.csv", index=False)

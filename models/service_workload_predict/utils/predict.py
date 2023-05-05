import torch
import torch.nn as nn

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
        print("{model_name} MSE Validation Loss: {loss:.6f}".format(
            model_name=item["model_name"], loss=val_loss.item()))


def MAPE_Loss(pred_tensors, actual_tensor):
    criterion = nn.L1Loss()
    for item in pred_tensors:
        val_loss = criterion(item["tensor"], actual_tensor)
        print("{model_name} MSE Validation Loss: {loss:.6f}".format(
            model_name=item["model_name"], loss=val_loss.item()))


def MAE_Loss(pred_tensors, actual_tensor):
    for item in pred_tensors:
        pct_error = 100.0 * \
            torch.abs((item["tensor"] - actual_tensor) /
                      actual_tensor)
        isinf = torch.isinf(pct_error)
        pct_error = torch.masked_select(pct_error, ~isinf)
        print(
            "{model_name} MSE Validation Loss: {loss:.6f}".format(model_name=item["model_name"], loss=torch.mean(pct_error)))


def plot_predictions(pred_tensors, actual):
    plt.figure(figsize=(12, 6))
    plt.xlabel("Timestamp")
    plt.ylabel("Workload")
    plt.plot(actual, label="Actual Values", color='blue')

    for item in pred_tensors:
        plt.plot(item["tensor"], label="{model_name} Predicted Values".format(
            model_name=item["model_name"]), color=item["color"])

    plt.legend()
    plt.show()
    plt.savefig('./data/img/result.jpg', dpi=800)


def save_result(y_test, arima_pred, n_beats_pred,
                lstm_pred, bi_lstm_pred, attention_pred):

    y_test = y_test.squeeze()
    arima_pred = arima_pred.squeeze()
    n_beats_pred = n_beats_pred.squeeze()
    lstm_pred = lstm_pred.squeeze()
    bi_lstm_pred = lstm_pred.squeeze()
    attention_pred = lstm_pred.squeeze()

    # 将这些数组组合成一个DataFrame对象
    df = pd.concat([pd.Series(y_test),
                    pd.Series(arima_pred),
                    pd.Series(n_beats_pred),
                    pd.Series(lstm_pred),
                    pd.Series(bi_lstm_pred),
                    pd.Series(attention_pred),
                    ], axis=1)

    # 将DataFrame对象写入CSV文件
    df.to_csv('./my_data.csv', index=False)

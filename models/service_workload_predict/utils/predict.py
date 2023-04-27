import torch
import torch.nn as nn

import matplotlib.pyplot as plt


def predict(model, predict_x, device):
    predict_x = torch.tensor(predict_x, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(predict_x)
    return y_pred.cpu().numpy()


def MSE_Loss(arima_pre_tensor, n_beats_pred_tensor, lstm_pred_tensor, bi_lstm_pred_tensor, attention_pred_tensor, y_test_tensor):
    criterion = nn.MSELoss()

    val_loss = criterion(arima_pre_tensor, y_test_tensor)
    print("ARIMA MSE Validation Loss: {:.6f}".format(val_loss.item()))

    val_loss = criterion(n_beats_pred_tensor, y_test_tensor)
    print("N-BEATS MSE Validation Loss: {:.6f}".format(val_loss.item()))

    val_loss = criterion(lstm_pred_tensor, y_test_tensor)
    print("LSTM MSE Validation Loss: {:.6f}".format(val_loss.item()))

    val_loss = criterion(bi_lstm_pred_tensor, y_test_tensor)
    print("BI-LSTM MSE Validation Loss: {:.6f}".format(val_loss.item()))

    val_loss = criterion(attention_pred_tensor, y_test_tensor)
    print(
        "BI-LSTM-Attention MSE Validation Loss: {:.6f}".format(val_loss.item()))


def MAPE_Loss(arima_pre_tensor, n_beats_pred_tensor, lstm_pred_tensor, bi_lstm_pred_tensor, attention_pred_tensor, y_test_tensor):
    criterion = nn.L1Loss()

    val_loss = criterion(arima_pre_tensor, y_test_tensor)
    print("ARIMA MAPE Validation Loss: {:.6f}".format(val_loss.item()))

    val_loss = criterion(n_beats_pred_tensor, y_test_tensor)
    print("N-BEATS MAPE Validation Loss: {:.6f}".format(val_loss.item()))

    val_loss = criterion(lstm_pred_tensor, y_test_tensor)
    print("LSTM MAPE Validation Loss: {:.6f}".format(val_loss.item()))

    val_loss = criterion(bi_lstm_pred_tensor, y_test_tensor)
    print("BI-LSTM MAPE Validation Loss: {:.6f}".format(val_loss.item()))

    val_loss = criterion(attention_pred_tensor, y_test_tensor)
    print(
        "BI-LSTM-Attention MAPE Validation Loss: {:.6f}".format(val_loss.item()))


def MAE_Loss(arima_pre_tensor, n_beats_pred_tensor, lstm_pred_tensor, bi_lstm_pred_tensor, attention_pred_tensor, y_test_tensor):
    pct_error = 100.0 * \
        torch.abs((arima_pre_tensor - y_test_tensor) /
                  y_test_tensor)
    isinf = torch.isinf(pct_error)
    pct_error = torch.masked_select(pct_error, ~isinf)
    print("ARIMA MAE Validation Loss: {:.6f}".format(torch.mean(pct_error)))

    pct_error = 100.0 * \
        torch.abs((n_beats_pred_tensor - y_test_tensor) /
                  y_test_tensor)
    isinf = torch.isinf(pct_error)
    pct_error = torch.masked_select(pct_error, ~isinf)
    print("N-BEATS MAE Validation Loss: {:.6f}".format(torch.mean(pct_error)))

    pct_error = 100.0 * \
        torch.abs((lstm_pred_tensor - y_test_tensor) /
                  y_test_tensor)
    isinf = torch.isinf(pct_error)
    pct_error = torch.masked_select(pct_error, ~isinf)
    print("LSTM MAE Validation Loss: {:.6f}".format(torch.mean(pct_error)))

    pct_error = 100.0 * \
        torch.abs((bi_lstm_pred_tensor -
                   y_test_tensor) / y_test_tensor)
    isinf = torch.isinf(pct_error)
    pct_error = torch.masked_select(pct_error, ~isinf)
    print("BI-LSTM MAE Validation Loss: {:.6f}".format(torch.mean(pct_error)))

    pct_error = 100.0 * \
        torch.abs((attention_pred_tensor -
                   y_test_tensor) / y_test_tensor)
    isinf = torch.isinf(pct_error)
    pct_error = torch.masked_select(pct_error, ~isinf)
    print(
        "BI-LSTM-Attention MAE Validation Loss: {:.6f}".format(torch.mean(pct_error)))


def plot_predictions(y_val, y_pre_arima,  y_pre_n_beat, y_pred_lstm, y_pred_bi_lstm, y_pred_attention):
    plt.figure(figsize=(12, 6))
    plt.ylim(0, 0.1)
    plt.plot(y_val, label="Actual Values", color='blue')
    plt.plot(y_pre_arima, label="ARIMA Predicted Values", color='pink')
    plt.plot(y_pre_n_beat, label="N-Beats Predicted Values", color='aqua')
    plt.plot(y_pred_lstm, label="LSTM Predicted Values", color='red')
    plt.plot(y_pred_bi_lstm, label="BI-LSTM Predicted Values", color='orange')
    plt.plot(y_pred_attention,
             label="BI-LSTM-Attention Predicted Values", color='green')

    plt.xlabel("Timestamp")
    plt.ylabel("Workload")
    plt.legend()
    plt.show()

    plt.savefig('./data/img/result.jpg', dpi=800)

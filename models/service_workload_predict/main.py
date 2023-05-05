import torch
import numpy as np

from ARIMA import ARIMAModle
from LSTM import LSTMModel
from Bi_LSTM import BiLSTMModel
from Bi_LSTM_Attention import BiLSTMAtteionModel
from N_Beats import NBeatsModel

from utils.train import *
from utils.predict import *

if __name__ == "__main__":
    print("service workload model train")

    ################################################################################
    # Data preparation stage
    ################################################################################
    # 定义步长 K
    TIME_STEP = 5

    input_dim = 1
    output_dim = 1
    batch_size = 10  # 每轮训练模型时，样本的数量
    epochs = 20       # 训练轮次
    hidden_size = 32
    layer_num = 2

    train_workload, val_workload, test_workload = load_service_workload()

    train_workload = split_array_by_step(train_workload, TIME_STEP)
    val_workload = split_array_by_step(val_workload, TIME_STEP)
    test_workload = split_array_by_step(test_workload, TIME_STEP)

    x_train, y_train = create_sequences(train_workload, TIME_STEP)
    x_val, y_val = create_sequences(val_workload, TIME_STEP)
    x_test, y_test = create_sequences(test_workload, TIME_STEP)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ################################################################################
    # ARIMA
    ################################################################################
    arima_pred = ARIMAModle.train(x_test)

    ################################################################################
    # N-Beats
    ################################################################################
    n_beats_model = NBeatsModel.model(
        input_dim=TIME_STEP, output_dim=output_dim, hidden_size=hidden_size)
    n_beats_model_trained = NBeatsModel.train_model(
        x_train, y_train, x_val, y_val, n_beats_model, epochs, batch_size, device)
    n_beats_pred = NBeatsModel.predict(n_beats_model_trained, x_test, device)
    # 保存模型
    save_model(n_beats_model, "./data/model/N-Beats.pt")

    ################################################################################
    # LSTM Model
    #################################################################################
    lstm_model = LSTMModel(input_dim, hidden_size,
                           layer_num, output_dim).to(device)
    lstm_model_trained = train(
        x_train, y_train, x_val, y_val, lstm_model, epochs, batch_size, device)
    lstm_pred = predict(lstm_model_trained, x_test, device)
    # 保存模型
    save_model(lstm_model, "./data/model/LSTM.pt")

    ################################################################################
    # BI-LSTM Model
    ################################################################################
    bi_lstm_model = BiLSTMModel(
        input_dim, hidden_size, layer_num, output_dim).to(device)
    bi_lstm_model_trained = train(
        x_train, y_train, x_val, y_val, bi_lstm_model, epochs, batch_size, device)
    bi_lstm_pred = predict(bi_lstm_model_trained, x_test, device)
    # 保存模型
    save_model(bi_lstm_model, "./data/model/BI-LSTM.pt")

    ################################################################################
    # BI-LSTM Attention Model
    ################################################################################
    attention_model = BiLSTMAtteionModel(
        input_dim, TIME_STEP, output_dim).to(device)
    # Reshape the data to match the expected input format of nn.Conv1d
    attention_x_trainx_train = np.transpose(x_train, (0, 2, 1))
    attention_x_val = np.transpose(x_val, (0, 2, 1))
    attention_x_test = np.transpose(x_test, (0, 2, 1))
    attention_model_trained = train(attention_x_trainx_train, y_train,
                                    attention_x_val, y_val, attention_model, epochs, batch_size, device)
    attention_pred = predict(attention_model_trained, attention_x_test, device)
    # 保存模型
    save_model(attention_model, "./data/model/BI-LSTM-Attention.pt")

    ################################################################################
    # Loss Compute
    ################################################################################
    pred_tensors = [
        {"model_name": "ARIMA", "tensor": torch.tensor(
            arima_pred), "color": "pink"},
        {"model_name": "LSTM", "tensor": torch.tensor(
            lstm_pred), "color": "green"},
        {"model_name": "Bi-LSTM",
            "tensor": torch.tensor(bi_lstm_pred), "color": "orange"},
        {"model_name": "Bi-LSTM-Attention",
            "tensor": torch.tensor(attention_pred), "color": "brown"},
        {"model_name": "N-Beats",
            "tensor": torch.tensor(n_beats_pred), "color": "red"},
    ]
    actual_tensor = torch.tensor(y_test)

    # MSE
    MSE_Loss(pred_tensors, actual_tensor)
    # MAE
    MAE_Loss(pred_tensors, actual_tensor)
    # MAPE
    MAPE_Loss(pred_tensors, actual_tensor)

    ################################################################################
    # Save Data
    ################################################################################
    plot_predictions(pred_tensors, actual_tensor)
    save_result(pred_tensors, actual_tensor)

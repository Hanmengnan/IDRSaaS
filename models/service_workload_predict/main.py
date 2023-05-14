import torch
import numpy as np

from ARIMA.ARIMA import ARIMAModle
from LSTM.LSTM import LSTMModel
from Bi_LSTM.Bi_LSTM import BiLSTMModel
from Bi_LSTM_Attention.BI_LSTM_Attention import BiLSTMAttentionModel
from N_Beats.N_Beats import NBeatsModel

from utils.train import *
from utils.predict import *

if __name__ == "__main__":
    print("service workload model train")

    ################################################################################
    # Data preparation stage
    ################################################################################
    # 定义步长 K
    TIME_STEP = 5

    train_df, val_df, test_df = load_service_workload()

    train_grouped_df = train_df.groupby('msinstanceid')
    val_grouped_df = val_df.groupby('msinstanceid')
    test_grouped_df = test_df.groupby('msinstanceid')

    train_http_rt_df = train_grouped_df.apply(lambda x: split_array_by_step(x, TIME_STEP + 1))
    val_http_rt_df = val_grouped_df.apply(lambda x: split_array_by_step(x, TIME_STEP + 1))
    test_http_rt_df = test_grouped_df.apply(lambda x: split_array_by_step(x, TIME_STEP + 1))

    train_workload = np.concatenate(train_http_rt_df, axis=0)
    val_workload = np.concatenate(val_http_rt_df, axis=0)
    test_workload = np.concatenate(test_http_rt_df, axis=0)

    x_train, y_train = create_sequences(train_workload, TIME_STEP)
    x_val, y_val = create_sequences(val_workload, TIME_STEP)
    x_test, y_test = create_sequences(test_workload, TIME_STEP)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ################################################################################
    # ARIMA
    ################################################################################
    arima_pred = ARIMAModle.train(x_test[:, :, :1])

    ################################################################################
    # N-Beats
    ################################################################################
    n_beats_model = NBeatsModel.model(input_dim=TIME_STEP, output_dim=1, nb_blocks_per_stack=4, thetas_dim=4, hidden_size=32)
    n_beats_model_trained, _, _ = NBeatsModel.train(x_train[:, :, :1], y_train[:, :1], x_val[:, :, :1], y_val[:, :1], n_beats_model, 30, batch_size=16, device=device)
    n_beats_pred = NBeatsModel.predict(n_beats_model_trained, x_test[:, :, :1], device)
    # 保存模型
    save_model(n_beats_model, "./data/model/N-Beats.pt")

    ################################################################################
    # LSTM Model
    #################################################################################
    lstm_model = LSTMModel(input_dim=4, output_dim=4, hidden_dim=64, num_layers=1).to(device)
    lstm_model_trained, _, _ = train(x_train, y_train, x_val, y_val, lstm_model, 30, batch_size=32, device=device)
    lstm_pred = predict(lstm_model_trained, x_test, device)
    # 保存模型
    save_model(lstm_model, "./data/model/LSTM.pt")

    ################################################################################
    # BI-LSTM Model
    ################################################################################
    bi_lstm_model = BiLSTMModel(input_dim=4, output_dim=4, hidden_dim=16, num_layers=2).to(device)
    bi_lstm_model_trained, _, _ = train(x_train, y_train, x_val, y_val, bi_lstm_model, 30, batch_size=16, device=device)
    bi_lstm_pred = predict(bi_lstm_model_trained, x_test, device)
    # 保存模型
    save_model(bi_lstm_model, "./data/model/BI-LSTM.pt")

    ################################################################################
    # BI-LSTM Attention Model
    ################################################################################
    # Reshape the data to match the expected input format of nn.Conv1d
    attention_x_train = np.transpose(x_train, (0, 2, 1))
    attention_x_val = np.transpose(x_val, (0, 2, 1))
    attention_x_test = np.transpose(x_test, (0, 2, 1))

    attention_model = BiLSTMAttentionModel(input_dim=4, output_dim=4, hidden_dim=64, num_layers=1).to(device)
    attention_model_trained, _, _ = train(attention_x_train, y_train, attention_x_val, y_val, attention_model, 30, batch_size=16, device=device)
    attention_pred = predict(attention_model_trained, attention_x_test, device)
    # 保存模型
    save_model(attention_model, "./data/model/BI-LSTM-Attention.pt")

    ################################################################################
    # Loss Compute
    ################################################################################
    pred_tensors = [
        {"model_name": "ARIMA", "tensor": torch.tensor(arima_pred), "color": "pink"},
        {"model_name": "LSTM", "tensor": torch.tensor(lstm_pred[:, :1]), "color": "green"},
        {"model_name": "Bi-LSTM", "tensor": torch.tensor(bi_lstm_pred[:, :1]), "color": "orange"},
        {"model_name": "Bi-LSTM-Attention", "tensor": torch.tensor(attention_pred[:, :1]), "color": "brown"},
        {"model_name": "N-Beats", "tensor": torch.tensor(n_beats_pred), "color": "red"},
    ]
    actual_tensor = torch.tensor(y_test[:, :1])

    # MSE
    MSE_Loss(pred_tensors, actual_tensor)
    # MAE
    MAE_Loss(pred_tensors, actual_tensor)
    # MAPE
    MAPE_Loss(pred_tensors, actual_tensor)
    # RMSE
    RMSE_Loss(pred_tensors, actual_tensor)

    ################################################################################
    # Save Data
    ################################################################################
    plot_predictions(pred_tensors, actual_tensor)
    
    try:
        save_result(pred_tensors, actual_tensor)
    except Exception as e:
        print(e)

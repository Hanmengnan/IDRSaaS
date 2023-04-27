from ARIMA import ARIMAModle
from LSTM import LSTMModel
from Bi_LSTM import BiLSTMModel
from Bi_LSTM_Attention import LSTM_fun_PyTorch_att
from N_Beats import NBeatsModel

from utils.predict import *
from utils.train import *

if __name__ == "__main__":
    print("service workload model train")

    # 定义步长 K
    TIME_STEP = 5
    input_dim = 1
    output_dim = 1
    batch_size = 10  # 每轮训练模型时，样本的数量
    epochs = 20       # 训练轮次
    hidden_size = 32
    layer_num = 2

    workload = loadData(TIME_STEP, chunkNum=10)

    # 将数据转换为LSTM模型的输入形式
    x_train, y_train = create_sequences(workload, TIME_STEP)

    x_train = np.expand_dims(x_train, axis=2)
    y_train = np.expand_dims(y_train, axis=1)

    train_size = int(len(x_train) * 0.6)
    val_size = int(len(x_train) * 0.2)
    test_size = len(x_train) - train_size-val_size

    x_train, x_val, x_test = x_train[:train_size, :, :], x_train[train_size:train_size +
                                                                 val_size, :, :], x_train[train_size+val_size:, :, :]
    y_train, y_val, y_test = y_train[:train_size], y_train[train_size:train_size +
                                                           val_size], y_train[train_size+val_size:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ################################################################################
    # ARIMA
    ################################################################################
    arima_pred = ARIMAModle.train(x_test)

    ################################################################################
    # N-Beats
    ################################################################################
    n_beats_model = NBeatsModel.model(input_dim=TIME_STEP,
                                      output_dim=output_dim, hidden_size=hidden_size)
    n_beats_model_trained = NBeatsModel.train_model(x_train, y_train, x_val, y_val, n_beats_model,
                                                    epochs, batch_size, device)
    n_beats_pred = NBeatsModel.predict(n_beats_model_trained, x_test, device)
    # 保存模型
    save_model(n_beats_model, "./data/model/N-Beats.pt")

    ################################################################################
    # LSTM Model
    #################################################################################
    lstm_model = LSTMModel(input_dim, hidden_size,
                           layer_num, output_dim).to(device)
    lstm_model_trained = train_model(x_train, y_train, x_val, y_val, lstm_model,
                                     epochs, batch_size, device)
    lstm_pred = predict(lstm_model_trained, x_test, device)
    # 保存模型
    save_model(lstm_model, "./data/model/LSTM.pt")

    ################################################################################
    # BI-LSTM Model
    ################################################################################
    bi_lstm_model = BiLSTMModel(
        input_dim, hidden_size, layer_num, output_dim).to(device)
    bi_lstm_model_trained = train_model(x_train, y_train, x_val, y_val, bi_lstm_model,
                                        epochs, batch_size, device)
    bi_lstm_pred = predict(bi_lstm_model_trained, x_test, device)
    # 保存模型
    save_model(bi_lstm_model, "./data/model/BI-LSTM.pt")

    ################################################################################
    # BI-LSTM Attention Model
    ################################################################################
    attention_model = LSTM_fun_PyTorch_att(
        input_dim, TIME_STEP, output_dim).to(device)
    # Reshape the data to match the expected input format of nn.Conv1d
    attention_x_trainx_train = np.transpose(x_train, (0, 2, 1))
    attention_x_val = np.transpose(x_val, (0, 2, 1))
    attention_x_test = np.transpose(x_test, (0, 2, 1))
    attention_model_trained = train_model(attention_x_trainx_train, y_train, attention_x_val, y_val, attention_model,
                                          epochs, batch_size, device)
    attention_pred = predict(attention_model_trained, attention_x_test, device)
    # 保存模型
    save_model(attention_model, "./data/model/BI-LSTM-Attention.pt")

    ################################################################################
    # Loss Compute
    ################################################################################
    arima_pred_tensor = torch.tensor(arima_pred)
    n_beats_pred_tensor = torch.tensor(n_beats_pred)
    lstm_pred_tensor = torch.tensor(lstm_pred)
    bi_lstm_pred_tensor = torch.tensor(bi_lstm_pred)
    attention_pred_tensor = torch.tensor(attention_pred)
    y_test_tensor = torch.tensor(y_test)

    # MSE
    MSE_Loss(arima_pred_tensor, n_beats_pred_tensor, lstm_pred_tensor, bi_lstm_pred_tensor,
             attention_pred_tensor, y_test_tensor)
    # MAE
    MAE_Loss(arima_pred_tensor, n_beats_pred_tensor, lstm_pred_tensor, bi_lstm_pred_tensor,
             attention_pred_tensor, y_test_tensor)
    # MAPE
    MAPE_Loss(arima_pred_tensor, n_beats_pred_tensor, lstm_pred_tensor, bi_lstm_pred_tensor,
              attention_pred_tensor, y_test_tensor)

    n_beats_pred = np.array([])
    lstm_pred = np.array([])
    attention_pred = np.array([])

    plot_predictions(y_test, arima_pred, n_beats_pred, lstm_pred,
                     bi_lstm_pred, attention_pred)

    ################################################################################
    # Save Data
    ################################################################################

    save_result(y_test, arima_pred, n_beats_pred,
                lstm_pred, bi_lstm_pred, attention_pred)

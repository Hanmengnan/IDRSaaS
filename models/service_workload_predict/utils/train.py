import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0, verbose=False, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            path (str): Path to save checkpoint.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.trace_func = trace_func
        self.best_loss = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model)
            if self.verbose:
                self.trace_func(
                    f"Validation loss decreased {self.delta:.6f} units or more, saving model.")

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f"Validation loss decreased. Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.best_loss = val_loss


def validate(model, criterion, dataloader):
    running_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        output = model(batch_x)
        loss = criterion(output, batch_y)

        running_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    epoch_loss = running_loss / total_samples

    return epoch_loss


def train_model(train_x, train_y, val_x, val_y, model, epochs, batch_size, device, patience=3):
    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    dataset = TensorDataset(train_x, train_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_x = torch.tensor(val_x, dtype=torch.float32).to(device)
    val_y = torch.tensor(val_y, dtype=torch.float32).to(device)
    val_dataset = TensorDataset(val_x, val_y)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    scheduler = ReduceLROnPlateau(optimizer, patience=1, verbose=True)
    # early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        running_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

        epoch_loss = running_loss / total_samples
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

        # 在每个epoch后进行验证并检查早停
        with torch.no_grad():
            val_loss = validate(model, criterion, val_dataloader)
            print(f"Validation Loss: {val_loss:.6f}")

            # 更新学习率
            scheduler.step(val_loss)

            # 检查早停
            # early_stopping(val_loss, model)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

    return model


def split_array_by_step(arr, k):
    return [arr[i:i+k] for i in range(0, len(arr)-k)]


def loadData(seq_num, chunk_num):
    chunksize = 10000
    reader = pd.read_csv(
        "/data/hmn_data/alibaba_cluster_data/MSRTQps_sort.csv", chunksize=chunksize)

    df = pd.DataFrame()
    index = 0
    # 循环读取每个数据块并添加到DataFrame中
    for index, chunk in enumerate(reader):
        if index > chunk_num:
            break
        df = pd.concat([df, chunk])

    grouped_df = df.groupby('msinstanceid')["HTTP_RT"].apply(
        lambda x: split_array_by_step(x, seq_num+1)).reset_index()

    combined_df = pd.DataFrame()

    # 对于每个分组
    for index, row in grouped_df.iterrows():
        if len(row["HTTP_RT"]) > 0:
            combined_df = pd.concat([combined_df, pd.DataFrame(
                np.stack([arr for arr in row["HTTP_RT"]], axis=0))])

    scaler = MinMaxScaler()
    workload = scaler.fit_transform(combined_df)

    return workload


def create_sequences(data, time_steps=1):
    xs, ys = [], []
    for i in range(len(data)):
        xs.append(data[i][:time_steps])
        ys.append(data[i][time_steps])
    return np.array(xs), np.array(ys)


def save_model(model, path):
    torch.save(model.state_dict(), path)


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

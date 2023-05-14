import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler

import pywt


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


def train(train_x, train_y, val_x, val_y, model, epochs, batch_size, device):
    loss_array, val_loss_array = [], []

    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).to(device)
    dataset = TensorDataset(train_x, train_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_x = torch.tensor(val_x, dtype=torch.float32).to(device)
    val_y = torch.tensor(val_y, dtype=torch.float32).to(device)
    val_dataset = TensorDataset(val_x, val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True)

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
        loss_array.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

        # 在每个epoch后进行验证并检查早停
        with torch.no_grad():
            val_loss = validate(model, criterion, val_dataloader)
            val_loss_array.append(val_loss)
            print(f"Validation Loss: {val_loss:.6f}")

            # 更新学习率
            scheduler.step(val_loss)

    return model, loss_array, val_loss_array

# 小波滤噪
def wavelet_denoising(data):
    # 小波函数取db8
    db4 = pywt.Wavelet('db8')

    # 分解
    coeffs = pywt.wavedec(data, db4)
    # 高频系数置零
    coeffs[len(coeffs) - 1] *= 0
    coeffs[len(coeffs) - 2] *= 0
    # 重构
    meta = pywt.waverec(coeffs, db4)
    meta = meta[:len(data)]
    return meta


def supplement_feature(df, K):
    df["Polyfit_HTTP_RT"] = df['HTTP_RT'].rolling(window=K).apply(lambda y: np.polyfit(range(K), y, 1)[0] if len(y.dropna()) == K else np.nan)
    df["Mean_HTTP_RT"] = df['HTTP_RT'].rolling(window=K).mean()
    df["Median_HTTP_RT"] = df['HTTP_RT'].rolling(window=K).median()

    df["HTTP_RT"] = wavelet_denoising(df["HTTP_RT"])

    df_no_nan = df.dropna()
    return df_no_nan


def split_array_by_step(df, K):
    if len(df) == 0:
        return []
    # supplement more feature
    more_feature_df = supplement_feature(df, K)
    more_feature_df = more_feature_df[["HTTP_RT", "Polyfit_HTTP_RT", "Mean_HTTP_RT", "Median_HTTP_RT"]]

    if len(more_feature_df) == 0:
        return []

    # normalizate 归一化
    scaler = MinMaxScaler()
    more_feature_df[["HTTP_RT", "Polyfit_HTTP_RT", "Mean_HTTP_RT", "Median_HTTP_RT"]] = scaler.fit_transform(more_feature_df[["HTTP_RT", "Polyfit_HTTP_RT", "Mean_HTTP_RT", "Median_HTTP_RT"]])
    # 遍历 DataFrame 的每一行，跳过前 K 行
    result = []
    for i in range(0, len(more_feature_df) - K, 1):
        window = more_feature_df.iloc[i:i + K]
        result.append(window)

    return result


def load_aimed_service(df, selected_ids):
    # 提取包含所选id的所有行
    selected_rows = []
    for selected_id in selected_ids:
        # 或者使用 df.loc[df['msinstanceid'] == selected_id]
        rows = df.query('msinstanceid == @selected_id')
        selected_rows.append(rows)

    # 将提取的行组合成一个新的DataFrame
    selected_rows_df = pd.concat(selected_rows)
    return selected_rows_df


def load_service_workload():
    chunksize = 10000
    reader = pd.read_csv("/data/hmn_data/alibaba_cluster_data/MSRT_Resource.csv", chunksize=chunksize)

    tmp_df = pd.DataFrame()
    for _, chunk in enumerate(reader):
        tmp_df = pd.concat([tmp_df, chunk])
    # 获取所有唯一的id值
    unique_ids = tmp_df['msinstanceid'].unique()
    # 从唯一的id值中随机选择三个，不放回
    train_ids = np.random.choice(unique_ids, 3, replace=False)
    # 从唯一的id值中随机选择两个，不放回
    val_ids = np.random.choice(unique_ids, 2, replace=False)
    # 从唯一的id值中随机选择一个，不放回
    test_ids = np.random.choice(unique_ids, 1, replace=False)

    train_df = load_aimed_service(tmp_df, train_ids)
    val_df = load_aimed_service(tmp_df, val_ids)
    test_df = load_aimed_service(tmp_df, test_ids)

    return train_df, val_df, test_df


# 将数据转换为LSTM模型的输入形式
def create_sequences(data, time_steps=1):
    xs, ys = [], []
    for i in range(len(data)):
        xs.append(data[i][:time_steps])
        ys.append(data[i][time_steps])
    return np.array(xs), np.array(ys)


def save_model(model, path):
    torch.save(model.state_dict(), path)

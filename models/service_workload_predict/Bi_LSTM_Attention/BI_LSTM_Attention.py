import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, lstm_hidden_size, attention_size):
        super(Attention, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.attention_size = attention_size
        self.attention_vec = nn.Linear(lstm_hidden_size * 2, attention_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, lstm_out):
        attention_probs = self.sigmoid(self.attention_vec(lstm_out))
        return attention_probs * lstm_out


class LSTM_fun_PyTorch_att(nn.Module):
    def __init__(self, input_dim, seq_len, ouput_dim, lstm_hidden_size=64, attention_size=128):
        super(LSTM_fun_PyTorch_att, self).__init__()
        self.conv1d = nn.Conv1d(input_dim, 32, kernel_size=1)
        self.relu = nn.ReLU()
        self.maxpool1d = nn.MaxPool1d(seq_len)
        self.dropout = nn.Dropout(0.1)
        self.bilstm = nn.LSTM(32, lstm_hidden_size, bidirectional=True)
        self.attention = Attention(lstm_hidden_size, attention_size)
        self.fc = nn.Linear(lstm_hidden_size * 2, ouput_dim)

    def forward(self, inputs):
        x = self.conv1d(inputs)
        x = self.relu(x)
        x = self.maxpool1d(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.bilstm(x)
        attention_mul = self.attention(lstm_out)
        output = self.fc(attention_mul)
        return output.squeeze(-1)

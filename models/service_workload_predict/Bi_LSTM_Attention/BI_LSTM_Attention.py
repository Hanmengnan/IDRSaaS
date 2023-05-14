import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(AdditiveAttention, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.attention_vec = nn.Linear(lstm_hidden_size * 2, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_out):
        attention_weights = self.tanh(self.attention_vec(lstm_out))
        attention_weights = self.softmax(attention_weights)
        return torch.sum(attention_weights * lstm_out, dim=1)


class BiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2):
        super(BiLSTMAttentionModel, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.bilstm = nn.LSTM(input_dim, hidden_dim,
                              num_layers=num_layers, bidirectional=True, batch_first=True)
        self.attention = AdditiveAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)  # Transpose inputs to (batch_size, seq_len, input_dim)
        lstm_out, _ = self.bilstm(inputs)
        lstm_out = self.dropout(lstm_out)
        attention_mul = self.attention(lstm_out)
        output = self.fc(attention_mul)
        return output

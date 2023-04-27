import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from nbeats_pytorch.model import NBeatsNet


class NBeatsModel:
    @staticmethod
    def model(input_dim=10, output_dim=1, hidden_size=32):
        model = NBeatsNet(stack_types=('generic', 'generic'),
                          nb_blocks_per_stack=3,
                          thetas_dim=(4, 4),
                          forecast_length=output_dim,
                          backcast_length=input_dim,
                          hidden_layer_units=hidden_size,
                          share_weights_in_stack=True)
        return model

    @staticmethod
    def validate(model, criterion, dataloader):
        running_loss = 0.0
        total_samples = 0

        for batch_x, batch_y in dataloader:
            _, output = model(batch_x)
            loss = criterion(output, batch_y)

            running_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

        epoch_loss = running_loss / total_samples
        return epoch_loss

    @staticmethod
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

        # scheduler = ReduceLROnPlateau(optimizer, patience=1, verbose=True)

        for epoch in range(epochs):
            running_loss = 0.0
            total_samples = 0

            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                _, output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_x.size(0)
                total_samples += batch_x.size(0)

            epoch_loss = running_loss / total_samples
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")

            # 在每个epoch后进行验证并检查早停
            # with torch.no_grad():
            #     val_loss = NBeatsModel.validate(
            #         model, criterion, val_dataloader)
            #     print(f"Validation Loss: {val_loss:.6f}")

            #     # 更新学习率
            #     scheduler.step(val_loss)

        return model

    @staticmethod
    def predict(model, predict_x, device):
        predict_x = torch.tensor(predict_x, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            _, y_pred = model(predict_x)
        return y_pred.cpu().numpy()

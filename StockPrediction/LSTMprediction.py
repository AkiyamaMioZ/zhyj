import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# 加载数据
data = pd.read_csv("tencent_stock_data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 仅保留收盘价用于预测
closing_price = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_price)

# 定义序列长度
sequence_length = 365  # 使用过去100天的数据预测未来价格
train_data = []
for i in range(sequence_length, len(scaled_data)):
    train_data.append(scaled_data[i-sequence_length:i+1])  # 包括未来价格作为标签

train_data = np.array(train_data)
X_train = train_data[:, :-1, :]  # 输入
y_train = train_data[:, -1, :]  # 标签

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建数据加载器
batch_size = 32
train_dataset = StockDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class StockPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(StockPriceLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(1, x.size(0), self.hidden_layer_size).to(x.device)
        c_0 = torch.zeros(1, x.size(0), self.hidden_layer_size).to(x.device)
        _, (hn, _) = self.lstm(x, (h_0, c_0))
        return self.linear(hn[-1])


model = StockPriceLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')



model.eval()
input_seq = scaled_data[-sequence_length:]  # 使用最后 sequence_length 天的数据进行预测
input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)

predicted_prices = []
with torch.no_grad():
    for _ in range(15):  # 预测未来15天
        predicted_price = model(input_seq)
        predicted_prices.append(predicted_price.item())
        # 将预测结果添加到输入序列，并删除最早的值（保持长度为 sequence_length）
        predicted_price = predicted_price.unsqueeze(2)  # 调整维度
        input_seq = torch.cat((input_seq[:, 1:, :], predicted_price), dim=1)

# 将预测结果逆缩放
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
print(f"未来15天预测的股价：{predicted_prices.flatten()}")


import matplotlib.pyplot as plt

# 将原始数据和预测数据合并到一个序列中，以便绘图
# 显示过去 sequence_length 天的真实价格以及未来预测的 N 天价格
past_data = scaler.inverse_transform(scaled_data[-sequence_length:].reshape(-1, 1)).flatten()
predicted_prices = predicted_prices.flatten()

# 创建时间轴标签
date_range = pd.date_range(data.index[-sequence_length], periods=sequence_length + 15, freq='B')

import random

# 添加随机正负扰动到预测结果
perturbed_predicted_prices = []
for price in predicted_prices:
    perturbation = random.uniform(-0.01, 0.01) * price
    perturbed_price = price + perturbation
    perturbed_predicted_prices.append(perturbed_price)

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(date_range[:sequence_length], past_data, label="Real Stock Price", color="blue")
plt.plot(date_range[sequence_length:], perturbed_predicted_prices, label="Perturbed Predicted Price", color="orange")

plt.title("Tencent Stock Price Prediction (With Perturbation)")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.grid()
plt.show()

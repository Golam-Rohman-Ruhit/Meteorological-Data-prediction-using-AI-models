import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


# ==========================================
# 1. Data Preparation (Hybrid 4D Input)
# ==========================================
def load_hybrid_data(file_path, seq_length=14):
    ds = xr.open_dataset(file_path)
    data = ds['t2m'].values
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=np.nanmean(data))

    scaler = MinMaxScaler()
    orig_shape = data.shape
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).reshape(orig_shape)

    X, Y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i: i + seq_length])
        target_avg = np.mean(data_scaled[i + seq_length])
        Y.append(target_avg)
    return np.array(X), np.array(Y), scaler


# ==========================================
# 2. Innovative CNN-LSTM Model Definition
# ==========================================
class HybridCNNLSTM(nn.Module):
    def __init__(self, seq_length):
        super(HybridCNNLSTM, self).__init__()
        # CNN Part: Processes each day's map independently
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # LSTM Part: Processes the sequence of CNN features
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        batch_size, seq_len, h, w = x.size()
        # Collapse batch and seq_len to pass through CNN
        x = x.view(batch_size * seq_len, 1, h, w)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)

        # Pass sequence of features to LSTM
        lstm_out, _ = self.lstm(cnn_out)
        out = self.fc(lstm_out[:, -1, :])
        return out


# ==========================================
# 3. Training and Impact Analysis
# ===============================
file_path = 't2m.nc'
seq_len = 14
batch_size = 16
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_all, Y_all, scaler = load_hybrid_data(file_path, seq_length=seq_len)
train_size = int(len(X_all) * 0.8)
X_train = torch.FloatTensor(X_all[:train_size]).to(device)
Y_train = torch.FloatTensor(Y_all[:train_size]).view(-1, 1).to(device)
X_test = torch.FloatTensor(X_all[train_size:]).to(device)
Y_test = torch.FloatTensor(Y_all[train_size:]).view(-1, 1).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
model = HybridCNNLSTM(seq_length=seq_len).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate for hybrid
criterion = nn.MSELoss()

print("Training Innovative Hybrid CNN-LSTM Model...")
for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test).cpu().numpy()
    actuals = Y_test.cpu().numpy()

preds_rescaled = scaler.inverse_transform(preds)
actuals_rescaled = scaler.inverse_transform(actuals)

# Metrics
r2 = r2_score(actuals_rescaled, preds_rescaled)
print(f"\nHybrid Model R² Score: {r2:.4f}")

# Visualization & Save
plt.figure(figsize=(12, 5))
plt.plot(actuals_rescaled, label='Actual Avg Temp', color='green')
plt.plot(preds_rescaled, label='Hybrid Predicted Temp', color='magenta', linestyle='--')
plt.title(f'Innovation: Hybrid CNN-LSTM Prediction\nR² Score: {r2:.4f}')
plt.legend()
plt.savefig('innovation_hybrid_plot.png')
plt.show()
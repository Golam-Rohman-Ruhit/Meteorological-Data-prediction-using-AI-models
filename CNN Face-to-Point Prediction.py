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
# 1. Data Preparation (Face-to-Point)
# ==========================================
def load_spatial_data(file_path, seq_length=7):
    ds = xr.open_dataset(file_path)
    # Extract t2m: (366, 45, 129) [cite: 302]
    data = ds['t2m'].values

    # Fill NaN values (using mean imputation) [cite: 304-306]
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=np.nanmean(data))

    # --- Debug Fix: Initializing Scaler and data_scaled ---
    scaler = MinMaxScaler()
    orig_shape = data.shape
    # Flatten logic to allow 3D scaling
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).reshape(orig_shape)

    # 2. Construct sequences [cite: 307-315]
    X, Y = [], []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i: i + seq_length])  # Take surface data for consecutive days
        # Calculate the average temperature of the entire region for the next day as the label
        target_avg = np.mean(data_scaled[i + seq_length])
        Y.append(target_avg)

    return np.array(X), np.array(Y), scaler


# ==========================================
# 2. Define CNN Model (Face-to-Point) [cite: 320-346]
# ==========================================
class TemperatureCNN(nn.Module):
    def __init__(self, seq_length):
        super(TemperatureCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=seq_length, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


# ==========================================
# 3. Training and Evaluation
# ==========================================
# Configuration [cite: 350-355]
file_path = 't2m.nc'  # Updated to your file
seq_len = 7
batch_size = 16
epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
X_all, Y_all, scaler = load_spatial_data(file_path, seq_length=seq_len)

# Split into training/test sets [cite: 358-364]
train_size = int(len(X_all) * 0.8)
X_train = torch.FloatTensor(X_all[:train_size]).to(device)
Y_train = torch.FloatTensor(Y_all[:train_size]).view(-1, 1).to(device)
X_test = torch.FloatTensor(X_all[train_size:]).to(device)
Y_test = torch.FloatTensor(Y_all[train_size:]).view(-1, 1).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)

# Initialize model [cite: 366-368]
model = TemperatureCNN(seq_length=seq_len).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop [cite: 369-380]
print("Training CNN...")
model.train()
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

# ==========================================
# 4. Evaluate Results [cite: 382-402]
# ==========================================
model.eval()
with torch.no_grad():
    preds = model(X_test).cpu().numpy()
    actuals = Y_test.cpu().numpy()

# Inverse scaling (restore to Kelvin temperature K)
preds_rescaled = scaler.inverse_transform(preds)
actuals_rescaled = scaler.inverse_transform(actuals)

# Calculate metrics
mae = mean_absolute_error(actuals_rescaled, preds_rescaled)
rmse = np.sqrt(mean_squared_error(actuals_rescaled, preds_rescaled))
r2 = r2_score(actuals_rescaled, preds_rescaled)

print("\n" + "=" * 30)
print("CNN Face-to-Point Prediction Metrics:")
print(f"MAE : {mae:.4f} K")
print(f"RMSE: {rmse:.4f} K")
print(f"R²  : {r2:.4f}")
print("=" * 30)

# ==========================================
# 5. Visualization [cite: 404-412]
# ==========================================
plt.figure(figsize=(12, 5))
plt.plot(actuals_rescaled, label='Actual Avg Temp', color='blue')
plt.plot(preds_rescaled, label='CNN Predicted Avg Temp', color='red', linestyle='--')
plt.title(f'CNN Face-to-Point Prediction (Test Set)\nMAE: {mae:.2f}K, R2: {r2:.2f}')
plt.legend()
plt.grid(True, alpha=0.3)

# Picture save করার কমান্ড
plt.savefig('cnn_prediction_output.png')
print("Graph saved as 'cnn_prediction_output.png'")

plt.show()
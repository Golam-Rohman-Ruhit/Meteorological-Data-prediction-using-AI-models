import xarray as xr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
def load_data(file_path):
    ds = xr.open_dataset(file_path)
    # spatial dimensions থেকে গড় তাপমাত্রা বের করা
    data = ds['t2m'].mean(dim=['latitude', 'longitude']).values
    data = data[~np.isnan(data)].reshape(-1, 1)
    return data

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ==========================================
# 2. LSTM Model Definition
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# ==========================================
# 3. Parameter Configuration
# ==========================================
# তোমার আপলোড করা ফাইলের নাম এখানে বসানো হয়েছে
file_path = 't2m.nc'
seq_length = 14     # গত ১৪ দিনের ডেটা ব্যবহার
train_split = 0.8    # ৮০% ট্রেনিংয়ের জন্য
epochs = 150
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ডেটা তৈরি
raw_data = load_data(file_path)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(raw_data)

X, y = create_sequences(scaled_data, seq_length)
train_size = int(len(X) * train_split)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Tensors-এ রূপান্তর
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)

# ==========================================
# 4. Train the Model
# ==========================================
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 30 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

# ==========================================
# 5. Model Evaluation (Test Set Prediction)
# ==========================================
model.eval()
with torch.no_grad():
    test_preds = model(X_test).cpu().numpy()
    test_preds_rescaled = scaler.inverse_transform(test_preds)
    y_test_rescaled = scaler.inverse_transform(y_test)

# মেট্রিক্স হিসাব
mae = mean_absolute_error(y_test_rescaled, test_preds_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, test_preds_rescaled))
r2 = r2_score(y_test_rescaled, test_preds_rescaled)

print("\n" + "="*30)
print("Model Evaluation Metrics (Test Set):")
print(f"MAE  (Mean Absolute Error): {mae:.4f} K")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f} K")
print(f"R²   (Coefficient of Determination): {r2:.4f}")
print("="*30)

# ==========================================
# 6. Rolling Forecast for Next 30 Days
# ==========================================
forecast_steps = 30
last_window = torch.FloatTensor(scaled_data[-seq_length:]).view(1, seq_length, 1).to(device)
rolling_preds = []

current_window = last_window
with torch.no_grad():
    for _ in range(forecast_steps):
        pred = model(current_window)
        rolling_preds.append(pred.item())
        pred_tensor = pred.view(1, 1, 1)
        current_window = torch.cat((current_window[:, 1:, :], pred_tensor), dim=1)

rolling_preds_rescaled = scaler.inverse_transform(np.array(rolling_preds).reshape(-1, 1))

# ==========================================
# 7. Visualization
# ==========================================
plt.figure(figsize=(14, 7))

# Actual values
plt.plot(range(len(raw_data)), raw_data, label='Actual Temperature', color='black', alpha=0.3)

# Test set prediction
test_start_idx = train_size + seq_length
plt.plot(range(test_start_idx, test_start_idx + len(test_preds_rescaled)),
         test_preds_rescaled, label='Test Set Prediction (One-step)', color='blue')

# Future rolling forecast
plt.plot(range(len(raw_data), len(raw_data) + forecast_steps),
         rolling_preds_rescaled, label='Future Rolling Forecast (30 days)', color='red', linestyle='--')

plt.axvline(x=test_start_idx, color='gray', linestyle='--', label='Test Split Start')
plt.title(f'T2M Prediction (R²: {r2:.3f})')
plt.xlabel('Days (2024)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('meteorological_forecast.png') # গ্রাফটি সেভ করা হচ্ছে
plt.show()
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# ==========================================
# 1. Advanced Data Preparation (5D Tensor)
# ==========================================
def load_convlstm_data(file_path, seq_length=7):
    ds = xr.open_dataset(file_path)
    # Full Spatial Data: (Time, Lat, Lon)
    data = ds['t2m'].values

    # Missing Value Imputation
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=np.nanmean(data))

    # MinMax Scaling (Global Scale)
    scaler = MinMaxScaler()
    orig_shape = data.shape  # (Time, H, W)
    data_flat = data.reshape(-1, 1)
    data_scaled = scaler.fit_transform(data_flat).reshape(orig_shape)

    # Create Sequences: (Batch, Time, Channels, Height, Width)
    X, Y = [], []
    for i in range(len(data_scaled) - seq_length):
        # Input: Past frames (seq_len, H, W) -> Add Channel Dim -> (seq_len, 1, H, W)
        x_seq = data_scaled[i: i + seq_length]
        x_seq = np.expand_dims(x_seq, axis=1)

        # Target: Next frame (1, H, W)
        y_target = data_scaled[i + seq_length]
        y_target = np.expand_dims(y_target, axis=0)

        X.append(x_seq)
        Y.append(y_target)

    return np.array(X), np.array(Y), scaler, orig_shape


# ==========================================
# 2. The ConvLSTM Cell (Research Grade Component)
# ==========================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


# ==========================================
# 3. ConvLSTM Network Architecture
# ==========================================
class ConvLSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, True)
        # Final Convolution to map back to 1 channel output
        self.final_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Channel, H, W)
        batch_size, seq_len, _, h, w = x.size()
        hidden_state = self.cell.init_hidden(batch_size, (h, w))

        # Unroll LSTM over time steps
        for t in range(seq_len):
            input_t = x[:, t, :, :, :]
            hidden_state = self.cell(input_t, hidden_state)

        # Last hidden state contains spatiotemporal features
        h_last, _ = hidden_state
        # Decode to image
        prediction = self.final_conv(h_last)
        return prediction


# ==========================================
# 4. Training Pipeline
# ==========================================
file_path = 't2m.nc'  # YOUR FILE PATH
seq_len = 7
batch_size = 8  # Lower batch size due to heavy memory usage
epochs = 50  # Advanced models converge faster
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Load
print("Loading Data for Spatiotemporal Analysis...")
X_raw, Y_raw, scaler, shape = load_convlstm_data(file_path, seq_len)
# shape comes as (Time, H, W) e.g., (366, 45, 129)

train_size = int(len(X_raw) * 0.8)
X_train = torch.FloatTensor(X_raw[:train_size]).to(device)
Y_train = torch.FloatTensor(Y_raw[:train_size]).to(device)  # Shape: (B, 1, H, W)
X_test = torch.FloatTensor(X_raw[train_size:]).to(device)
Y_test = torch.FloatTensor(Y_raw[train_size:]).to(device)

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)

# Initialize Advanced Model
model = ConvLSTM_Model(input_dim=1, hidden_dim=32, kernel_size=(3, 3), num_layers=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Training Advanced ConvLSTM on {device}...")
history = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        # Output is entire map
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    history.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

# ==========================================
# 5. Advanced Visualization (Map Comparison)
# ==========================================
model.eval()
with torch.no_grad():
    # Predict on one test sample
    sample_idx = 10
    sample_input = X_test[sample_idx:sample_idx + 1]  # (1, Seq, 1, H, W)
    sample_target = Y_test[sample_idx:sample_idx + 1]  # (1, 1, H, W)

    prediction = model(sample_input)

    # Reshape for plotting
    pred_map = prediction.cpu().numpy().squeeze()
    target_map = sample_target.cpu().numpy().squeeze()

    # Inverse Transform (Manual for Maps)
    # We use the scaler's min/scale to revert normalization manually for 2D maps
    min_val = scaler.min_[0]
    scale_val = scaler.scale_[0]

    pred_map_K = (
                             pred_map - min_val) / scale_val  # Fix: Inverse logic might differ based on scaler implementation, usually (x - min) / scale is normalization.
    # Correct Inverse: X_orig = X_scaled / scale + min - (min/scale) ... actually simpler:
    pred_map_K = pred_map * (1 / scale_val) + (
                min_val * -1 / scale_val)  # Standard scaler inverse logic is confusing manually.
    # Easy way:
    pred_map_K = scaler.inverse_transform(pred_map.reshape(-1, 1)).reshape(pred_map.shape)
    target_map_K = scaler.inverse_transform(target_map.reshape(-1, 1)).reshape(target_map.shape)

# Plotting Side-by-Side Heatmaps
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.title("Actual Temperature Map (Ground Truth)")
plt.imshow(target_map_K, cmap='jet')
plt.colorbar(label='Temp (K)')

plt.subplot(1, 2, 2)
plt.title("ConvLSTM Predicted Map (Innovation)")
plt.imshow(pred_map_K, cmap='jet')
plt.colorbar(label='Temp (K)')

plt.suptitle(f"Spatiotemporal Forecast (Epoch {epochs})", fontsize=16)
plt.savefig('innovation_convlstm_map.png')
print("Advanced Map Prediction saved as 'innovation_convlstm_map.png'")
plt.show()

# Metric Calculation (Average over the whole map)
mae = mean_absolute_error(target_map_K.flatten(), pred_map_K.flatten())
r2 = r2_score(target_map_K.flatten(), pred_map_K.flatten())
print(f"\nAdvanced Spatial Metrics -> MAE: {mae:.4f} K, R²: {r2:.4f}")
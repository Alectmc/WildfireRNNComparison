#!/usr/bin/env python
# coding: utf-8

# # Comparing Performance of Neural Network Architectures Using Data on Wildfire Propagation
# 
# ## Author: Alec Creasy
# 
# ### This notebook/script allows you to re-create my experiment in my Final paper for CSCI 6620 - Research Methods in Computer Science at Middle Tennessee State University.
# 
# ## IMPORTANT: Before running this notebook, make sure you have installed all dependencies and be sure you get a Map API key for the satellite data! (Info in the README.md file.)

# # Collection of the Data

# In[ ]:


import requests
import pandas as pd
from io import StringIO
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torchview import draw_graph
import torchmetrics
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Define your MAP_KEY
MAP_KEY = "00000000000000000"  # Replace with your actual MAP_KEY

# Try and get the data. If valid, you will see information pertaining to your key. Otherwise
# an error will be reported. This usually has to do with having an invalid API key.
url = 'https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=' + MAP_KEY
try:
  df = pd.read_json(url,  typ='series')
  display(df)
except:
  # possible error, wrong MAP_KEY value, check for extra quotes, missing letters
  print ("There is an issue with the query. \nTry in your browser: %s" % url)


# In[ ]:


#Define time range
start_date = '2018-11-01'
end_date   = '2018-12-01'

#Build FIRMS API URL. The URL contains the dates to look for data (November 1, 2018-December 1, 2018) and the latitude and longitude range
# (Butte County, CA)
modis_url = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/' + MAP_KEY + f'/MODIS_SP/-122,39.5,-121.2,40.1/10/2018-11-01/2018-12-01'

# Get MODIS Satellite data from November 2, 2018-November 10, 2018, when the Butte County fires took place.
df_area = pd.read_csv(modis_url)

#Displays the dataset
display(df_area)


# # Pre-process the Data

# In[ ]:


# Generates binary grids from the satellite data that represent whether or not an area ir burning or not.
def build_fire_grid(df, lat_min, lat_max, lon_min, lon_max, grid_size=0.01):

    # Create the latitude and longitude cells with a cell size of .01 degrees (latitude and longitude).
    # Store the size of these as height and width of the grids, and convert the acquired
    # dates from the data to datetime objects to extract the minimum dates and maximum dates.
    lat_bins = np.arange(lat_min, lat_max + grid_size, grid_size)
    lon_bins = np.arange(lon_min, lon_max + grid_size, grid_size)
    H, W = len(lat_bins), len(lon_bins)
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    all_dates = pd.date_range(df['acq_date'].min(), df['acq_date'].max())

    # Create an empty list for the grids
    daily_grids = []

    # For each date, create an area of zeros. Then, for each cell, determine if there was a fire
    # (that is, the latitude/longitude coordinate is in the dataset). If so, update the cell to 1 and add
    # to the grid list.
    for current_date in all_dates:
        day_data = df[df['acq_date'] == current_date]
        grid = np.zeros((H, W), dtype=np.uint8)
        
        lat_idx = np.digitize(day_data['latitude'], lat_bins) - 1
        lon_idx = np.digitize(day_data['longitude'], lon_bins) - 1
        
        for y, x in zip(lat_idx, lon_idx):
            if 0 <= y < H and 0 <= x < W:
                grid[y, x] = 1
        daily_grids.append(grid)

    # Return the grid along with its height and width.
    return np.array(daily_grids), H, W


# In[ ]:


# This will take our daily grids and make 3x3 spatial grids with 5 time steps
# that will be used as our data for the model.
def generate_patch_sequences(daily_grids, seq_len=5, patch_size=3):
    T, H, W = daily_grids.shape

    # Adds spatial padding
    pad = patch_size // 2
    padded_grids = np.pad(daily_grids, ((0, 0), (pad, pad), (pad, pad)), mode='constant')

    # Create empty lists for our X and Y values in training (X is the
    # 5 previous time steps, y is the 6th correct time step)
    x_seqs, y_targets = [], []

    # Loops through every pixel in the daily fire grids.
    for i in range(H):
        for j in range(W):

            
            pixel_series = daily_grids[:, i, j] 

            # If no fire is present in this cell, skip this iteration and continue.
            if pixel_series.sum() == 0:
                continue

            
            for t in range(T - seq_len):
                # Sequence of patches around (i, j). The x_seq is the sequence of the last 5 time steps.
                # y is the next time step to be predicted.
                x_seq = padded_grids[t:t+seq_len, i:i+patch_size, j:j+patch_size]  # (seq_len, patch, patch)
                y = daily_grids[t+seq_len, i, j]

                # Reshape to (seq_len, 1, patch_size, patch_size)
                x_seqs.append(x_seq[:, None, :, :])
                y_targets.append([[y]])

    # Convert the 3x3 patches to tensors and return them
    x_tensor = torch.tensor(x_seqs, dtype=torch.float32)  # (N, seq_len, 1, patch, patch)
    y_tensor = torch.tensor(y_targets, dtype=torch.float32)  # (N, 1, 1)
    return x_tensor, y_tensor


# In[ ]:


# Create a Data Module using PyTorch Lightning. Sets up the data loaders for the training and validation
# data, as well as splits the data into 80% training data and 20% validation data.
class FireDataModule(pl.LightningDataModule):
    def __init__(self, x, y, batch_size=32, num_workers=2):
        super().__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        N = len(self.x)
        split = int(N * 0.8)
        self.train_dataset = FireDataset(self.x[:split], self.y[:split])
        self.val_dataset = FireDataset(self.x[split:], self.y[split:])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)


# In[ ]:


# Set bounded region for Butte County, CA
lat_min, lat_max = 39.5, 40.1
lon_min, lon_max = -122.0, -121.2

# 1. Grid the fire data
daily_grids, H, W = build_fire_grid(df_area, lat_min, lat_max, lon_min, lon_max)

# 2. Generate patch sequences
x_tensor, y_tensor = generate_patch_sequences(daily_grids, seq_len=5, patch_size=3)

# 3. Create DataModule and set it up.
data_module = FireDataModule(x_tensor, y_tensor, batch_size=32)
data_module.setup()


# # Create the Models

# In[ ]:


# Creates the ConvLSTM cell to be used in the ConvLSTM model. Modifies the LSTM cell to include Convolution.
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=input_channels + hidden_channels,
                              out_channels=4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat([x, h_prev], dim=1)
        conv_output = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# In[ ]:


# Creates a module to stack various ConvLSTM cells to be used in the ConvLSTM model.
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dims = hidden_dims

        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list.append(ConvLSTMCell(cur_input_dim, hidden_dims[i], kernel_size))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor):
        B, T, _, H, W = input_tensor.size()

        h = [torch.zeros(B, hidden_dim, H, W, device=input_tensor.device) for hidden_dim in self.hidden_dims]
        c = [torch.zeros(B, hidden_dim, H, W, device=input_tensor.device) for hidden_dim in self.hidden_dims]

        layer_outputs = []

        for t in range(T):
            x = input_tensor[:, t]  # (B, C, H, W)

            for i, cell in enumerate(self.cell_list):
                h[i], c[i] = cell(x, (h[i], c[i]))
                x = h[i]  # pass to next layer

            layer_outputs.append(h[-1])  # keep only last layer's output

        output_seq = torch.stack(layer_outputs, dim=1)
        return output_seq


# In[ ]:


# Creates the ConvLSTM Model.
class ConvLSTMModel(pl.LightningModule):
    def __init__(self, input_channels=1, hidden_dims=[64, 128, 256], kernel_size=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Creates the stacked ConvLSTM layers.
        self.convlstm = ConvLSTM(
            input_dim=input_channels,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            num_layers=len(hidden_dims)
        )

        # Final 3D conv layer to collapse time + channel dims
        self.final_conv = nn.Conv3d(
            in_channels=hidden_dims[-1],
            out_channels=1,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )

        self.sigmoid = nn.Sigmoid()

        self.train_f1 = torchmetrics.classification.BinaryF1Score(threshold=0.5)
        self.val_f1 = torchmetrics.classification.BinaryF1Score(threshold=0.5)
        self.test_f1 = torchmetrics.classification.BinaryF1Score(threshold=0.5)

    def forward(self, x):
        features = self.convlstm(x) 
        features = features.permute(0, 2, 1, 3, 4) 

        out = self.final_conv(features)           
        out = self.sigmoid(out[:, :, -1])         
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        center = y_hat[:, :, 1, 1]
        loss = F.binary_cross_entropy(center, y.squeeze(-1))
        acc = self.train_f1(center, y.squeeze(-1))
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        center = y_hat[:, :, 1, 1]
        loss = F.binary_cross_entropy(center, y.squeeze(-1))
        acc = self.val_f1(center, y.squeeze(-1))
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        center = y_hat[:, :, 1, 1]
        loss = F.binary_cross_entropy(center, y.squeeze(-1))
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# In[ ]:


# Creates the ConvGRU cell to be used in the ConvGRU model. Modifies the GRU cell to include Convolution.
class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.reset_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=bias)
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=bias)
        self.out_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=bias)

    def forward(self, x, h_prev):
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim, x.size(2), x.size(3), device=x.device)

        combined = torch.cat([x, h_prev], dim=1)
        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))

        combined_reset = torch.cat([x, reset * h_prev], dim=1)
        out = torch.tanh(self.out_gate(combined_reset))

        h_new = (1 - update) * h_prev + update * out
        return h_new


# In[ ]:


# Creates a module to stack various ConvGRU cells to be used in the ConvGRU model.
class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers):
        super().__init__()
        assert len(hidden_dims) == num_layers, "hidden_dims must match num_layers"

        self.num_layers = num_layers
        self.hidden_dims = hidden_dims

        cells = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell = ConvGRUCell(cur_input_dim, hidden_dims[i], kernel_size)
            cells.append(cell)

        self.cells = nn.ModuleList(cells)

    def forward(self, x):
        B, T, _, H, W = x.size()
        h = [None] * self.num_layers
        outputs = []

        for t in range(T):
            input_t = x[:, t]
            for i, cell in enumerate(self.cells):
                h[i] = cell(input_t, h[i])
                input_t = h[i] 
            outputs.append(h[-1].unsqueeze(1))

        return torch.cat(outputs, dim=1) 


# In[ ]:


# Creates the ConvGRU model using the ConvGRU stacked layers.
class ConvGRUModel(pl.LightningModule):
    def __init__(self, input_channels=1, hidden_dims=[64, 128, 256], kernel_size=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Creates the stacked ConvGRU layers.
        self.convgru = ConvGRU(
            input_dim=input_channels,
            hidden_dims=hidden_dims,
            kernel_size=kernel_size,
            num_layers=len(hidden_dims)
        )

        # Final 3D conv layer to collapse time + channel dims
        self.final_conv = nn.Conv3d(
            in_channels=hidden_dims[-1],
            out_channels=1,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )

        self.sigmoid = nn.Sigmoid()

        self.train_f1 = torchmetrics.classification.BinaryF1Score(threshold=0.5)
        self.val_f1 = torchmetrics.classification.BinaryF1Score(threshold=0.5)
        self.test_f1 = torchmetrics.classification.BinaryF1Score(threshold=0.5)

    def forward(self, x):
        features = self.convgru(x)
        features = features.permute(0, 2, 1, 3, 4)

        out = self.final_conv(features)
        out = self.sigmoid(out[:, :, -1])
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        center = y_hat[:, :, 1, 1]
        loss = F.binary_cross_entropy(center, y.squeeze(-1))
        acc = self.train_f1(center, y.squeeze(-1))
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        center = y_hat[:, :, 1, 1]
        loss = F.binary_cross_entropy(center, y.squeeze(-1))
        acc = self.val_f1(center, y.squeeze(-1))
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        center = y_hat[:, :, 1, 1]
        loss = F.binary_cross_entropy(center, y.squeeze(-1))
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# In[ ]:


# Create the ConvLSTM model and print a summary of the model
lstm_model = ConvLSTMModel()
summary(lstm_model, input_size=(32, 5, 1, 32, 32))  # (B, T, C, H, W)


# In[ ]:


# Create the ConvGRU model and print a summary of the model
gru_model = ConvGRUModel()
summary(gru_model, input_size=(32, 5, 1, 32, 32))  # (B, T, C, H, W)


# In[ ]:


# Model the ConvLSTM model.
model_graph = draw_graph(lstm_model, input_size=(32, 5, 1, 3, 3), depth=1)
model_graph.visual_graph


# In[ ]:


# Model the ConvGRU model.
model_graph = draw_graph(gru_model, input_size=(32, 5, 1, 3, 3), depth=1)
model_graph.visual_graph


# # Training the ConvLSTM model

# In[ ]:


logger = pl.loggers.CSVLogger("logs",
                              name="LSTM-Log",
                             version="lstm")


# In[ ]:


trainer = pl.Trainer(logger=logger,
                     max_epochs=50,
                     enable_progress_bar=True,
                     log_every_n_steps=0,
                     callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=20)])


# In[ ]:


trainer.fit(lstm_model, data_module)


# # Training the ConvGRU Model

# In[ ]:


logger = pl.loggers.CSVLogger("logs",
                              name="GRU-Log",
                             version="gru")


# In[ ]:


trainer = pl.Trainer(logger=logger,
                     max_epochs=50,
                     enable_progress_bar=True,
                     log_every_n_steps=0,
                     callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=20)])


# In[ ]:


trainer.fit(gru_model, data_module)


# In[ ]:


lstm_results = pd.read_csv("logs/LSTM-Log/lstm/metrics.csv")
gru_results = pd.read_csv("logs/GRU-Log/gru/metrics.csv")


# # Plotting of Results

# In[ ]:


plt.plot(lstm_results["epoch"][np.logical_not(np.isnan(lstm_results["train_loss"]))],
         lstm_results["train_loss"][np.logical_not(np.isnan(lstm_results["train_loss"]))],
         label="LSTM")
plt.plot(gru_results["epoch"][np.logical_not(np.isnan(gru_results["train_loss"]))],
         gru_results["train_loss"][np.logical_not(np.isnan(gru_results["train_loss"]))],
         label="GRU")
plt.legend()
plt.ylabel("BCE Loss")
plt.xlabel("Epoch")
plt.title("Training Loss")
plt.show()


# In[ ]:


plt.plot(lstm_results["epoch"][np.logical_not(np.isnan(lstm_results["train_acc"]))],
         lstm_results["train_acc"][np.logical_not(np.isnan(lstm_results["train_acc"]))],
         label="LSTM")
plt.plot(gru_results["epoch"][np.logical_not(np.isnan(gru_results["train_acc"]))],
         gru_results["train_acc"][np.logical_not(np.isnan(gru_results["train_acc"]))],
         label="GRU")
plt.legend()
plt.ylabel("BCE Loss")
plt.xlabel("Epoch")
plt.title("Training Loss")
plt.show()


# In[ ]:


plt.plot(lstm_results["epoch"][np.logical_not(np.isnan(lstm_results["val_loss"]))],
         lstm_results["val_loss"][np.logical_not(np.isnan(lstm_results["val_loss"]))],
         label="ConvLSTM")
plt.plot(gru_results["epoch"][np.logical_not(np.isnan(gru_results["val_loss"]))],
         gru_results["val_loss"][np.logical_not(np.isnan(gru_results["val_loss"]))],
         label="ConvGRU")
plt.legend()
plt.ylabel("BCE Loss")
plt.xlabel("Epoch")
plt.title("Validation Loss")
plt.show()


# In[ ]:


plt.plot(lstm_results["epoch"][np.logical_not(np.isnan(lstm_results["val_acc"]))],
         lstm_results["val_acc"][np.logical_not(np.isnan(lstm_results["val_acc"]))],
         label="ConvLSTM")
plt.plot(gru_results["epoch"][np.logical_not(np.isnan(gru_results["val_acc"]))],
         gru_results["val_acc"][np.logical_not(np.isnan(gru_results["val_acc"]))],
         label="ConvGRU")
plt.legend()
plt.ylabel("F1 Score")
plt.xlabel("Epoch")
plt.title("Validation F1")
plt.show()


# In[ ]:


model_graph = draw_graph(model, input_size=(32, 5, 1, 3, 3), depth=1)
model_graph.visual_graph


# In[ ]:


trainer.fit(model, data_module)


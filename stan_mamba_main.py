import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from data_engineering import span_data_3d
from Train import stan_main



# Config
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "S-FFSD.csv")
LOCAL_DATA_PATH = r"D:\Do\Đồ án\Code\S-FFSD\S-FFSD.csv"

if os.path.exists(DEFAULT_DATA_PATH):
    DATA_PATH = DEFAULT_DATA_PATH
elif os.path.exists(LOCAL_DATA_PATH):
    DATA_PATH = LOCAL_DATA_PATH
else:
    DATA_PATH = os.path.join("data", "S-FFSD.csv")

TIME_WINDOWS = [1, 3, 5, 10, 20, 50, 100, 500]
SPATIO_WINDOWS = [1, 2, 3, 4, 5]

# Device
device = "cpu"
print("=" * 60)
print("DEVICE:", device)
print("=" * 60)

# Load Data
print("\nLoading dataset...")
data = pd.read_csv(DATA_PATH)
print("Dataset shape:", data.shape)

# Feature Engineering
print("\nCreating STAN features...")
X, y = span_data_3d(
    data,
    time_windows=TIME_WINDOWS,
    spatio_windows=SPATIO_WINDOWS)

print("X shape:", X.shape)
print("Y shape:", y.shape)

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y # Phân chia phân tầng để giữ tỷ lệ gian lận
)

print("\nTRAIN")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTEST")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Train STAN + MAMBA
print("\n")
print("=" * 60)
print("START STAN + MAMBA")
print("=" * 60)

# Truyền trực tiếp các biến array vào hàm huấn luyện
stan_main(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    epochs=20,
    batch_size=256,
    attention_hidden_dim=150,
    lr=0.003,
    device=device)
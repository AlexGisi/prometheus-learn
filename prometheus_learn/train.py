import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from eval_model import EvalModel
from util import weights_from_file, weights_to_file


###----------

LR = 1e-3
BATCH_SIZE = 32
TEST_SPLIT = 0.2
EPOCHS = 100
DTYPE = torch.float32
NP_DTYPE = np.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_DIR = "../runs/test1"
WEIGHTS_INITIAL_FP = "../io/weights_initial.txt"
DATA_FP = "../data/balanced_positions_with_vecs_test.csv"

###----------

weights_initial_fp = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_INITIAL_FP)
)
data_fp = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FP)
)
run_dir_fp = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), RUN_DIR)
)

print("Loading data...")
data = []
data_df = pd.read_csv(data_fp)

for index, row in data_df.iterrows():
    position_vector = [int(x) for x in row["white"].split()] + [
        int(x) for x in row["black"].split()
    ]
    data.append(position_vector)

X = torch.tensor(data, dtype=DTYPE)
Y = torch.tensor(data_df["result"].values, dtype=DTYPE)

print(f"Data loaded: X has shape {X.shape} and Y has shape {Y.shape}.")
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=TEST_SPLIT, random_state=1337
)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

table_names, weights_initial_np = weights_from_file(weights_initial_fp, dtype=NP_DTYPE)
weights_initial = torch.from_numpy(weights_initial_np)
model = EvalModel(weights_initial)

optimizer = optim.SGD(model.parameters(), lr=LR)
criterion = nn.MSELoss()

writer = SummaryWriter(run_dir_fp)
print(f"Writing tensorboad log to {run_dir_fp}")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    writer.add_scalar("Loss/train", avg_train_loss, epoch)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    writer.add_scalar("Loss/validation", avg_val_loss, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram("param/" + name, param.cpu(), epoch)
        writer.add_histogram("grad/" + name, param.grad.cpu(), epoch)

    print(
        f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
    )

writer.close()

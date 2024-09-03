from datetime import datetime
import time
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

LR = 1.74
BATCH_SIZE = 128
TEST_SPLIT = 0.15
EPOCHS = 100
OPTIMIZER = 'Adagrad'
DTYPE = torch.float32
NP_DTYPE = np.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PERIOD = 5
SEED = 1337

RUN_DIR = "../runs/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
WEIGHTS_INITIAL_FP = "../io/weights_initial.txt"
WEIGHTS_FINAL_FP = "../io/weights_final.txt"
DATA_FP = "../data/balanced_positions_with_vecs.csv"

###----------

torch.manual_seed(SEED)

weights_initial_fp = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_INITIAL_FP)
)
weights_final_fp = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_FINAL_FP)
)
data_fp = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FP)
)
run_dir_fp = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), RUN_DIR)
)

print(f"Using device {DEVICE}.")

os.mkdir(run_dir_fp)
with open(__file__, 'r') as f_read:
    with open(os.path.join(run_dir_fp, 'train.py'), 'w') as f_write:
        f_write.write(f_read.read())

print("Loading data...")
data = []
data_df = pd.read_csv(data_fp)

for index, row in data_df.iterrows():
    position_vector = [int(x) for x in row["white"].split()] + [
        int(x) for x in row["black"].split()
    ]
    data.append(position_vector)

X = torch.tensor(data, dtype=DTYPE, device=DEVICE)
Y = torch.tensor(data_df["result"].values, dtype=DTYPE, device=DEVICE)

print(f"Data loaded: X has shape {X.shape} and Y has shape {Y.shape}.")
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=TEST_SPLIT, random_state=1337
)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

table_names, weights_initial_np = weights_from_file(weights_initial_fp, dtype=NP_DTYPE)
weights_initial = torch.from_numpy(weights_initial_np).to(DEVICE)
model = EvalModel(weights_initial).to(DEVICE)

if OPTIMIZER == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=LR)
elif OPTIMIZER == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif OPTIMIZER == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=LR)
elif OPTIMIZER == 'Adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=LR)
elif OPTIMIZER == 'Adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=LR)
elif OPTIMIZER == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=LR)
else:
    raise ValueError(f"Optimizer {OPTIMIZER} is not available.")
criterion = nn.MSELoss()


def train(model_, optimizer_, train_loader_, val_loader_):
    model_.train()
    running_loss = 0.0
    for inputs, targets in train_loader_:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer_.zero_grad()
        outputs = model_(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer_.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader_)

    model_.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader_:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model_(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader_)

    return model_, optimizer_, avg_train_loss, avg_val_loss


if __name__ == '__main__':
    writer = SummaryWriter(run_dir_fp)
    print(f"Writing tensorboard log to {run_dir_fp}")

    start = time.monotonic()
    for epoch in range(EPOCHS):
        model, optimizer, train_loss, val_loss = train(model, optimizer, train_loader, val_loader)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)
        for name, param in model.named_parameters():
            writer.add_histogram("param/" + name, param.cpu(), epoch)
            writer.add_histogram("grad/" + name, param.grad.cpu(), epoch)

        print(
            f"Epoch {epoch+1}/{EPOCHS}\t Train Loss: {train_loss:.8f}\t Validation Loss: {val_loss:.8f}"
        )

        # For loading see https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        if (epoch+1) % CHECKPOINT_PERIOD == 0:
            dir = os.path.join(run_dir_fp, 'checkpoints')
            if not os.path.exists(dir):
                os.mkdir(dir)

            fp = os.path.join(dir, f'epoch-{epoch}.tar')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimzer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, fp)

            weights_to_file(table_names, model.weights.detach().cpu(), fp=os.path.join(dir, 'weights.txt'))

    end = time.monotonic()
    print("Training finished in " + str(end-start) + " seconds")

    writer.close()
    weights_to_file(table_names, model.weights.detach().cpu(), fp=weights_final_fp)
    weights_to_file(table_names, model.weights.detach().cpu(), fp=os.path.join(run_dir_fp, 'pst-final.txt'))

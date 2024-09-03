import optuna
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import train
from eval_model import EvalModel
from util import weights_from_file

EVAL_EPOCH = 10
TRIAL_N = 800

def objective(trial):
    lr = trial.suggest_float('LR', 1e-2, 1e1, log=True)  # Learning rate
    batch_size = trial.suggest_categorical('BATCH_SIZE', [32, 64, 128, 256])  # Batch size
    optimizer_name = trial.suggest_categorical('OPTIMIZER', ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW'])

    _, weights_initial_np = weights_from_file(train.weights_initial_fp, dtype=train.NP_DTYPE)
    weights_initial = torch.from_numpy(weights_initial_np).to(train.DEVICE)
    model = EvalModel(weights_initial).to(train.DEVICE)

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(train.train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(train.val_dataset, batch_size=batch_size, shuffle=False)

    for _ in range(EVAL_EPOCH):
        model, optimizer, _, avg_val_loss = train.train(model, optimizer, train_loader, val_loader)

    return avg_val_loss  # Return final validation loss.

study = optuna.create_study(direction='minimize')

start = time.monotonic()
study.optimize(objective, n_trials=TRIAL_N)
end = time.monotonic()

print(f"Study finished in {str(end-start)}s")
print("Best hyperparameters found: ", study.best_params)
print("Best validation loss: ", study.best_value)

sorted_trials = sorted(study.trials, key=lambda trial: trial.value)

n = 10
print(f"Best {n} trials:")
for i, trial in enumerate(sorted_trials[:n]):
    print(f"Trial {i+1}:")
    print(f"  Value: {trial.value}")
    print(f"  Params: {trial.params}")
    print(f"  Number: {trial.number}")
    print(f"  User Attributes: {trial.user_attrs}")
    print(f"  State: {trial.state}")

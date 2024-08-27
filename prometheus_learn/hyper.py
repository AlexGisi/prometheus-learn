import optuna
import time
import torch.optim as optim
from torch.utils.data import DataLoader
import train
from eval_model import EvalModel

EVAL_EPOCH = 2000
TRIAL_N = 2000

def objective(trial):
    lr = trial.suggest_float('LR', 1e-2, 1e1, log=True)  # Learning rate
    batch_size = trial.suggest_categorical('BATCH_SIZE', [32, 64, 128, 256])  # Batch size
    optimizer_name = trial.suggest_categorical('OPTIMIZER', ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW'])

    model = EvalModel(train.weights_initial).to(train.DEVICE)

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

    train_loader = DataLoader(train.train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train.val_dataset, batch_size=batch_size, shuffle=False)

    avg_loss = 0
    for _ in range(EVAL_EPOCH):
        model, optimizer, _, avg_val_loss = train.train(model, optimizer, train_loader, val_loader)
        avg_loss += avg_val_loss
    avg_loss /= EVAL_EPOCH

    return avg_loss

study = optuna.create_study(direction='minimize')

start = time.monotonic()
study.optimize(objective, n_trials=TRIAL_N)
end = time.monotonic()

print(f"Study finished in {str(end-start)}s")
print("Best hyperparameters found: ", study.best_params)
print("Best validation loss: ", study.best_value)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .data import create_loaders


def train(model, dataset, save_dir, epochs, batch_size, criterion, learning_rate, optimizer,
          val_split=0.05, frequency=10, anneal=False, patience=2):

    # Setup dataloaders and model
    train_loader, val_loader = create_loaders(dataset, batch_size, val_split)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    assert hasattr(model, 'name'), 'model.name attrbute required for saving checkpoints'
    print(f'Loaded model on {str(device).upper()} device')

    # Optimizer, criterion, and scheduler
    if isinstance(criterion, str):
        try:
            criterion = getattr(nn, criterion)
        except AttributeError:
            raise ValueError(f"Requested criterion '{criterion}' does not exist in torch.nn")
    if isinstance(optimizer, str):
        try:
            optim_class = getattr(optim, optimizer)
        except AttributeError:
            raise ValueError(f"Requested optimizer '{optimizer}' does not exist in torch.optim")
        optimizer = optim_class(model.parameters(), lr=learning_rate)
    elif optim.__name__ not in optimizer.__module__:
        raise ValueError(f"Optimizer '{optimizer}' is not str or member of torch.optim module")
    else:
        print(f'Optimizer learning rate {optimizer.param_groups[0]["lr"]} overrides paramter {learning_rate}')
    if anneal:
        scheduler = ReduceLROnPlateau(optimizer, patience=patience)
    else:
        scheduler = None

    # Setup checkpointing
    save_dir = os.path.realpath(save_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, mode=0o755)

    # Loop over batches
    val_costs = []
    for epoch in range(epochs):

        lr = optimizer.param_groups[0]['lr']
        for batch_id, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            if batch_id % frequency == 0:
                print(f'Epoch {epoch}/{epochs - 1} | '
                      f'batch {batch_id}/{len(train_loader)} | '
                      f'LR {lr:.0e} | '
                      f'loss {loss:.4f}')

        val_cost = 0
        for batch_id, (inputs, targets) in enumerate(val_loader):
            scores = model(inputs)
            loss = criterion(scores, targets)
            val_cost += loss.item() / len(val_loader)

        val_costs.append(val_cost)
        msg = f'Epoch {epoch} | validation cost {val_cost:.4f}'
        border = '=' * len(msg)
        print(f'{border}\n{msg}\n{border}')
        if scheduler is not None:
            scheduler.step(val_costs[-1])

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_cost': val_costs[-1],
            'loss': loss}, os.path.join(save_dir, f'{model.name}_epoch_{epoch}.pt'))

import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    model.train()
    total_loss = 0 
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/ len(dataloader)
import torch.nn as nn
from tqdm import tqdm


def train(model, train_loader, optimizer, writer, epochs=60):
    global inputs
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for inputs, targets in pbar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.BCELoss()(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()

            predicted = (outputs > 0.5).float()
            total = targets.size(0)
            correct = (predicted == targets).sum().item()

            epoch_loss += loss.item()
            epoch_correct += correct
            epoch_total += total

            pbar.set_postfix(
                {'loss': epoch_loss / (epoch_total + 1e-7), 'accuracy': 100. * epoch_correct / epoch_total})
        outputs = model(inputs)
        writer.add_histogram('activations', outputs, epoch)
        writer.add_scalar('Loss', epoch_loss, epoch)
        writer.add_scalar('Accuracy', 100. * epoch_correct / epoch_total, epoch)

import torch


def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def mae(preds, y):
    return torch.mean(torch.abs(preds - y)).item()


def _compute_metric(logits, y, task: str):
    if task == "classification":
        return accuracy(logits, y)
    preds = logits.squeeze(-1)
    return mae(preds, y)


def train_one_epoch(model, loader, optimizer, loss_fn, device, task: str = "classification"):
    model.train()
    total_loss = 0.0
    total_metric = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        if task == "regression":
            logits = logits.squeeze(-1)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_metric += _compute_metric(logits, y, task) * bs
        n += bs

    return total_loss / n, total_metric / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, task: str = "classification"):
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if task == "regression":
            logits = logits.squeeze(-1)
        loss = loss_fn(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_metric += _compute_metric(logits, y, task) * bs
        n += bs

    return total_loss / n, total_metric / n

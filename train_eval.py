import torch
from sklearn import metrics
from torch.optim import Adam
from torch_geometric.data import DataLoader
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)

def evaluate_metric(model, loader, device):
    model.eval()

    y_true = []
    y_score = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out = model(data)

            y_true.append(data.y.view(-1).cpu())
            y_score.append(out.view(-1).cpu())

    y_true = torch.cat(y_true).numpy()
    y_score = torch.cat(y_score).numpy()

    auc = metrics.roc_auc_score(y_true, y_score)
    aupr = metrics.average_precision_score(y_true, y_score)

    return auc, aupr


def train_epochs(train_dataset, test_dataset, model, args):

    train_loader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=4
    )

    test_loader = DataLoader(
        test_dataset,
        1024,
        shuffle=False,
        num_workers=4
    )

    model.to(args.device).reset_parameters()

    optimizer = Adam(
        model.parameters(),
        lr=args.lr
    )

    best_auc = 0
    best_aupr = 0

    for epoch in range(1, args.epochs + 1):

        train_loss = train(
            model,
            optimizer,
            train_loader,
            args.device,
            args.lambda_reg
        )

        if epoch % args.valid_interval == 0:

            auc, aupr = evaluate_metric(
                model,
                test_loader,
                args.device
            )

            print(
                'Epoch {}, Loss {:.4f}, AUC {:.4f}, AUPR {:.4f}'.format(
                    epoch,
                    train_loss,
                    auc,
                    aupr
                )
            )

            if auc > best_auc:
                best_auc = auc
                best_aupr = aupr

    return best_auc, best_aupr


def train(model, optimizer, loader, device,
          lambda_reg,
          max_grad_norm=1.0):

    model.train()

    total_loss = 0

    loss_function = torch.nn.BCEWithLogitsLoss()

    for data in loader:

        optimizer.zero_grad()

        data = data.to(device)

        predict = model(data)

        bce_loss = loss_function(
            predict.view(-1),
            data.y.view(-1)
        )

        bnm_loss = (
            lambda_reg *
            torch.norm(data.x, p='nuc')
        )

        loss = bce_loss + bnm_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm
        )

        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)
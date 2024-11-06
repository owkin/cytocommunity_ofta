import torch
from cytocommunity_ofta.utils.constants import device
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score


def train(model, train_loader, optimizer, beta, train_dataset):
    model.train()
    loss_all = 0
    loss_CE_all = 0
    loss_MinCut_all = 0
    all_preds = []
    all_labels = []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, mc_loss, o_loss, _, _ = model(data.x, data.adj, data.mask)
        
        loss_CE = F.nll_loss(out, data.y.view(-1))
        loss_MinCut = mc_loss + o_loss
        loss = loss_CE * (1 - beta) + loss_MinCut * beta
        loss.backward()
        
        loss_all += data.y.size(0) * loss.item()
        loss_CE_all += data.y.size(0) * loss_CE.item()
        loss_MinCut_all += data.y.size(0) * loss_MinCut.item()
        
        # Collect predictions and labels for AUC calculation
        all_preds.extend(torch.exp(out)[:, 1].detach().cpu().numpy())  # Assume binary classification
        all_labels.extend(data.y.view(-1).detach().cpu().numpy())
        
        optimizer.step()
    
    # Calculate AUC
    # In the train function
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else None
    
    return (
        loss_all / len(train_dataset),
        loss_CE_all / len(train_dataset),
        loss_MinCut_all / len(train_dataset),
        auc
    )


@torch.no_grad()
def test(loader, model):
    model.eval()
    correct = 0
    pr_Table = np.zeros([1, 4])
    loss_MinCut_all = 0
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        ModelResultPr, mc_loss, o_loss, _, _ = model(data.x, data.adj, data.mask)
        pred = ModelResultPr.max(dim=1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()

        # MinCut loss
        loss_MinCut = mc_loss + o_loss
        loss_MinCut_all += data.y.size(0) * loss_MinCut.item()

        pred_info = np.column_stack(
            (
                np.array(torch.exp(ModelResultPr)),
                np.array(pred),
                np.array(data.y.view(-1)),
            )
        )
        pr_Table = np.row_stack((pr_Table, pred_info))
        
        # Collect predictions and labels for AUC calculation
        all_preds.extend(torch.exp(ModelResultPr)[:, 1].detach().cpu().numpy())  # Assuming binary classification
        all_labels.extend(data.y.view(-1).detach().cpu().numpy())

    # Calculate AUC
    auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else None
    
    return (
        correct / len(loader.dataset),
        pr_Table,
        auc,
        loss_MinCut_all / len(loader.dataset)
    )

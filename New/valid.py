import torch
from utils import confusion_matrix
from sklearn.metrics import roc_auc_score


def valid(config, net, val_loader, criterion, use_tta=False):
    device = next(net.parameters()).device
    net.eval()

    print("START VALIDATING")
    epoch_loss = 0
    y_true, y_score = [], []

    cm = torch.zeros((config.class_num, config.class_num), device=device)

    with torch.no_grad():
        for pack in val_loader:
            images = pack['imgs'].to(device)
            labels = pack['labels'].to(device)

            if images.shape[1] == 1:
                images = images.expand((-1, 3, -1, -1))

            # ===============================
            # TTA: Original + Horizontal Flip
            # ===============================
            if use_tta:
                output1 = net(images)
                images_flip = torch.flip(images, dims=[3])
                output2 = net(images_flip)
                output = (output1 + output2) / 2
            else:
                output = net(images)

            loss = criterion(output, labels)
            epoch_loss += loss.item()

            probs = torch.softmax(output, dim=1)
            preds = probs.argmax(dim=1)

            cm = confusion_matrix(preds, labels, cm)

            y_true.extend(labels.cpu().numpy().tolist())
            y_score.extend(probs[:, 1].cpu().numpy().tolist())

    avg_epoch_loss = epoch_loss / len(val_loader)

    # ===============================
    # Metrics
    # ===============================
    acc = cm.diag().sum() / cm.sum()
    spe, sen = cm.diag() / (cm.sum(dim=0) + 1e-6)
    pre = cm.diag()[1] / (cm.sum(dim=1) + 1e-6)[1]
    rec = sen
    f1score = 2 * pre * rec / (pre + rec + 1e-6)

    auc = roc_auc_score(y_true, y_score)

    return avg_epoch_loss, acc, sen, spe, auc, pre, f1score

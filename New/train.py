import torch
import os
import torch.nn as nn
import utils
from config import config
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold   # ✅ CHANGED
from torch.utils.data import DataLoader, SubsetRandomSampler
from valid import valid
from hovertrans import create_model
from utils import confusion_matrix
import math


def train(config, train_loader, test_loader, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===============================
    # MODEL
    # ===============================
    model = create_model(
        img_size=config.img_size,
        num_classes=config.class_num,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        patch_size=config.patch_size,
        dim=config.dim,
        depth=config.depth,
        num_heads=config.num_heads,
        num_inner_head=config.num_inner_head
    ).to(device)

    # ===============================
    # LOSS (Label Smoothing)
    # ===============================
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05).to(device)

    # ===============================
    # OPTIMIZER
    # ===============================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-4
    )

    # ===============================
    # COSINE LR + WARMUP
    # ===============================
    lr_lambda = lambda epoch: (
        epoch / config.warmup_epochs
        if epoch < config.warmup_epochs else
        0.5 * (1 + math.cos(
            (epoch - config.warmup_epochs) /
            (config.epochs - config.warmup_epochs) * math.pi))
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    writer = SummaryWriter(
        comment=f'_{config.model_name}_{config.writer_comment}_fold{fold}'
    )

    print("START TRAINING")

    best_acc = 0.0

    ckpt_path = os.path.join(
        config.model_path,
        config.model_name,
        config.writer_comment
    )
    model_save_path = os.path.join(ckpt_path, str(fold))
    os.makedirs(model_save_path, exist_ok=True)

    # ===============================
    # TRAINING LOOP
    # ===============================
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        cm = torch.zeros((config.class_num, config.class_num))

        for pack in train_loader:
            images = pack['imgs'].to(device)
            if images.shape[1] == 1:
                images = images.expand(-1, 3, -1, -1)

            labels = pack['labels'].to(device)

            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1)
            cm = confusion_matrix(pred.detach(), labels.detach(), cm)
            epoch_loss += loss.item()

        lr_scheduler.step()

        print(f"[Epoch {epoch+1}/{config.epochs}] Train Loss: {epoch_loss:.4f}")

        # ===============================
        # VALIDATION (TTA ENABLED ✅)
        # ===============================
        if (epoch + 1) % config.log_step == 0:
            with torch.no_grad():
                val_loss, val_acc, sen, spe, auc, pre, f1score = valid(
                    config,
                    model,
                    test_loader,
                    criterion,
                    use_tta=True        # ✅ CHANGED
                )

            writer.add_scalar('Val/Acc', val_acc, epoch)
            writer.add_scalar('Val/F1', f1score, epoch)

            print(
                f" Val Acc: {val_acc:.4f} | Sen: {sen:.4f} | "
                f"Spe: {spe:.4f} | AUC: {auc:.4f}"
            )

            if epoch > config.epochs // 4 and val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(model_save_path, 'bestmodel.pth')
                )
                print("=> Saved Best Model")

                with open(os.path.join(model_save_path, 'result.txt'), 'w') as f:
                    f.write("Best Result:\n")
                    f.write(
                        f"Acc: {val_acc:.6f}, Spe: {spe:.6f}, "
                        f"Sen: {sen:.6f}, AUC: {auc:.6f}, "
                        f"Pre: {pre:.6f}, F1: {f1score:.6f}\n"
                    )

    # ===============================
    # FINE-TUNING
    # ===============================
    print("START FINE-TUNING")

    model.load_state_dict(
        torch.load(os.path.join(model_save_path, 'bestmodel.pth'))
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr * 0.1,
        weight_decay=1e-4
    )

    for epoch in range(20):
        model.train()
        for pack in train_loader:
            images = pack['imgs'].to(device)
            if images.shape[1] == 1:
                images = images.expand(-1, 3, -1, -1)

            labels = pack['labels'].to(device)

            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(
        model.state_dict(),
        os.path.join(model_save_path, 'finetuned_model.pth')
    )

    print("Fine-tuning completed")

    # ===============================
    # FINAL RESULT (TTA ENABLED)
    # ===============================
    with torch.no_grad():
        val_loss, val_acc, sen, spe, auc, pre, f1score = valid(
            config,
            model,
            test_loader,
            criterion,
            use_tta=True          # ✅ CHANGED
        )

    with open(os.path.join(model_save_path, 'result.txt'), 'a') as f:
        f.write("\nFinal Result:\n")
        f.write(
            f"Acc: {val_acc:.6f}, Spe: {spe:.6f}, "
            f"Sen: {sen:.6f}, AUC: {auc:.6f}, "
            f"Pre: {pre:.6f}, F1: {f1score:.6f}\n"
        )

    print(f"=> Results saved to {model_save_path}/result.txt")


# ===============================
# SEED
# ===============================
def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===============================
# MAIN
# ===============================
if __name__ == '__main__':
    seed_torch(42)
    args = config()

    train_set = utils.get_dataset(
        args.data_path, args.csv_path, args.img_size, mode='train'
    )
    test_set = utils.get_dataset(
        args.data_path, args.csv_path, args.img_size, mode='test'
    )

    labels = train_set.info['label'].values     # ✅ REQUIRED for StratifiedKFold

    cv = StratifiedKFold(
        n_splits=args.fold,
        shuffle=True,
        random_state=42
    )

    for fold, (train_idx, test_idx) in enumerate(cv.split(train_set, labels)):
        print(f"\nFold {fold}")

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=2
        )

        test_loader = DataLoader(
            test_set,
            batch_size=1,
            sampler=SubsetRandomSampler(test_idx),
            num_workers=2
        )

        train(args, train_loader, test_loader, fold)

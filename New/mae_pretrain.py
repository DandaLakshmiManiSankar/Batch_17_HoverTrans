import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_ssl import UltrasoundDataset
from mae_model import MAEHover
from config import config

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = UltrasoundDataset(config.dataset_path, ssl=True)
loader = DataLoader(dataset,
                    batch_size=config.ssl_batch_size,
                    shuffle=True,
                    num_workers=2)

model = MAEHover()

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(config.ssl_pretrain_epochs):
    model.train()
    total_loss = 0

    for imgs in loader:
        imgs = imgs.to(device)

        recon = model(imgs)
        loss = criterion(recon, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"SSL Epoch {epoch+1}/{config.ssl_pretrain_epochs} "
          f"Loss: {total_loss/len(loader)}")

torch.save(model.module.encoder.state_dict()
           if isinstance(model, torch.nn.DataParallel)
           else model.encoder.state_dict(),
           "mae_encoder.pth")

print("SSL Pretraining Finished.")

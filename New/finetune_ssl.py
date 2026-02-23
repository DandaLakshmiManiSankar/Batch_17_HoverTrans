import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_ssl import UltrasoundDataset
from hovertrans import create_model
from config import config

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = UltrasoundDataset(config.dataset_path, ssl=False)
loader = DataLoader(dataset,
                    batch_size=config.finetune_batch_size,
                    shuffle=True,
                    num_workers=2)

model = create_model(
    img_size=config.img_size,
    num_classes=config.class_num,
    patch_size=config.patch_size,
    dim=config.dim,
    depth=config.depth,
    num_heads=config.num_heads,
    num_inner_head=config.num_inner_head
)

# Load SSL weights
state_dict = torch.load("mae_encoder.pth")
model.load_state_dict(state_dict, strict=False)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(config.finetune_epochs):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"FineTune Epoch {epoch+1}/{config.finetune_epochs} "
          f"Loss: {total_loss/len(loader)}")

torch.save(model.module.state_dict()
           if isinstance(model, torch.nn.DataParallel)
           else model.state_dict(),
           "hover_ssl_finetuned.pth")

print("Fine-tuning Finished.")

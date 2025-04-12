import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from lnet import LNet
from data_utils import SegmentationDataset
from laplacian_loss import laplacian_pyramid_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LNet(num_classes=8).to(device)

train_set = SegmentationDataset("data/train_images", "data/train_labels", target_size=(512, 512))
val_set = SegmentationDataset("data/val_images", "data/val_labels", target_size=(512, 512))

train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
val_loader = DataLoader(val_set, batch_size=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def combined_loss(pred, target):
    ce = criterion(pred, target)
    pred_mask = pred.argmax(dim=1).float()
    target = target.float()
    lp = laplacian_pyramid_loss(target, pred_mask)
    return ce + 0.1 * lp

for epoch in range(25):
    model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = combined_loss(output, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), "l_net_best.pth")
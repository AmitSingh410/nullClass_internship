# %%
import torch 
from torch.utils.data import DataLoader,random_split
import torchvision
from utils import (
    CIFARColorizationDataset,
    UNet32,
    rgb_to_gray_with_clahe,
    lab_to_rgb_torch,
    train_model,
    evaluate_model,
    targeted_colorization_with_segmentation,
    sam_targeted_colorization
)

# %%
# Set device

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
# Load Dataset

root_dir="./data"
full_train_ds= CIFARColorizationDataset(root_dir,train=True,augment=True)
train_len = int(0.9 * len(full_train_ds))
val_len = len(full_train_ds) - train_len
train_ds, val_ds = random_split(full_train_ds, [train_len, val_len])

train_loader= DataLoader(train_ds,batch_size=128,shuffle=True,num_workers=0,pin_memory=True)
val_loader=DataLoader(val_ds,batch_size=128,shuffle=False,num_workers=0,pin_memory=True)

test_ds= CIFARColorizationDataset(root_dir,train=False,augment=False)
test_loader=DataLoader(test_ds,batch_size=128,shuffle=False,num_workers=0,pin_memory=True)

# %%
# Initialize Model

model=UNet32(base_ch=64).to(device)
model.load_state_dict(torch.load("best_cifar_unet_clahe_lab.pth", map_location=device))

# %%
evaluate_model(model, val_loader, device)

# %%




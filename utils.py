import os
import time
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset,random_split
from skimage import color
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
import torchvision.models.segmentation as segmentation
from segment_anything import sam_model_registry, SamPredictor
import glob
import kornia as K
from torch.amp import autocast,GradScaler   


# -----------------------------
# Section 1: Utility Functions
# -----------------------------

class GPUTransform:
    def __init__(self, img_size=224):
        self.resize = K.geometry.Resize((img_size, img_size), interpolation='bilinear')
        self.rgb2gray= K.color.RgbToGrayscale()
        self.rgb2lab = K.color.RgbToLab()    # returns L∈[0,100], a/b∼[-110,110]
    def __call__(self, img_cpu):
    # img_cpu: [B,3,H,W] or [3,H,W]
        single = False
        if img_cpu.dim() == 3:
            img_cpu = img_cpu.unsqueeze(0)
            single = True

        img = img_cpu.float()        # [B,3,H,W]
        img = self.resize(img)
        gray = self.rgb2gray(img)                      # [B,1,H,W]
        gray = K.enhance.equalize_clahe(gray, grid_size=(8,8), clip_limit=1.5)
        lab  = self.rgb2lab(img)                       # [B,3,H,W]
        lab_norm = torch.cat([lab[:,0:1]/100.0, lab[:,1:]/110.0], dim=1)

        return img, gray, lab_norm if not single else lab_norm.squeeze(0)

_gpu_tfm = GPUTransform(img_size=224)




def rgb_to_lab(tensor):
    """Convert normalized tensor image [3,H,W] in range [0,1] to Lab."""
    img = tensor.permute(1, 2, 0).cpu().numpy()  # [H,W,3]
    lab = color.rgb2lab(img).astype(np.float32)
    lab = torch.from_numpy(lab).permute(2, 0, 1) / torch.tensor([100.0, 128.0, 128.0]).view(3, 1, 1)
    return lab

def srgb_to_linear(img_srgb:torch.Tensor) -> torch.Tensor:
    
    #Convert sRGB-encoded tensor [B,3,H,W] in [0,1] -> linear RGB [B,3,H,W]
    
    mask = (img_srgb <= 0.04045).to(img_srgb.dtype)
    c_lin_low= img_srgb / 12.92
    c_lin_high = ((img_srgb + 0.055) / 1.055) ** 2.4
    return mask * c_lin_low + (1.0 - mask) * c_lin_high

def lab_to_rgb_torch(lab_tensor):
    
    single_image = False
    if lab_tensor.dim() == 3:
        lab_tensor = lab_tensor.unsqueeze(0)
        single_image = True

    lab_tensor = lab_tensor.clone().detach().cpu()

    L = lab_tensor[:, 0:1, :, :] * 100.0          # [0,100]
    a = lab_tensor[:, 1:2, :, :] * 128.0          # [-128,127]
    b = lab_tensor[:, 2:3, :, :] * 128.0

    lab = torch.cat([L, a, b], dim=1)             # [B,3,H,W]
    B, C, H, W = lab.shape

    rgb_imgs = []
    for i in range(B):
        lab_img = lab[i].permute(1, 2, 0).numpy()             # [H,W,3]
        
         # 🚨 SAFETY CLIPPING
        lab_img[:, :, 0] = np.clip(lab_img[:, :, 0], 0, 100)     # L ∈ [0,100]
        lab_img[:, :, 1] = np.clip(lab_img[:, :, 1], -128, 127)  # a ∈ [-128,127]
        lab_img[:, :, 2] = np.clip(lab_img[:, :, 2], -128, 127)  # b ∈ [-128,127]
        
        rgb_img = color.lab2rgb(lab_img.astype(np.float64))   # [H,W,3] in [0,1]
        rgb_imgs.append(torch.from_numpy(rgb_img).permute(2, 0, 1))  # [3,H,W]

    rgb_tensor = torch.stack(rgb_imgs)  # [B,3,H,W]

    return rgb_tensor.squeeze(0) if single_image else rgb_tensor

def torch_rgb_to_hsv(rgb):
  r,g,b=rgb[:,0,:,:],rgb[:,1,:,:],rgb[:,2,:,:]
  max_val, _=torch.max(rgb,dim=1)
  min_val,_=torch.min(rgb,dim=1)
  diff=max_val-min_val

  h = torch.zeros_like(r)
  mask=(max_val==r)&(g>=b)
  h[mask]=(g[mask]-b[mask])/diff[mask]
  mask=(max_val==r)&(g<b)
  h[mask]=(g[mask]-b[mask])/diff[mask] + 6.0
  mask=max_val==g
  h[mask]=(b[mask]-r[mask])/diff[mask] + 2.0
  mask=max_val==b
  h[mask]=(r[mask]-g[mask])/diff[mask] + 4.0
  h=h/6.0
  h[diff==0.0]=0.0

  s=torch.zeros_like(r)
  s[diff!=0.0]=diff[diff!=0.0]/max_val[diff!=0.0]

  v=max_val

  return torch.stack([h,s,v],dim=1)

def torch_hsv_to_rgb(hsv):
  h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
  i = (h * 6.0).floor()
  f = h * 6.0 - i
  p = v * (1.0 - s)
  q = v * (1.0 - s * f)
  t = v * (1.0 - s * (1.0 - f))

  i_mod = i % 6
  r = torch.zeros_like(h)
  g = torch.zeros_like(h)
  b = torch.zeros_like(h)

  r[i_mod == 0.0] = v[i_mod == 0.0]
  g[i_mod == 0.0] = t[i_mod == 0.0]
  b[i_mod == 0.0] = p[i_mod == 0.0]

  r[i_mod == 1.0] = q[i_mod == 1.0]
  g[i_mod == 1.0] = v[i_mod == 1.0]
  b[i_mod == 1.0] = p[i_mod == 1.0]

  r[i_mod == 2.0] = p[i_mod == 2.0]
  g[i_mod == 2.0] = v[i_mod == 2.0]
  b[i_mod == 2.0] = t[i_mod == 2.0]

  r[i_mod == 3.0] = p[i_mod == 3.0]
  g[i_mod == 3.0] = q[i_mod == 3.0]
  b[i_mod == 3.0] = v[i_mod == 3.0]

  r[i_mod == 4.0] = t[i_mod == 4.0]
  g[i_mod == 4.0] = p[i_mod == 4.0]
  b[i_mod == 4.0] = v[i_mod == 4.0]

  r[i_mod == 5.0] = v[i_mod == 5.0]
  g[i_mod == 5.0] = p[i_mod == 5.0]
  b[i_mod == 5.0] = q[i_mod == 5.0]

  return torch.stack([r, g, b], dim=1)

def exaggerate_colors(images,saturation_factor=1.5,value_factor=1.05):
  images=images.clone()

  images_hsv=torch_rgb_to_hsv(images)

  images_hsv[:,1,:,:]=torch.clamp(images_hsv[:,1,:,:]*saturation_factor,0,1)
  images_hsv[:,2,:,:]=torch.clamp(images_hsv[:,2,:,:]*value_factor,0,1)

  color_exaggerated_images=torch_hsv_to_rgb(images_hsv)

  return color_exaggerated_images

def imshow(img):
  #Convert to NumPy
  if torch.is_tensor(img):
    npimg=img.cpu().numpy()
  else:
    npimg=img
  # Handle Channel Ordering
  if npimg.ndim == 3 and npimg.shape[0] in [1, 3]:
    npimg=np.transpose(npimg, (1, 2, 0))
  plt.imshow(npimg.squeeze(),cmap='gray' if npimg.ndim==2 or npimg.shape[-1]==1 else None)
  plt.axis('off')
    

def visualize_all_three(original_images,grayscale_images,colorized_images,n=5):

  fig=plt.figure(figsize=(3*n,4))
  for i in range(n):
    ax=plt.subplot(1,3*n,3*i+1)
    imshow(original_images[i])
    ax.set_title("Original")
    ax.axis("off")

    ax=plt.subplot(1,3*n,3*i + 2)
    gray_img= grayscale_images[i]
    if gray_img.ndim == 3 and gray_img.shape[0] == 1:
      gray_img=gray_img[0]
    imshow(gray_img)
    ax.set_title("Grayscale (CLAHE)")
    ax.axis("off")

    ax=plt.subplot(1,3*n,3*i + 3)
    imshow(colorized_images[i])
    ax.set_title("Colorized")
    ax.axis("off")

  plt.tight_layout()
  plt.show()
  
# -----------------------------
# SECTION 2: Dataset & Model
# -----------------------------

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import glob

class CustomImageNet100Dataset(Dataset):
    def __init__(self, root_dir, train=True, augment=True, img_size=224):
        self.img_size = img_size
        self.augment = augment

        # Shard folders
        if train:
            shard_dirs = [os.path.join(root_dir, f"train.X{i}") for i in range(1, 5)]
        else:
            shard_dirs = [os.path.join(root_dir, "val.X")]

        self.samples = []
        for shard in shard_dirs:
            class_dirs = glob.glob(os.path.join(shard, "*"))
            for class_dir in class_dirs:
                label = os.path.basename(class_dir)
                images = glob.glob(os.path.join(class_dir, "*.JPEG"))
                self.samples.extend([(img, label) for img in images])

        self.class_to_idx = {label: idx for idx, label in enumerate(sorted(set(l for _, l in self.samples)))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]
        img = Image.open(img_path).convert("RGB")  # 🔥 ensures always 3-channel
        img = transforms.ToTensor()(img)           # [3, H, W] in [0,1]
  # [C, H, W], C could be 1 or 3
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)  # Convert grayscale to 3 channels
        img = img.float() / 255.0
        img = K.geometry.Resize((self.img_size, self.img_size), interpolation='bilinear')(img.unsqueeze(0)).squeeze(0)
        if self.augment:
            img = K.augmentation.RandomHorizontalFlip()(img.unsqueeze(0)).squeeze(0)
        
        return img


        
class UNet32(nn.Module):
    def __init__(self,base_ch=64):
        super(UNet32, self).__init__()
        
        #Encoder Stage 1: [1-64]
        self.enc1=nn.Sequential(
            nn.Conv2d(1, base_ch,   kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        self.pool1=nn.MaxPool2d(2)
        
        #Encoder Stage 2: [64-128]
        self.enc2=nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, kernel_size=3,padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*2, base_ch*2, kernel_size=3,padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True)
        )
        self.pool2=nn.MaxPool2d(2)
        
        #Encoder Stage 3: [128-256]
        self.enc3=nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, kernel_size=3,padding=1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*4, base_ch*4, kernel_size=3,padding=1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True)
        )
        self.pool3=nn.MaxPool2d(2)
        
        #Bottleneck: [256-512]
        self.bottleneck=nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*8, kernel_size=3,padding=1),
            nn.BatchNorm2d(base_ch*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*8, base_ch*8, kernel_size=3,padding=1),
            nn.BatchNorm2d(base_ch*8),
            nn.ReLU(inplace=True)
        )
        
        #Decoder Stage 3: [512-256]
        self.up3=nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3=nn.Sequential(
            nn.Conv2d(base_ch*8, base_ch*4, kernel_size=3,padding=1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*4, base_ch*4, kernel_size=3,padding=1),
            nn.BatchNorm2d(base_ch*4),
            nn.ReLU(inplace=True)
        )
        
        #Decoder Stage 2: [256-128]
        self.up2=nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2=nn.Sequential(
            nn.Conv2d(base_ch*4, base_ch*2, kernel_size=3,padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch*2, base_ch*2, kernel_size=3,padding=1),
            nn.BatchNorm2d(base_ch*2),
            nn.ReLU(inplace=True)
        )
        
        #Decoder Stage 1: [128-64]
        self.up1=nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1=nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, kernel_size=3,padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3,padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )
        
        #Final 1x1 conv -> 2 channels (a,b), then tanh to [-1,1]
        self.final=nn.Conv2d(base_ch, 2, kernel_size=1)
        self.tanh=nn.Tanh()
        
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        b = self.bottleneck(p3)
        
        u3 = self.up3(b)
        cat3 = torch.cat((u3, e3), dim=1)
        d3 = self.dec3(cat3)
        
        u2 = self.up2(d3)
        cat2 = torch.cat((u2, e2), dim=1)
        d2 = self.dec2(cat2)
        
        u1 = self.up1(d2)
        cat1 = torch.cat((u1, e1), dim=1)
        d1 = self.dec1(cat1)
        
        out=self.tanh(self.final(d1))
        return out

# --------------------------------
# SECTION 3: Segmentation Helpers
# --------------------------------

def targeted_colorization_with_segmentation(pil_img_path, model, gray_fn, selected_classes, output_prefix="output"):
    device = next(model.parameters()).device

    # Load image
    pil_img = Image.open(pil_img_path).convert('RGB')

    # Resize + tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Grayscale
    _, gray_tensor, _ = _gpu_tfm(img_tensor.cpu())


    # Colorization
    model.eval()
    with torch.no_grad():
        pred_ab = model(gray_tensor)
        lab = torch.cat([gray_tensor, torch.clamp(pred_ab, -1.0, 1.0)], dim=1)
        rgb_pred = lab_to_rgb_torch(lab).to(device)

    # Segmentation
    seg_model = segmentation.deeplabv3_resnet50(pretrained=True).eval().to(device)
    seg_transform = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    seg_input = seg_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        seg_output = seg_model(seg_input)['out']
    seg_mask = torch.argmax(seg_output.squeeze(), dim=0).cpu().numpy()

    seg_mask_resized = cv2.resize(seg_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    seg_tensor = torch.from_numpy(seg_mask_resized).unsqueeze(0).unsqueeze(0).to(device)
    mask_tensor = torch.zeros_like(seg_tensor, dtype=torch.bool).to(device)
    for cls in selected_classes:
        mask_tensor |= (seg_tensor == cls)

    gray_3ch = gray_tensor.repeat(1, 3, 1, 1)
    final_tensor = torch.where(mask_tensor, rgb_pred, gray_3ch).to(device)

    # Save colorized output
    to_pil = transforms.ToPILImage()
    to_pil(final_tensor.squeeze(0).cpu().clamp(0, 1)).save(f"{output_prefix}_targeted_colorized.jpeg")

    # Save segmentation mask
    seg_color = cv2.applyColorMap((seg_mask_resized * 10).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{output_prefix}_segmentation_mask.jpeg", seg_color)

    print(f"✅ Saved: {output_prefix}_targeted_colorized.jpeg")
    print(f"✅ Saved: {output_prefix}_segmentation_mask.jpeg")



# -----------------------------
# SECTION 4: Training Loop
# -----------------------------

import time  # make sure this is at the top of your file

import warnings
import time
from tqdm import tqdm

import warnings
import time
from tqdm import tqdm

def train_model(model, train_loader, val_loader, device, save_path="checkpoint_imagenet_unet_clahe_lab.pth", epochs=75, patience=15):
    print("Using", device)

    criteria = nn.L1Loss()
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device)
    for p in vgg.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.5)
    scaler = GradScaler("cuda")

    start_epoch = 1
    epochs_since_improve = 0
    epoch_start_time = time.time()

    # Resume logic
    if os.path.exists(save_path):
        print(f"🔄 Resuming from checkpoint: {save_path}")
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val = checkpoint['best_val']
    else:
        best_val = float('inf')

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_train = 0.0
        start_time = time.time()

        print(f"\n🔁 Epoch {epoch}/{epochs} [Training]")
        for i, batch in enumerate(tqdm(train_loader, desc="Loading Batches", leave=True)):
            try:
                batch = batch.to(device,non_blocking=True)
                _, gray, lab = _gpu_tfm(batch)   # All on GPU
                L_gt  = lab[:, 0:1]              # [B,1,H,W]
                ab_gt = lab[:, 1:]               # [B,2,H,W]

                with autocast("cuda"):
                    ab_pred = model(gray)
                    ab_pred = torch.clamp(ab_pred, -1.0, 1.0)
                    loss_ab = criteria(ab_pred, ab_gt)

                    lab_pred = torch.cat([gray * 100.0, ab_pred * 128.0], dim=1)
                    lab_gt   = torch.cat([L_gt * 100.0, ab_gt * 128.0], dim=1)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rgb_pred = lab_to_rgb_torch(lab_pred).to(device)
                        rgb_gt   = lab_to_rgb_torch(lab_gt).to(device)

                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

                    rgb_pred_small = F.interpolate(rgb_pred, size=(112, 112), mode="bilinear", align_corners=False)
                    rgb_gt_small   = F.interpolate(rgb_gt,   size=(112, 112), mode="bilinear", align_corners=False)

                    pred_norm = (rgb_pred_small - mean) / std
                    gt_norm   = (rgb_gt_small - mean) / std

                    feat_p = vgg(pred_norm.float())
                    feat_g = vgg(gt_norm.float())
                    loss_perc = F.l1_loss(feat_p, feat_g)

                    loss = loss_ab + 0.01 * loss_perc

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_train += loss.item()

            except Exception as e:
                print(f"❌ Error in batch {i}: {e}")
                continue

        train_avg = running_train / len(train_loader)

        # ------------------- VALIDATION -------------------
        model.eval()
        running_val = 0.0
        print(f"\n🔍 Epoch {epoch}/{epochs} [Validation]")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc="Validating", leave=False)):
                try:
                    _, gray_v, lab_v = _gpu_tfm(batch)
                    ab_gt_v = lab_v[:, 1:]
                    ab_pred_v = model(gray_v)
                    running_val += criteria(ab_pred_v, ab_gt_v).item()
                except Exception as e:
                    print(f"❌ Error in validation batch {i}: {e}")
                    continue

        val_avg = running_val / len(val_loader)

        # ------------------- ETA -------------------
        elapsed_epoch = time.time() - start_time
        total_elapsed = time.time() - epoch_start_time
        avg_epoch_time = total_elapsed / (epoch - start_epoch + 1)
        eta = avg_epoch_time * (epochs - epoch)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))

        # ------------------- LOG -------------------
        print(f"\n📊 Epoch {epoch:>2d}/{epochs} | Train L1(ab): {train_avg:.4f} | Val L1(ab): {val_avg:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | ETA: {eta_str}")

        if val_avg < best_val:
            best_val = val_avg
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'best_val': best_val
            }, save_path)
            print(f"✅ Saved best model to: {save_path}")
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            print(f"⚠️ No improvement for {epochs_since_improve} epoch(s).")
            if epochs_since_improve >= patience:
                print(f"⏹️ Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

        scheduler.step()







# -----------------------------
# SECTION 5: Model Inference
# -----------------------------

def infer_and_save(model, image_path, device):
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    pil_img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    _, gray_tensor, _ = _gpu_tfm(img_tensor.cpu())
    model.eval()
    with torch.no_grad():
        pred_ab = model(gray_tensor)
        lab_norm = torch.cat([gray_tensor, torch.clamp(pred_ab, -1.0, 1.0)], dim=1)
        rgb_pred = lab_to_rgb_torch(lab_norm)
    gray_pil = to_pil(gray_tensor.squeeze(0).cpu())
    gray_pil.save('eiffeltower_gray_updated.jpeg')
    colorized_pil = to_pil(rgb_pred.squeeze(0).cpu())
    colorized_pil.save('colorized_eiffeltower_updated.jpeg')
    print("✅ Saved CLAHE gray + properly reconstructed colorized images")

# -----------------------------
# SECTION 6: Evaluate on Test Set
# -----------------------------

def evaluate_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            grayscale_images = batch["gray"].to(device)
            pred_ab = model(grayscale_images)
            L_gt = batch["L_gt"].to(device)
            pred_lab = torch.cat([L_gt, pred_ab], dim=1).to(device)
            gt_lab = torch.cat([L_gt, batch["ab_gt"].to(device)], dim=1).to(device)
            colorized_images_rgb = lab_to_rgb_torch(pred_lab)
            original_images_rgb = lab_to_rgb_torch(gt_lab)
            colorized_images_rgb = exaggerate_colors(colorized_images_rgb)
            grayscale_images_cpu = grayscale_images.cpu().squeeze(1)
            visualize_all_three(
                original_images=original_images_rgb,
                grayscale_images=grayscale_images_cpu,
                colorized_images=colorized_images_rgb,
                n=5
            )
            if i == 10:
                break

def sam_targeted_colorization(
        image_path,
        model,
        point_coords,
        sam_checkpoint_path="segment-anything/checkpoints/sam_vit_h_4b8939.pth",
        model_type="vit_h",
        output_prefix="sam_output"
):
    
    device = next(model.parameters()).device

    # Load and preprocess input image
    image_bgr = cv2.imread(image_path)
    image_rgb= cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img=Image.fromarray(image_rgb)
    transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    img_tensor=transform(pil_img).unsqueeze(0).to(device)

    # Convert to grayscale
    _, gray_tensor, _ = _gpu_tfm(img_tensor.cpu())

    # Predict ab channels
    model.eval()
    with torch.no_grad():
        pred_ab = model(gray_tensor)
        lab_tensor= torch.cat([gray_tensor, torch.clamp(pred_ab, -1.0, 1.0)], dim=1)
        color_tensor = lab_to_rgb_torch(lab_tensor).to(device)

    # Load SAM and get mask 
    sam=sam_model_registry[model_type](checkpoint=sam_checkpoint_path).to(device)
    predictor=SamPredictor(sam)
    predictor.set_image(image_rgb)

    input_point = np.array([point_coords])
    input_label = np.array([1])  # 1 for foreground

    masks,scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )
    selected_mask=masks[np.argmax(scores)]

    # Resize mask to match model input
    mask_tensor=torch.from_numpy(selected_mask).unsqueeze(0).unsqueeze(0).to(device)
    if mask_tensor.shape[-2:]!=gray_tensor.shape[-2:]:
        mask_tensor=F.interpolate(mask_tensor.float(),size=gray_tensor.shape[-2:],mode='nearest')

    # Blend
    gray_3ch=gray_tensor.repeat(1, 3, 1, 1)
    final_tensor=torch.where(mask_tensor.bool(),color_tensor,gray_3ch)

    # Save output 
    to_pil= transforms.ToPILImage()
    colorized_pil = to_pil(final_tensor.squeeze(0).cpu().clamp(0, 1))
    output_path = f"{output_prefix}_sam_colorized.jpeg"
    colorized_pil.save(output_path)

    # Save mask overlay for reference
    mask_visual=cv2.applyColorMap((selected_mask*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay_path=f"{output_prefix}_sam_mask.jpeg"
    cv2.imwrite(overlay_path, mask_visual)

    print(f"✅ Saved SAM colorized image: {output_path}")
    print(f"✅ Saved SAM mask overlay: {overlay_path}")

    return output_path 
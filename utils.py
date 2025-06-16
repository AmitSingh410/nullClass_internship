import os
import time
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset,random_split
from skimage import color
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
from torchvision.models.segmentation import segmentation

# -----------------------------
# Section 1: Utility Functions
# -----------------------------

def srgb_to_linear(img_srgb:torch.Tensor) -> torch.Tensor:
    
    #Convert sRGB-encoded tensor [B,3,H,W] in [0,1] -> linear RGB [B,3,H,W]
    
    mask = (img_srgb <= 0.04045).to(img_srgb.dtype)
    c_lin_low= img_srgb / 12.92
    c_lin_high = ((img_srgb + 0.055) / 1.055) ** 2.4
    return mask * c_lin_low + (1.0 - mask) * c_lin_high

def rgb_to_gray_with_clahe(img:torch.Tensor, clip_limit:float=1.5, tile_grid_size=(2, 2)) -> torch.Tensor:
    
   
    img_lin= srgb_to_linear(img) #sRGB to linear RGB conversion
    weights=torch.tensor([0.299, 0.587, 0.114], device=img_lin.device, dtype=img_lin.dtype).view(1, 3, 1, 1)
    y = (img_lin * weights).sum(dim=1, keepdim=True)  # [B,1,H,W]
    
    out=[]
    for b in range(y.size(0)):
        y_np=(y[b,0].cpu().numpy()*255).astype(np.uint8)  # Convert to [0,255] for OpenCV
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        y_eq = clahe.apply(y_np)
        y_eq_t=torch.from_numpy(y_eq.astype(np.float32) / 255.0)
        out.append(y_eq_t.unsqueeze(0))
    out=torch.stack(out, dim=0)
    return out.to(img.device)

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

class CIFARColorizationDataset(Dataset):
    def __init__(self,root_dir,train=True,augment=True):
        self.augment=augment
        
        self.base_dataset=torchvision.datasets.CIFAR10(
            root=root_dir,
            train=train,
            download=False,
            transform=None
        )
        
        self.aug_transform=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15,interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2)
        ])
        self.to_Tensor=transforms.ToTensor()
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # 1) Get the PIL image out of the torchvision CIFAR10 dataset
        pil_img, _ = self.base_dataset[idx]  # pil_img is 32×32 RGB (PIL Image)
        if self.augment:
            pil_img = self.aug_transform(pil_img)

        # 2) Convert to Tensor [3×32×32] in [0,1]
        img_rgb = self.to_Tensor(pil_img)

        # 3) Compute weighted+CLAHE grayscale via gray_utils
        gray = rgb_to_gray_with_clahe(img_rgb.unsqueeze(0)).squeeze(0)  # [1,32,32]

        # 4) Convert GT PIL→NumPy→Lab and normalize
        img_np = np.array(pil_img) / 255.0  # [32,32,3] ∈ [0,1]
        lab = color.rgb2lab(img_np.astype(np.float32))  # L∈[0,100], a,b∈[-128,127]
        L = lab[..., 0:1] / 100.0        # [32,32,1] ∈ [0,1]
        ab = lab[..., 1:3] / 128.0       # [32,32,2] ∈ [-1,1]

        L_t  = torch.from_numpy(L.astype(np.float32)).permute(2,0,1)   # [1,32,32]
        ab_t = torch.from_numpy(ab.astype(np.float32)).permute(2,0,1)  # [2,32,32]

        return {
            "gray":  gray,   # [1,32,32]
            "L_gt":  L_t,    # [1,32,32]
            "ab_gt": ab_t    # [2,32,32]
        }
        
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
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Grayscale
    gray_tensor = gray_fn(img_tensor)

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

    seg_mask_resized = cv2.resize(seg_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
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

def train_model(model, train_loader, val_loader, device, save_path="best_cifar_unet_clahe_lab.pth", epochs=75, patience=15):
    print("Using", device)
    criteria = nn.L1Loss()
    vgg = models.vgg16(pretrained=True).features[:16].to(device)
    for p in vgg.parameters():
        p.requires_grad = False
    up = nn.Upsample(scale_factor=3, mode="bilinear", align_corners=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.5)

    best_val = float('inf')
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_train = 0.0

        for batch in train_loader:
            gray = batch['gray'].to(device)
            L_gt = batch['L_gt'].to(device)
            ab_gt = batch['ab_gt'].to(device)
            ab_pred = model(gray)
            loss_ab = criteria(ab_pred, ab_gt)

            L_pred = gray * 100.0
            ab_pred_s = ab_pred * 128.0
            lab_pred = torch.cat([L_pred, ab_pred_s], dim=1)
            rgb_pred32 = lab_to_rgb_torch(lab_pred)
            rgb_pred96 = up(rgb_pred32)

            L_gt_full = L_gt * 100.0
            ab_gt_s = ab_gt * 128.0
            lab_gt = torch.cat([L_gt_full, ab_gt_s], dim=1)
            rgb_gt32 = lab_to_rgb_torch(lab_gt)
            rgb_gt96 = up(rgb_gt32)

            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            rgb_pred96 = rgb_pred96.to(device)
            rgb_gt96 = rgb_gt96.to(device)

            pred_norm = (rgb_pred96 - mean) / std
            gt_norm = (rgb_gt96 - mean) / std
            feat_p = vgg(pred_norm.float())
            feat_g = vgg(gt_norm.float())
            loss_perc = F.l1_loss(feat_p, feat_g)

            loss = loss_ab + 0.01 * loss_perc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train += loss.item()

        train_avg = running_train / len(train_loader)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for batch in val_loader:
                gray_v = batch["gray"].to(device)
                ab_gt_v = batch["ab_gt"].to(device)
                ab_pred_v = model(gray_v)
                running_val += criteria(ab_pred_v, ab_gt_v).item()
        val_avg = running_val / len(val_loader)

        print(f"Epoch {epoch:>2d}/{epochs}  |  Train L1(ab): {train_avg:.4f}  |  Val L1(ab): {val_avg:.4f}  |  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_avg < best_val:
            best_val = val_avg
            torch.save(model.state_dict(), save_path)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
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
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    gray_tensor = rgb_to_gray_with_clahe(img_tensor)
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

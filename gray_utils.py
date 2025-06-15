import torch 
import numpy as np
import cv2

def srgb_to_linear(img_srgb:torch.Tensor) -> torch.Tensor:
    
    #Convert sRGB-encoded tensor [B,3,H,W] in [0,1] -> linear RGB [B,3,H,W]
    
    mask = (img_srgb <= 0.04045).to(img_srgb.dtype)
    c_lin_low= img_srgb / 12.92
    c_lin_high = ((img_srgb + 0.055) / 1.055) ** 2.4
    return mask * c_lin_low + (1.0 - mask) * c_lin_high

def rgb_to_gray_with_clahe(img:torch.Tensor, clip_limit:float=1.5, tile_grid_size=(2, 2)) -> torch.Tensor:
    
    """
    Convert sRGB [B,3,H,W] → grayscale [B,1,H,W] by:
      1) sRGB → linear-RGB
      2) Weighted sum Y = 0.299R_lin + 0.587G_lin + 0.114B_lin
      3) CLAHE on each [H,W]
    Returns Y ∈ [0,1].
    """
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
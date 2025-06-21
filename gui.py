import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from utils import UNet32, rgb_to_gray_with_clahe, lab_to_rgb_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet32(base_ch=64).to(device)
model.load_state_dict(torch.load("best_cifar_unet_clahe_lab.pth", map_location=device))
model.eval()

def colorize_from_editor(editor_data):
    if not editor_data or "background" not in editor_data or "layers" not in editor_data:
        raise gr.Error("Invalid input from editor.")

    original_img = editor_data["background"]
    mask_img = editor_data["layers"][0] if editor_data["layers"] else None

    if original_img is None or mask_img is None:
        raise gr.Error("Draw a white mask on the image before submitting.")

    original_img = original_img.resize((256, 256)).convert("RGB")
    mask_img = mask_img.resize((256, 256)).convert("L")

    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(original_img).unsqueeze(0).to(device)
    gray_tensor = rgb_to_gray_with_clahe(img_tensor)

    with torch.no_grad():
        pred_ab = model(gray_tensor)
        lab_tensor = torch.cat([gray_tensor, pred_ab.clamp(-1, 1)], dim=1)
        color_tensor = lab_to_rgb_torch(lab_tensor).to(device)

    mask_tensor = to_tensor(mask_img).unsqueeze(0).to(device) > 0.5
    gray_3ch = gray_tensor.repeat(1, 3, 1, 1)
    result_tensor = torch.where(mask_tensor, color_tensor, gray_3ch)

    to_pil = transforms.ToPILImage()
    return original_img, mask_img, to_pil(result_tensor.squeeze(0).cpu().clamp(0, 1))

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## ✅ Upload image and draw a white mask for colorization")

    editor = gr.ImageEditor(type="pil", label="Upload + Draw")

    btn = gr.Button("Colorize Drawn Region")

    out1 = gr.Image(label="Original")
    out2 = gr.Image(label="Drawn Mask")
    out3 = gr.Image(label="Colorized Output")

    btn.click(fn=colorize_from_editor, inputs=[editor], outputs=[out1, out2, out3])

if __name__ == "__main__":
    demo.launch()

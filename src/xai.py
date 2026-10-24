"""
Shared explainability module for Phase 2.
Provides per-architecture Grad-CAM target layers and reshape transforms.
One model loaded at a time (8 GB VRAM constraint).
"""
import torch
import numpy as np
from train import build_model
from data import IMAGENET_MEAN, IMAGENET_STD
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def load_model(name: str, checkpoint_path: str, device) -> torch.nn.Module:
    """Build model, load checkpoint, set eval. Returns model on device."""
    model = build_model(name).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Reshape transforms: token sequence -> 2D spatial map ──
def vit_reshape_transform(tensor, height=24, width=24):
    """
    ViT-Base/16 at 384px: 576 patches + 1 CLS = 577 tokens.
    Drop CLS (index 0), reshape 576 -> 24x24, move channels to dim 1.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def tinyvit_reshape_transform(tensor, height=12, width=12):
    """
    TinyViT-21M final stage at 384px: 144 tokens (no CLS), reshape 144 -> 12x12.
    """
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def get_cam_config(model, name: str):
    """
    Returns (target_layers, reshape_transform) for the given model.
    reshape_transform is None for the pure CNN.
    """
    if name == "resnet50":
        return [model.layer4[-1]], None
    elif name == "vit_base_patch16_384":
        return [model.blocks[-1].norm1], vit_reshape_transform
    elif name == "tiny_vit_21m_384":
        return [model.stages[-1].blocks[-1].local_conv], None
    else:
        raise ValueError(f"No CAM config for {name}")


# ImageNet stats - single source of truth from data.py
_MEAN = np.array(IMAGENET_MEAN)
_STD  = np.array(IMAGENET_STD)


def tensor_to_rgb(img_tensor):
    """
    Convert a normalized CHW tensor back to an un-normalized HWC image in [0,1].
    pytorch-grad-cam overlays the heatmap on this RGB image.
    """
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)  
    img = (img * _STD) + _MEAN                         
    return np.clip(img, 0, 1)                          


# Maps method name string -> CAM class, so callers just pass "gradcam"/"gradcam++"
CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
}


def run_cam(model, name, input_tensor, method="gradcam", target_class=None):
    """
    Run one CAM method on one image for one model.
    """
    target_layers, reshape = get_cam_config(model, name)

    if target_class is None:
        with torch.no_grad():
            target_class = model(input_tensor).argmax(dim=1).item()

    cam_class = CAM_METHODS[method]
    cam = cam_class(model=model, target_layers=target_layers, reshape_transform=reshape)

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=[ClassifierOutputTarget(target_class)])
    return grayscale_cam[0], target_class   # drop batch dim

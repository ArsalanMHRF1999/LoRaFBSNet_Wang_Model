import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from spikingjelly.activation_based import neuron
from src.model.lorafb_snet import lorafb_snet18, ConvRecurrentContainer
from src.utils.functional import set_backend, set_step_mode

def create_model_with_feedback(weights_path, num_classes=101, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Create a LoRA-FB-SNet model with feedback connections and load pretrained weights.

    Args:
        num_classes (int): Number of output classes for the model.
        pretrained_weights_path (str or Path): Path to the pretrained weights file.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: The model with feedback connections and loaded weights.
    """

    # ==========================================
    # 1. Initialize model
    # ==========================================
    model = lorafb_snet18(num_classes=num_classes, cnf="ADD", feedback_connection=[(4, 3)])
    set_backend(model, 'cupy', neuron.BaseNode, (ConvRecurrentContainer,))
    set_step_mode(model, 'm',(ConvRecurrentContainer,))

    # ==========================================
    # 2. Load pretrained weights
    # ==========================================
    
    print(f"Loading weights from {weights_path}...", flush=True)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    
    else:
        state_dict = checkpoint
    
    # Remove DataParallel prefix
    state_dict = {
        k.replace("module.", ""): v
        for k, v in state_dict.items()
    }
    
    # Remap recurrent layer names
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if ".sub_" in k:
            parts = k.split(".")
            for idx, part in enumerate(parts):
                if part.startswith("sub_") and part[4:].isdigit():
                    sub_idx = part.split("_")[1]
                    parts[idx] = "sub_module"
                    parts.insert(idx + 1, sub_idx)
                    break
            k = ".".join(parts)
        new_state_dict[k] = v

    # Load pretrained backbone weights
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict,strict=False)

    # ==========================================
    # 3. Feedbach connection weights detection and initialization
    # ==========================================
    unexpected_backbone_keys = [
        k for k in missing_keys
        if not k.startswith("feedback_modules.")
    ]
    if unexpected_backbone_keys:
        raise RuntimeError(
            f"Unexpected missing pretrained parameters: "
            f"{unexpected_backbone_keys}"
        )

    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys}")

    # ==========================================
    # 4. Move model to device
    # ==========================================
    model.to(device)
    print("Pretrained weights loaded successfully.")
    print("Missing feedback parameters:", len(missing_keys))
    
    return model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from spikingjelly.activation_based import neuron
from src.model.lorafb_snet import lorafb_snet18, ConvRecurrentContainer
from src.utils.functional import set_backend, set_step_mode

def create_model(weights_path, num_classes, device):
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
    model = lorafb_snet18(num_classes=num_classes, cnf="ADD")
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



def create_model_with_feedback(weights_path, feedback_connection, num_classes, device):
    """
    Initialize LoRaFB-SNet with external feedback connections
    and load compatible pretrained backbone weights.

    Newly introduced feedback parameters remain initialized
    by the current model.
    """

    # ==========================================
    # 1. Initialize model
    # ==========================================

    model = lorafb_snet18(num_classes=num_classes, cnf="ADD", feedback_connection=feedback_connection)
    set_backend(model, 'cupy', neuron.BaseNode, (ConvRecurrentContainer,))
    set_step_mode(model, 's', (ConvRecurrentContainer,))

    # ==========================================
    # 2. Load checkpoint
    # ==========================================

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    print(f"Loading weights from {weights_path}...", flush=True)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # ==========================================
    # 3. Remove DataParallel prefix
    # ==========================================

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    # ==========================================
    # 4. Remap old recurrent layer names
    # ==========================================

    new_state_dict = {}
    for key, value in cleaned_state_dict.items():
        parts = key.split(".")
        for idx, part in enumerate(parts):
            if part.startswith("sub_") and part[4:].isdigit():
                sub_idx = part[4:]
                parts[idx] = "sub_module"
                parts.insert(idx + 1, sub_idx)
                break
        new_key = ".".join(parts)
        new_state_dict[new_key] = value

    # ==========================================
    # 5. Compare checkpoint and current model
    # ==========================================

    model_state = model.state_dict()
    compatible_state_dict = {}
    unexpected_checkpoint_keys = []
    shape_mismatches = []
    for key, value in new_state_dict.items():
        if key not in model_state:
            unexpected_checkpoint_keys.append(key)
            continue
        if value.shape != model_state[key].shape:
            shape_mismatches.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            # Different classifier dimensions are allowed.
            if key in ["fc.weight", "fc.bias"]:
                continue
            continue
        compatible_state_dict[key] = value

    # ==========================================
    # 6. Load compatible weights
    # ==========================================

    missing_keys, unexpected_keys = model.load_state_dict(compatible_state_dict, strict=False)

    # ==========================================
    # 7. Identify expected feedback parameters
    # ==========================================

    def is_feedback_parameter(key):
        return (
            key.startswith("feedback_modules.")
            or ".feedback_pwconv." in key
            or ".feedback_bn." in key
        )

    allowed_missing_keys = [
        key for key in missing_keys
        if is_feedback_parameter(key)
        or key in ["fc.weight", "fc.bias"]
    ]

    unexpected_missing_keys = [
        key for key in missing_keys
        if key not in allowed_missing_keys
    ]

    # ==========================================
    # 8. Validate loading
    # ==========================================

    if unexpected_checkpoint_keys:
        raise RuntimeError(
            "Unexpected checkpoint keys:\n"
            + "\n".join(unexpected_checkpoint_keys)
        )
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys after loading: {unexpected_keys}")
    if unexpected_missing_keys:
        raise RuntimeError(
            "Missing pretrained backbone parameters:\n"
            + "\n".join(unexpected_missing_keys)
        )
    invalid_shape_mismatches = [
        item for item in shape_mismatches
        if item[0] not in ["fc.weight", "fc.bias"]
    ]
    if invalid_shape_mismatches:
        raise RuntimeError(
            "Backbone parameter shape mismatches:\n"
            + "\n".join(
                f"{key}: checkpoint={old_shape}, model={new_shape}"
                for key, old_shape, new_shape
                in invalid_shape_mismatches
            )
        )

    # ==========================================
    # 9. Initialize new feedback fusion weights
    # ==========================================

    # Small nonzero initialization allows gradients to reach
    # the upstream FeedbackConnection modules immediately.

    for module in model.modules():
        if (isinstance(module, ConvRecurrentContainer) and module.feedback_pwconv is not None):
            nn.init.normal_(module.feedback_pwconv.weight, mean=0.0, std=1e-3)

    # ==========================================
    # 10. Move model to device
    # ==========================================

    model = model.to(device)
    print("\nPretrained loading completed successfully.")
    print(f"Loaded parameters: {len(compatible_state_dict)}")
    print(f"New feedback parameters: {len(allowed_missing_keys)}")
    if shape_mismatches:
        print("\nSkipped classifier shape mismatches:")
        for key, old_shape, new_shape in shape_mismatches:
            print(
                f"  {key}: checkpoint={old_shape}, "
                f"model={new_shape}"
            )

    print("\nFeedback connections:", feedback_connection)

    return model

    
from src.model.lorafb_snet import ConvRecurrentContainer
from spikingjelly.activation_based import neuron


def configure_feedback_training(model):

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze top-down feedback connections
    for param in model.feedback_modules.parameters():
        param.requires_grad = True

    # Unfreeze feedback fusion inside target containers
    for module in model.modules():
        if isinstance(module, ConvRecurrentContainer):
            if module.feedback_pwconv is not None:
                for param in module.feedback_pwconv.parameters():
                    param.requires_grad = True
                for param in module.feedback_bn.parameters():
                    param.requires_grad = True

    # Collect all trainable parameters
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    return trainable_params


def set_feedback_train_mode(model):
    model.eval()                                  # frozen BN uses running stats
    for m in model.modules():
        if isinstance(m, neuron.BaseNode):
            m.train()                             # surrogate gradients need train mode
    model.feedback_modules.train()                # trainable feedback path (incl. its BN)
    for m in model.modules():
        if hasattr(m, "feedback_bn") and m.feedback_bn is not None:
            m.feedback_bn.train()
        # (the other BNs in the containers stay in eval)
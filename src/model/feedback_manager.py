import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer

class FeedbackConnection(nn.Module):
    def __init__(self,source_channels,target_channels,source_stride,target_stride):
        super().__init__()
        self.last_output = None
        self.source_channels = source_channels
        self.target_channels = target_channels
        self.source_stride = source_stride
        self.target_stride = target_stride
        if source_stride % target_stride != 0:
            raise ValueError(
                f"Source Stride {source_stride} Must Be "
                f"Divisible by Target Stride {target_stride}"
            )
        scale_factor = source_stride // target_stride
        modules = []
        current_channels = source_channels
        if scale_factor > 1:
            if scale_factor & (scale_factor - 1) != 0:
                raise ValueError("Scale Factor must be a Power of 2")
            num_upsamples = int(torch.log2(torch.tensor(float(scale_factor))).item())
            for _ in range(num_upsamples):
                modules.append(
                    nn.ConvTranspose2d(
                        current_channels,
                        current_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        groups=current_channels,
                        bias=False
                    )
                )
                modules.append(layer.BatchNorm2d(current_channels))
                modules.append(neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True))
            if current_channels != target_channels:
                modules.append(
                    nn.Conv2d(
                        current_channels,
                        target_channels,
                        kernel_size=1,
                        bias=False
                    )
                )
                modules.append(neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True))
            self.feedback = nn.Sequential(*modules)
    def forward(self,x):
        out = self.feedback(x)
        self.last_output = out.detach()
        return out


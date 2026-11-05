import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer

class FeedbackConnection(nn.Module):

    def __init__(
        self,
        source_channels,
        target_channels,
        source_stride,
        target_stride
    ):
        super().__init__()

        self.source_channels = source_channels
        self.target_channels = target_channels

        if source_stride % target_stride != 0:
            raise ValueError("Invalid stride ratio")

        scale_factor = source_stride // target_stride

        if scale_factor & (scale_factor - 1):
            raise ValueError("Scale factor must be a power of 2")

        modules = []
        current_channels = source_channels

        # Spatial upsampling
        num_upsamples = int(torch.log2(torch.tensor(float(scale_factor))).item())

        for _ in range(num_upsamples):

            modules.extend([
                nn.ConvTranspose2d(
                    current_channels,
                    current_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    groups=current_channels,
                    bias=False
                ),
                layer.BatchNorm2d(current_channels),
                neuron.LIFNode(
                    surrogate_function=surrogate.ATan(),
                    detach_reset=True
                )
            ])

        # Channel adaptation
        if current_channels != target_channels:

            modules.extend([
                nn.Conv2d(
                    current_channels,
                    target_channels,
                    kernel_size=1,
                    bias=False
                ),
                layer.BatchNorm2d(target_channels),
                neuron.LIFNode(
                    surrogate_function=surrogate.ATan(),
                    detach_reset=True
                )
            ])

        self.feedback = (
            nn.Sequential(*modules)
            if modules else nn.Identity()
        )

        # Input-dependent gate
        self.gate = nn.Conv2d(
            target_channels * 2,
            target_channels,
            kernel_size=1,
            bias=True
        )

        # Initially allow a small amount of feedback
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

        self.last_output = None
        self.last_gate = None

    def forward(self, source, target):
        temporal = source.dim() == 5
        if temporal:
            T, B, C, H, W = source.shape
            source = source.reshape(T * B, C, H, W)
            target = target.reshape(
                T * B,
                target.shape[2],
                target.shape[3],
                target.shape[4]
            )

        feedback = self.feedback(source)
        if feedback.shape != target.shape:
            raise RuntimeError(
                f"Feedback shape {feedback.shape} "
                f"does not match target shape {target.shape}"
            )

        # Calculate gate
        gate_input = torch.cat([target, feedback], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))

        # Gated feedback integration
        output = target + gate * feedback

        if temporal:
            output = output.reshape(T, B, *output.shape[1:])
            gate = gate.reshape(T, B, *gate.shape[1:])

        self.last_output = output.detach()
        self.last_gate = gate.detach()

        return output
import torch
import torch.nn as nn
from spikingjelly.activation_based import base, layer, neuron, surrogate, functional
from src.model.feedback_manager import FeedbackConnection

class ConvRecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module, in_channels, out_channels, stride, step_mode="s", external_feedback_channels=0):
        super().__init__()
        self.step_mode = step_mode
        assert not hasattr(sub_module, "step_mode") or sub_module.step_mode == "s"
        self.sub_module_out_channels = out_channels
        self.sub_module = sub_module
        self.external_feedback_channels = external_feedback_channels
        if stride > 1:
            self.dwconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=stride, 
                                             padding=1, output_padding=1, groups=out_channels, bias=False)
            self.bn1 = layer.BatchNorm2d(out_channels)
            self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        else:
            self.dwconv = None
        recurrent_fusion_channels = in_channels + out_channels
        self.pwconv = nn.Conv2d(recurrent_fusion_channels, in_channels, kernel_size=1, stride=1, bias=False)
        self.bn = layer.BatchNorm2d(in_channels)
        if external_feedback_channels > 0:
            self.feedback_pwconv = nn.Conv2d(external_feedback_channels, in_channels, kernel_size=1, bias=False)
            self.feedback_bn = layer.BatchNorm2d(in_channels)
            nn.init.normal_(self.feedback_pwconv.weight, mean=0.0, std=1e-3)
        else:
            self.feedback_pwconv = None
            self.feedback_bn = None
        self.sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.register_memory("y", None)
    
    def single_step_forward(self, x, external_feedback=None):
        if self.y is None:
            if self.dwconv is None:
                self.y = torch.zeros(x.size(0), self.sub_module_out_channels, x.size(2), x.size(3)).to(x)
            else:
                h = (x.size(2) + 2 - (3 - 1) - 1) // 2 + 1
                w = (x.size(3) + 2 - (3 - 1) - 1) // 2 + 1
                self.y = torch.zeros(x.size(0), self.sub_module_out_channels, h, w).to(x)
        if self.dwconv is None:
            recurrent = self.y
        else:
            recurrent = self.sn1(self.bn1(self.dwconv(self.y)))
        original_fusion = torch.cat([x, recurrent], dim=1)
        original_fusion = self.bn(self.pwconv(original_fusion))
        if self.feedback_pwconv is not None:
            if external_feedback is None:
                external_feedback = x.new_zeros(
                    x.shape[0],
                    self.external_feedback_channels,
                    x.shape[-2],
                    x.shape[-1]
                )
            if external_feedback.shape[-2:] != x.shape[-2:]:
                raise ValueError("External feedback spatial dimensions must match container input dimensions.")
            if external_feedback.shape[1] != self.external_feedback_channels:
                raise ValueError(
                    f"Expected {self.external_feedback_channels} "
                    f"feedback channels, got "
                    f"{external_feedback.shape[1]}."
                )
            feedback_fusion = self.feedback_pwconv(external_feedback)
            feedback_fusion = self.feedback_bn(
                feedback_fusion
            )
            fused = original_fusion + feedback_fusion
        else:
            fused = original_fusion
        fused = self.sn(fused)
        self.y = self.sub_module(fused)
        return self.y


def sew_function(x, y, cnf):
    if cnf == "ADD":
        return x + y
    elif cnf == "AND":
        return x * y
    elif cnf == "OR":
        return x + y - x * y
    else:
        raise NotImplementedError


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, cnf=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.stride = stride
        self.cnf = cnf
    
    def forward(self, x):
        identity = x
        
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.sn2(self.bn2(self.conv2(out)))
        
        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))
        
        out = sew_function(out, identity, self.cnf)
        
        return out
    
    def extra_repr(self):
        return super().extra_repr() + f"cnf={self.cnf}"


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, cnf=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.sn2 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.stride = stride
        self.cnf = cnf
    
    def forward(self, x):
        identity = x
        
        out = self.sn1(self.bn1(self.conv1(x)))
        out = self.sn2(self.bn2(self.conv2(out)))
        out = self.sn3(self.bn3(self.conv3(out)))
        
        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))
        
        out = sew_function(out, identity, self.cnf)
        
        return out
    
    def extra_repr(self):
        return super().extra_repr() + f"cnf={self.cnf}"


class LoRaFBSNet(nn.Module):
    def __init__(self, block, layers, num_classes=101, groups=1, width_per_groups=64, 
                 feedback_connection=None, norm_layer=None, cnf=None, zero_init_residual=False):
        super(LoRaFBSNet, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_planes = 64
        self.stage_channels = {
            1: 64 * block.expansion,
            2: 128 * block.expansion,
            3: 256 * block.expansion,
            4: 512 * block.expansion
        }
        self.stage_strides = {
            1: 4,
            2: 8,
            3: 16,
            4: 32
        }
        self.feedback_connection = tuple(feedback_connection or ())
        self.feedback_modules = nn.ModuleDict()
        incoming_feedback_channels = {i: 0 for i in range(1, 5)}
        for source, target in self.feedback_connection:
            if source not in (1, 2, 3, 4):
                raise ValueError(f"Invalid source stage: {source}")
            if target not in (1, 2, 3, 4):
                raise ValueError(f"Invalid target stage: {target}")
            if source <= target:
                raise ValueError("Feedback must connect a higher stage to a lower stage.")

            key = f"{source}_to_{target}"
            if key in self.feedback_modules:
                raise ValueError(f"Duplicate feedback connection: {key}")
            
            self.feedback_modules[key] = FeedbackConnection(
                source_channels=self.stage_channels[source],
                target_channels=self.stage_channels[target],
                source_stride=self.stage_strides[source],
                target_stride=self.stage_strides[target]
            )
            incoming_feedback_channels[target] += self.stage_channels[target]
        self.groups = groups
        self.base_width = width_per_groups
        self.conv1 = layer.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.in_planes)
        self.sn1 = neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layer1 = self._make_layer(block, 64, layers[0], cnf=cnf)
        self.recurrent_layer1 = ConvRecurrentContainer(layer1, in_channels=64, out_channels=self.stage_channels[1], stride=1, external_feedback_channels=incoming_feedback_channels[1])
        layer2 = self._make_layer(block, 128, layers[1], stride=2, cnf=cnf)
        self.recurrent_layer2 = ConvRecurrentContainer(layer2, in_channels=self.stage_channels[1], out_channels=self.stage_channels[2], stride=2, external_feedback_channels=incoming_feedback_channels[2])
        layer3 = self._make_layer(block, 256, layers[2], stride=2, cnf=cnf)
        self.recurrent_layer3 = ConvRecurrentContainer(layer3, in_channels=self.stage_channels[2], out_channels=self.stage_channels[3], stride=2, external_feedback_channels=incoming_feedback_channels[3])
        layer4 = self._make_layer(block, 512, layers[3], stride=2, cnf=cnf)
        self.recurrent_layer4 = ConvRecurrentContainer(layer4, in_channels=self.stage_channels[3], out_channels=self.stage_channels[4], stride=2, external_feedback_channels=incoming_feedback_channels[4])
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(self.stage_channels[4], num_classes)
        
        for m in self.modules():
            if isinstance(m, (layer.Conv2d, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, ConvRecurrentContainer):
                if m.feedback_pwconv is not None:
                    nn.init.normal_(m.feedback_pwconv.weight, mean=0.0, std=1e-3)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, num_blocks, stride=1, cnf=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, self.groups, self.base_width, self._norm_layer, cnf))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, groups=self.groups, base_width=self.base_width, norm_layer=self._norm_layer, cnf=cnf))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        functional.reset_net(self)
        T, B, C, H, W = x.shape
        stages = [self.recurrent_layer1, self.recurrent_layer2, self.recurrent_layer3, self.recurrent_layer4]
        outputs = []
        for t in range(T):
            x_t = x[t]
            x_t = self.conv1(x_t)
            x_t = self.bn1(x_t)
            x_t = self.sn1(x_t)
            x_t = self.maxpool(x_t)
            previous_outputs = [stage.y for stage in stages]
            for source_idx, stage in enumerate(stages):
                target_stage = source_idx + 1
                incoming_feedback = []
                for source, target in self.feedback_connection:
                    if target != target_stage:
                        continue
                    key = f"{source}_to_{target}"
                    source_feature = previous_outputs[source - 1]
                    if source_feature is None:
                        feedback_feature = x_t.new_zeros(B, self.stage_channels[target], x_t.shape[-2], x_t.shape[-1]).to(x_t)
                    else:
                        feedback_feature = self.feedback_modules[key](source_feature)
                    incoming_feedback.append(feedback_feature)
                if len(incoming_feedback) > 0:
                    external_feedback = torch.cat(incoming_feedback, dim=1)
                else:
                    external_feedback = None

                if external_feedback is not None:
                    print(
                        f"Stage {source_idx + 1}: "
                        f"input={tuple(x_t.shape)}, "
                        f"feedback={tuple(external_feedback.shape)}"
                    )
                x_t = stage(x_t, external_feedback)
            pooled = self.avgpool(x_t)
            pooled = torch.flatten(pooled, 1)
            logits = self.fc(pooled)
            outputs.append(logits)
        outputs = torch.stack(outputs, dim=0)
        return outputs

        
def lorafb_snet18(**kwargs):
    return LoRaFBSNet(BasicBlock, [2, 2, 2, 2], **kwargs)



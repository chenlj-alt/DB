#
# reference: https://github.com/WenmuZhou/PytorchOCR.git
#

import torch
import torch.nn as nn
import math

from collections import OrderedDict

BatchNorm2d = nn.BatchNorm2d

__all__ = ['resnet_vd_18', 'resnet_vd_34', 'resnet_vd_50', 'resnet_vd_101', 'resnet_vd_152']


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'relu':
            self.act = nn.ReLU()
        elif act is None:
            self.act = None

    def load_3rd_state_dict(self, _3rd_name, _state, _name_prefix):
        to_load_state_dict = OrderedDict()
        if _3rd_name == 'paddle':
            to_load_state_dict['conv.weight'] = torch.Tensor(_state[f'{_name_prefix}_weights'])
            if _name_prefix == 'conv1':
                bn_name = f'bn_{_name_prefix}'
            else:
                bn_name = f'bn{_name_prefix[3:]}'
            to_load_state_dict['bn.weight'] = torch.Tensor(_state[f'{bn_name}_scale'])
            to_load_state_dict['bn.bias'] = torch.Tensor(_state[f'{bn_name}_offset'])
            to_load_state_dict['bn.running_mean'] = torch.Tensor(_state[f'{bn_name}_mean'])
            to_load_state_dict['bn.running_var'] = torch.Tensor(_state[f'{bn_name}_variance'])
            self.load_state_dict(to_load_state_dict)
        else:
            pass

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvBNACTWithPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, act=None):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=(kernel_size - 1) // 2,
                              groups=groups,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act is None:
            self.act = None
        else:
            self.act = nn.ReLU()

    def load_3rd_state_dict(self, _3rd_name, _state, _name_prefix):
        to_load_state_dict = OrderedDict()
        if _3rd_name == 'paddle':
            to_load_state_dict['conv.weight'] = torch.Tensor(_state[f'{_name_prefix}_weights'])
            if _name_prefix == 'conv1':
                bn_name = f'bn_{_name_prefix}'
            else:
                bn_name = f'bn{_name_prefix[3:]}'
            to_load_state_dict['bn.weight'] = torch.Tensor(_state[f'{bn_name}_scale'])
            to_load_state_dict['bn.bias'] = torch.Tensor(_state[f'{bn_name}_offset'])
            to_load_state_dict['bn.running_mean'] = torch.Tensor(_state[f'{bn_name}_mean'])
            to_load_state_dict['bn.running_var'] = torch.Tensor(_state[f'{bn_name}_variance'])
            self.load_state_dict(to_load_state_dict)
        else:
            pass

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ShortCut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, name, if_first=False):
        super().__init__()
        assert name is not None, 'shortcut must have name'

        self.name = name
        if in_channels != out_channels or stride != 1:
            if if_first:
                self.conv = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                      padding=0, groups=1, act=None)
            else:
                self.conv = ConvBNACTWithPool(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                              groups=1, act=None)
        elif if_first:
            self.conv = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                  padding=0, groups=1, act=None)
        else:
            self.conv = None

    def load_3rd_state_dict(self, _3rd_name, _state):
        if _3rd_name == 'paddle':
            if self.conv:
                self.conv.load_3rd_state_dict(_3rd_name, _state, self.name)
        else:
            pass

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first, name):
        super().__init__()
        assert name is not None, 'bottleneck must have name'
        self.name = name
        self.conv0 = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                               groups=1, act='relu')
        self.conv1 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, groups=1, act='relu')
        self.conv2 = ConvBNACT(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, stride=1,
                               padding=0, groups=1, act=None)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels * 4, stride=stride,
                                 if_first=if_first, name=f'{name}_branch1')
        self.relu = nn.ReLU()
        self.output_channels = out_channels * 4

    def load_3rd_state_dict(self, _3rd_name, _state):
        self.conv0.load_3rd_state_dict(_3rd_name, _state, f'{self.name}_branch2a')
        self.conv1.load_3rd_state_dict(_3rd_name, _state, f'{self.name}_branch2b')
        self.conv2.load_3rd_state_dict(_3rd_name, _state, f'{self.name}_branch2c')
        self.shortcut.load_3rd_state_dict(_3rd_name, _state)

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, if_first, name):
        super().__init__()
        assert name is not None, 'block must have name'
        self.name = name

        self.conv0 = ConvBNACT(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1, groups=1, act='relu')
        self.conv1 = ConvBNACT(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                               groups=1, act=None)
        self.shortcut = ShortCut(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                 name=f'{name}_branch1', if_first=if_first, )
        self.relu = nn.ReLU()
        self.output_channels = out_channels

    def load_3rd_state_dict(self, _3rd_name, _state):
        if _3rd_name == 'paddle':
            self.conv0.load_3rd_state_dict(_3rd_name, _state, f'{self.name}_branch2a')
            self.conv1.load_3rd_state_dict(_3rd_name, _state, f'{self.name}_branch2b')
            self.shortcut.load_3rd_state_dict(_3rd_name, _state)
        else:
            pass

    def forward(self, x):
        y = self.conv0(x)
        y = self.conv1(y)
        y = y + self.shortcut(x)
        return self.relu(y)


class ResNetVd(nn.Module):
    def __init__(self, blockclass, layers, num_layers, num_classes=1000,
                 dcn=None, stage_with_dcn=(False, False, False, False)):
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        self.inplanes = 32
        super(ResNetVd, self).__init__()

        # ===> head: 1 7*7 conv --> 3 3*3 conv
        self.conv1 = nn.Sequential(
            ConvBNACT(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, act='relu'),
            ConvBNACT(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, act='relu'),
            ConvBNACT(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, act='relu')
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ===> 4 stages
        self.stages = nn.ModuleList()
        self.out_channels = []
        num_filters = [64, 128, 256, 512]
        in_channels = 64
        for block in range(len(layers)):
            block_list = []
            for i in range(layers[block]):
                if num_layers >= 50:
                    if num_layers in [101, 152, 200] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                else:
                    conv_name = f'res{str(block + 2)}{chr(97 + i)}'

                block_list.append(blockclass(in_channels,
                                             num_filters[block],
                                             stride=2 if i == 0 and block != 0 else 1,
                                             if_first=block == i == 0,
                                             name=conv_name))

                in_channels = block_list[-1].output_channels
            self.out_channels.append(in_channels)
            self.stages.append(nn.Sequential(*block_list))

    def load_3rd_state_dict(self, _3rd_name, _state):
        if _3rd_name == 'paddle':
            for m_conv_index, m_conv in enumerate(self.conv1, 1):
                m_conv.load_3rd_state_dict(_3rd_name, _state, f'conv1_{m_conv_index}')
            for m_stage in self.stages:
                for m_block in m_stage:
                    m_block.load_3rd_state_dict(_3rd_name, _state)
        else:
            pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)

        return out


def resnet_vd_18(**kwargs):
    """Constructs a ResNet_vd_18 model."""
    model = ResNetVd(BasicBlock, [2, 2, 2, 2], 18, **kwargs)
    return model


def resnet_vd_34(**kwargs):
    """Constructs a ResNet_vd_34 model."""
    model = ResNetVd(BasicBlock, [3, 4, 6, 3], 34, **kwargs)
    return model


def resnet_vd_50(**kwargs):
    """Constructs a ResNet_vd_50 model."""
    model = ResNetVd(BottleneckBlock, [3, 4, 6, 3], 50, **kwargs)
    return model


def resnet_vd_101(**kwargs):
    """Constructs a ResNet_vd_101 model."""
    model = ResNetVd(BottleneckBlock, [3, 4, 23, 3], 101, **kwargs)
    return model


def resnet_vd_152(**kwargs):
    """Constructs a ResNet_vd_152 model."""
    model = ResNetVd(BottleneckBlock, [3, 8, 36, 3], 152, **kwargs)
    return model

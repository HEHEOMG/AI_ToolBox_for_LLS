from __future__ import division

import torch.nn as nn



def conv_batch(in_num, out_num, kernel_size = 3, padding = 1,stride = 1):
    """
    basic block for darknet
    :param in_num:
    :param out_num:
    :param kernel_size:
    :param padding:
    :param stride:
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(in_num,
                  out_num,
                  kernel_size= kernel_size,
                  stride=stride,
                  padding=padding,
                  bias = False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU()
    )



# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)

        out += residual
        return out



class Darknet(nn.Module):
    """Darknet model"""

    arch_settings = {
        53: (DarkResidualBlock, (1, 2, 8, 8, 4))
    }

    def __init__(self, depth, in_channels = 3, num_stages = 5, strides = (2,2,2,2,2), num_classes = 1000):
        """

        :param depth:
        :param in_channels:
        :param num_stages:
        :param strides:
        :param num_classes:
        """

        super(Darknet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth{} for Darknet'.format(depth))
        self.depth = depth
        self.in_chanels = in_channels
        self.num_stages = num_stages
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 32
        self.num_classes = num_classes

        self.conv1 = conv_batch(in_channels,32)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            pre_layer = conv_batch(self.inplanes, self.inplanes*2, stride=stride)
            pre_layer_name = 'conv{}'.format(i + 2)
            self.add_module(pre_layer_name, pre_layer)
            self.res_layers.append(pre_layer_name)

            res_layers = self.make_res_layer(self.block,
                                             in_channels = self.inplanes*2,
                                             num_blocks = num_blocks)
            res_layers_name = 'residual_block{}'.format(i+1)
            self.add_module(res_layers_name, res_layers)
            self.res_layers.append(res_layers_name)
            self.inplanes *=2

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, self.num_classes)



    def forward(self, x, targets = None):
        out = self.conv1(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            out = res_layer(out)
        out = self.global_avg_pool(out)
        out = out.view(-1,1024)
        return out



    def make_res_layer(self,
                       block,
                       in_channels,
                       num_blocks):
        layers = []

        for i in range(num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)



if __name__ == '__main__':
    darknet = Darknet(53)
    print(darknet)

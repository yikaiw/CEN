import torch
import torch.nn.functional as F
import torch.nn as nn
from .modules import *
import config as cf


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2,
                 padding=1, activation=True, instance_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = ModuleParallel(nn.Conv2d(input_size, output_size, kernel_size, stride, padding))
        self.activation = activation
        self.lrelu = ModuleParallel(nn.LeakyReLU(0.2, True))
        self.instance_norm = instance_norm
        self.insnorm_conv = InstanceNorm2dParallel(output_size)
        self.use_exchange = cf.use_exchange
        
        if self.use_exchange:
            self.exchange = Exchange()
            self.insnorm_threshold = cf.insnorm_threshold
            self.insnorm_list = []
            for module in self.insnorm_conv.modules():
                if isinstance(module, nn.InstanceNorm2d):
                    self.insnorm_list.append(module)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.instance_norm:
            out = self.insnorm_conv(out)
            if self.use_exchange and len(x) > 1:
                out = self.exchange(out, self.insnorm_list, self.insnorm_threshold)
        return out


class ConvBlockShare(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2,
                 padding=1, activation=True, instance_norm=True):
        super(ConvBlockShare, self).__init__()
        self.conv = ModuleParallel(nn.Conv2d(input_size, output_size, kernel_size, stride, padding))
        self.activation = activation
        self.lrelu = ModuleParallel(nn.LeakyReLU(0.2, True))
        self.instance_norm = instance_norm
        self.insnorm = ModuleParallel(nn.InstanceNorm2d(output_size, affine=True, track_running_stats=True))

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.instance_norm:
            out = self.insnorm(out)
        return out


class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2,
                 padding=1, instance_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = ModuleParallel(nn.ConvTranspose2d(
            input_size, output_size, kernel_size, stride, padding))
        self.insnorm_deconv = InstanceNorm2dParallel(output_size)
        self.drop = ModuleParallel(nn.Dropout(0.5))
        self.relu = ModuleParallel(nn.ReLU(True))
        self.instance_norm = instance_norm
        self.dropout = dropout

    def forward(self, x):
        if self.instance_norm:
            out = self.insnorm_deconv(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            out = self.drop(out)
        return out


class Generator(nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, instance_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv6 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv7 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv8 = ConvBlock(num_filter * 8, num_filter * 8, instance_norm=False)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=True)
        self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
        self.deconv5 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
        self.deconv6 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
        self.deconv7 = DeconvBlock(num_filter * 2 * 2, num_filter)
        self.deconv8 = DeconvBlock(num_filter * 2, output_dim, instance_norm=False)
        self.tanh = ModuleParallel(nn.Tanh())

        self.alpha = nn.Parameter(torch.ones(cf.num_parallel, requires_grad=True))
        self.register_parameter('alpha', self.alpha)

    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        enc7 = self.conv7(enc6)
        enc8 = self.conv8(enc7)
        # Decoder with skip-connections
        dec1 = self.deconv1(enc8)
        dec1 = [torch.cat([dec_, enc_], 1) for (dec_, enc_) in zip(dec1, enc7)]
        dec2 = self.deconv2(dec1)
        dec2 = [torch.cat([dec_, enc_], 1) for (dec_, enc_) in zip(dec2, enc6)]
        dec3 = self.deconv3(dec2)
        dec3 = [torch.cat([dec_, enc_], 1) for (dec_, enc_) in zip(dec3, enc5)]
        dec4 = self.deconv4(dec3)
        dec4 = [torch.cat([dec_, enc_], 1) for (dec_, enc_) in zip(dec4, enc4)]
        dec5 = self.deconv5(dec4)
        dec5 = [torch.cat([dec_, enc_], 1) for (dec_, enc_) in zip(dec5, enc3)]
        dec6 = self.deconv6(dec5)
        dec6 = [torch.cat([dec_, enc_], 1) for (dec_, enc_) in zip(dec6, enc2)]
        dec7 = self.deconv7(dec6)
        dec7 = [torch.cat([dec_, enc_], 1) for (dec_, enc_) in zip(dec7, enc1)]
        dec8 = self.deconv8(dec7)
        out = self.tanh(dec8)

        ens = 0
        alpha_soft = F.softmax(self.alpha, dim=0)
        for l in range(cf.num_parallel):
            ens += alpha_soft[l] * out[l].detach()
        out.append(ens)
        return out, alpha_soft

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean, std)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean, std)


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlockShare(input_dim, num_filter, activation=False, instance_norm=False)
        self.conv2 = ConvBlockShare(num_filter, num_filter * 2)
        self.conv3 = ConvBlockShare(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlockShare(num_filter * 4, num_filter * 8, stride=1)
        self.conv5 = ConvBlockShare(num_filter * 8, output_dim, stride=1, instance_norm=False)
        self.sigmoid = ModuleParallel(nn.Sigmoid())

    def forward(self, x, label):
        if isinstance(label, list):
            x = [torch.cat([x_, label_], 1) for (x_, label_) in zip(x, label)]
        else:
            x = [torch.cat([x_, label], 1) for x_ in x]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = self.sigmoid(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean, std)


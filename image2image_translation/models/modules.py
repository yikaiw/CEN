import torch.nn as nn
import config as cf
import torch


class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, insnorm, insnorm_threshold):
        insnorm1, insnorm2 = insnorm[0].weight.abs(), insnorm[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, insnorm1 >= insnorm_threshold] = x[0][:, insnorm1 >= insnorm_threshold]
        x1[:, insnorm1 < insnorm_threshold] = x[1][:, insnorm1 < insnorm_threshold]
        x2[:, insnorm2 >= insnorm_threshold] = x[1][:, insnorm2 >= insnorm_threshold]
        x2[:, insnorm2 < insnorm_threshold] = x[0][:, insnorm2 < insnorm_threshold]
        return [x1, x2]


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class InstanceNorm2dParallel(nn.Module):
    def __init__(self, num_features):
        super(InstanceNorm2dParallel, self).__init__()
        for i in range(cf.num_parallel):
            setattr(self, 'insnorm_' + str(i), nn.InstanceNorm2d(num_features, affine=True, track_running_stats=True))

    def forward(self, x_parallel):
        return [getattr(self, 'insnorm_' + str(i))(x) for i, x in enumerate(x_parallel)]

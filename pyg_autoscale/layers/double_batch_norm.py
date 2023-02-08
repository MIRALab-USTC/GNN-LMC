import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d

class BNWapper(torch.nn.Module):
    def __init__(self, bn):
        super().__init__()
        self.bn = bn

    def forward(self, input: Tensor, batch_size=None) -> Tensor:
        return self.bn(input)

    def reset_parameters(self):
        self.bn.reset_parameters()

class SharedBN(torch.nn.Module):
    def __init__(self,num_features,*args,):
        super().__init__()
        self.bn = BatchNorm1d(num_features, *args,)

    def forward(self, input: Tensor, batch_size=None) -> Tensor:
        if batch_size is None or not self.training:
            return self.bn(input)
        else:
            output_ib = self.bn(input[:batch_size])
            self.bn.track_running_stats = False
            output_ob = self.bn(input[batch_size:])
            self.bn.track_running_stats = True
            return torch.cat([output_ib, output_ob], dim=0)

    def reset_parameters(self):
        self.bn.reset_parameters()


class DoubleBN(torch.nn.Module):
    def __init__(self,num_features,*args,):
        super().__init__()
        self.bn_ib = BatchNorm1d(num_features, *args,)
        self.bn_ob = BatchNorm1d(num_features, *args,)

    def forward(self, input: Tensor, batch_size=None) -> Tensor:
        if batch_size is None:
            return self.bn_ib(input)
        else:
            output_ib = self.bn_ib(input[:batch_size])
            output_ob = self.bn_ob(input[batch_size:])
            return torch.cat([output_ib, output_ob], dim=0)

    def reset_parameters(self):
        self.bn_ib.reset_parameters()
        self.bn_ob.reset_parameters()
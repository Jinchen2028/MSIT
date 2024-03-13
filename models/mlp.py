import torch.nn as nn
import torch
from models import register
import numpy as np

@register('mlp')
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list, act='relu'):
        super().__init__()

        if act is None:
            self.act = None
        elif act.lower() == 'relu':
            self.act = nn.ReLU(True) 
        elif act.lower() == 'gelu':
            self.act = nn.GELU()
        else:
            assert False, f'activation {act} is not supported'

        layers = []
        lastv = in_dim

        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            if self.act:
                layers.append(self.act)
            lastv = hidden

        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)

if __name__ == "__main__":
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = MLP (in_dim=256,out_dim=3,hidden_list=[256,256,256,256], act='gelu')
    print(count_parameters(model))

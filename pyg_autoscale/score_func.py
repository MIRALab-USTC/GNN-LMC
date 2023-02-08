import torch
import numpy as np

def linear(x):
    return x

def const(x):
    return torch.ones_like(x)

def concave(x):
    return x*(2-x)

def convex(x):
    return x*x

def sin(x):
    return torch.sin((x-0.5)*np.pi)
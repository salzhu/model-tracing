import torch

def spcor(x,y):
  n = len(x)
  return 1 - torch.sum(6 * torch.square(x-y)) / (n * (n - 1))

def pdists(x,y):
  with torch.no_grad():
    xsum = torch.sum(torch.square(x),axis=-1)
    ysum = torch.sum(torch.square(y),axis=-1)

    dists = xsum.view(-1,1) + ysum.view(1,-1) - 2 * x@y.T

  return dists
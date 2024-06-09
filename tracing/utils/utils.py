import torch

def spcor(x,y):
  n = len(x)
  with torch.no_grad():
    r = 1 - torch.sum(6 * torch.square(x-y)) / (n * (n - 1))
  
  return r

def pdists(x,y):
  x = x.to("cuda")
  y = y.to("cuda")

  with torch.no_grad():
    xsum = torch.sum(torch.square(x),axis=-1)
    ysum = torch.sum(torch.square(y),axis=-1)

    dists = xsum.view(-1,1) + ysum.view(1,-1) - 2 * x@y.T

  return dists.cpu()

def cossim(x,y):
  x = x.to("cuda")
  y = y.to("cuda")

  with torch.no_grad():
    similarities = x@y.T / (torch.linalg.norm(x,axis=-1).view(-1,1) * torch.linalg.norm(y,axis=-1).view(1,-1))
  
  return similarities
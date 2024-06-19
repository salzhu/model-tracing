import torch
import matplotlib.pyplot as plt

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
  
  return similarities.cpu()

def normalize_trace(trace, alphas):
    slope = trace[-1] - trace[0]
    start = trace[0]
    for i in range(len(trace)):
        trace[i] -= slope * alphas[i]
        trace[i] -= start
    return trace

def plot_trace(losses, alphas, normalize, model_a_name, model_b_name, plot_path):

    plt.figure(figsize=(8, 6))
    if normalize: 
        losses = normalize_trace(losses, alphas)
    plt.plot(alphas, losses, 'o-')
            
    plt.xlabel("Alpha")
    plt.ylabel("Loss")
    plt.title(f"{model_a_name} (Left) vs {model_b_name} (Right)")
    plot_filename = f"{plot_path}.png"

    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()
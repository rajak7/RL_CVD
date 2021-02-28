import numpy as np
import torch
import os

def log_normal(x,m,sigma):
    pi_const = 2.0 * np.pi
    z = - ((x-m) ** 2)/(2.0 * sigma)  - torch.log(torch.sqrt(pi_const * sigma))
    return z

def sample_gaussian(m,v):
    epsilon = torch.normal(torch.zeros(m.size()),torch.ones(m.size()))
    z = m + torch.sqrt(v) * epsilon
    return z

#creates a visulization grid
#0->no_atom, 1->2H, 2-1T, 3->defect 
def viz_grid(Xs, padding):
    N, H, W = Xs.shape
    grid_height = H * 4 + padding * (4 + 1)
    grid_width = W * 5 + padding * (5 + 1)
    grid = np.zeros((grid_height, grid_width))
    next_idx = 0
    y0, y1 = padding, H + padding
    for y in range(4):
        x0, x1 = padding, W + padding
        for x in range(5):
            if next_idx < N:
                img = Xs[next_idx]
                grid[y0:y1, x0:x1] = img
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid

#save a trained lstm model
def save_model(model,model_name):
    save_dir = os.path.join('checkpoints', model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-latest.pt')
    d = {
        'encoder' : model.encoder.state_dict(),
        'decoder' : model.decoder.state_dict(),
        'lstm' : model.lstm.state_dict(),
        'sigma_type' : model.sigma_type,
        'sigma' : model.sigma,
        'loss_type' : model.loss_type
    }
    torch.save(d,file_path)
    return

#load a trained lstm model
def load_model(model,model_name,strict=True,device=None):
    path = os.path.join('checkpoints',model_name,'model-latest.pt')
    ckpt = torch.load(path,map_location=device)
    model.encoder.load_state_dict(ckpt['encoder'],strict=strict)
    model.decoder.load_state_dict(ckpt['decoder'],strict=strict)
    model.lstm.load_state_dict(ckpt['lstm'],strict=strict)
    model.sigma_type = ckpt['sigma_type']
    model.sigma = ckpt['sigma']
    model.loss_type = ckpt['loss_type']
    return




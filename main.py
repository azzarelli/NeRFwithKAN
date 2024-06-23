import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from nerf_models.nerf import NerfModel, KANeRF


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)


emb_x = 10 # emb_x: [batch_size, embedding_dim_pos * 6]
emb_d = 4 # emb_d: [batch_size, embedding_dim_direction * 6]


def positional_encoding(x, L):
    out = [x]
    for j in range(L):
        out.append(torch.sin(2 ** j * x))
        out.append(torch.cos(2 ** j * x))
    return torch.cat(out, dim=1)

def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = ray_origins.device
    
    t = torch.linspace(hn, hf, nb_bins, device=device).expand(ray_origins.shape[0], nb_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)   # [batch_size, nb_bins, 3]
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1) 
    
  
    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))


    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background 
    return c + 1 - weight_sum.unsqueeze(-1)


def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
          nb_bins=192, H=400, W=400, use_kan=False):

    training_loss = []
    # Training loop
    for epoch in range(nb_epochs):
        nerf_model.train()
        epoch_loss = 0
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{nb_epochs}") as pbar:
            for i, batch in enumerate(data_loader):
                ray_origins = batch[:, :3].to(device)
                ray_directions = batch[:, 3:6].to(device)
                ground_truth_px_values = batch[:, 6:].to(device)

                regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins) 
                loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                training_loss.append(loss.item())

                # Update progress bar with current loss
                pbar.set_postfix({"Loss": loss.item(), "Iteration": i})
                pbar.update(1)
            
            # Update scheduler
            scheduler.step()

        # Evaluate model
        nerf_model.eval()
        testing_results = []
        for img_index in range(200):
            result = test(hn, hf, testing_dataset, img_index=img_index, nb_bins=nb_bins, H=H, W=W)
            testing_results.append(result)
        
        avg_test_result = sum(testing_results) / len(testing_results)
        # Print testing results for this epoch
        print(f"Testing results after epoch {epoch+1}: {avg_test_result}")

    return training_loss

@torch.no_grad()
def test(hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    """
    Args:
        hn: near plane distance
        hf: far plane distance
        dataset: dataset to render
        chunk_size (int, optional): chunk size for memory efficiency. Defaults to 10.
        img_index (int, optional): image index to render. Defaults to 0.
        nb_bins (int, optional): number of bins for density estimation. Defaults to 192.
        H (int, optional): image height. Defaults to 400.
        W (int, optional): image width. Defaults to 400.
        
    Returns:
        None: None
    """
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = []   # list of regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)        
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f'novel_views/img_{img_index}.png', bbox_inches='tight')
    plt.close()




if __name__ == '__main__':

    use_kan = True

    device = 'cuda'
    training_dataset = torch.from_numpy(np.load('data/training_data.pkl', allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('data/testing_data.pkl', allow_pickle=True))
    print(f'Using the Device:', device)

    if use_kan:
        model = KANeRF().to(device) # NerfModel(hidden_dim=256).to(device)
    else:
        model = NerfModel(hidden_dim=256).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    data_loader = DataLoader(training_dataset, batch_size=1024, shuffle=True)
    train(model, model_optimizer, scheduler, data_loader, nb_epochs=16, device=device, hn=2, hf=6, nb_bins=192, H=400,
          W=400, use_kan=use_kan)
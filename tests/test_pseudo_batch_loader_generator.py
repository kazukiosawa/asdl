import torch
from torch.utils.data import DataLoader
from asdfghjkl import PseudoBatchLoaderGenerator


# create a base dataloader
dataset_size = 10
x_all = torch.tensor(range(dataset_size))
dataset = torch.utils.data.TensorDataset(x_all)
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True)

# create a pseudo-batch loader generator
pseudo_batch_loader_generator = PseudoBatchLoaderGenerator(data_loader, 5)

for i, loader in enumerate(pseudo_batch_loader_generator):
    print(f'pseudo-batch at step {i}')
    print(list(loader))

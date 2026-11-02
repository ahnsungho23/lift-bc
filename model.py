import torch
from torch import nn
from torch.utils.data import Dataset


class DemoDataset(Dataset):
    def __init__(self, obs, actions):
        self.obs = torch.from_numpy(obs).float()
        self.actions = torch.from_numpy(actions).float()

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]


class BCPolicy(nn.Module):
    def __init__(self, obs_dim=19, act_dim=7, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x):
        return self.net(x)
import torch
import torch.nn as nn

class SimpleDiffusionNet(nn.Module):
    def __init__(self, t_dim=16):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim)
        )
        
        self.net = nn.Sequential(
            nn.Linear(2 + t_dim + 1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, t, y):
        t_emb = self.time_mlp(t.float())
        y_emb = y.view(-1, 1).float()
        x_in = torch.cat([x, t_emb, y_emb], dim=1)
        return self.net(x_in)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDiffusion(nn.Module):
    def __init__(self, input_dim=15, cond_dim=1, hidden_dim=256):
        super().__init__()
        # Time embedding to help the model identify the noise level
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main architecture: Input (x_t) + Property (y) + Time Embedding
        self.model = nn.Sequential(
            nn.Linear(input_dim + cond_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, y):
        # x: noisy data, t: timestep, y: target property (e.g., solubility)
        t_emb = self.time_mlp(t.unsqueeze(-1).float())
        combined = torch.cat([x, t_emb, y], dim=-1)
        return self.model(combined)
import torch
import torch.nn as nn

class FiLMHead(nn.Module):
    def __init__(self,
                 pose_dim=64,
                 hidden_dim=512,
                 token_dim=768):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim * 2)
        )

    def forward(self, pose_feat):
        out = self.mlp(pose_feat)

        gamma, beta = out.chunk(2, dim=-1)

        return gamma.unsqueeze(1), beta.unsqueeze(1)
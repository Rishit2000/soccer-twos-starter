import torch
import torch.nn as nn

class PPOGeneralistNetwork(nn.Module):
    """
    A unified network architecture. At inference time, we only use the actor.
    """
    def __init__(self, obs_dim, action_branches):
        super().__init__()
        
        hidden_dim = 256
        self.action_branches = action_branches
        total_action_dim = sum(action_branches)

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, total_action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_actor(self, obs):
        """Used at inference time to get deterministic actions."""
        logits = self.actor(obs)
        branch_logits = torch.split(logits, self.action_branches, dim=-1)
        actions = [torch.argmax(b, dim=-1) for b in branch_logits]
        return torch.stack(actions, dim=-1)

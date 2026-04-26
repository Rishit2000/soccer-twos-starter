import os
import numpy as np
import torch
from soccer_twos import AgentInterface
from .model import PPOGeneralistNetwork

class TeamAgent(AgentInterface):
    """
    Shared-policy MAPPO Agent. 
    Both agents use the exact same brain.
    """
    def __init__(self, env):
        self.name = "The_Chosen_One"
        self.action_branches = env.action_space.nvec.tolist()
        self.obs_dim = env.observation_space.shape[0]

        self.model = PPOGeneralistNetwork(self.obs_dim, self.action_branches)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "shared_checkpoint.pth")

        if os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            print("WARNING: Checkpoints not found. Agents will act randomly.")

        self.model.eval()

    def act(self, observation):
        """
        Args:
            observation: dict {player_id: obs_array}
        Returns:
            actions: dict {player_id: action_array}
        """
        actions = {}

        for player_id, obs in observation.items():
            state_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            
            with torch.no_grad():
                action_tensor = self.model.forward_actor(state_tensor)

            actions[player_id] = action_tensor.squeeze(0).tolist()
            
        return actions
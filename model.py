import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldModel(nn.Module):
    def __init__(self, obs_dim, hidden_dim, compressed_dim, action_dim):
        """
        In the constructor, define all the layers and components of the network.
        """
        super(WorldModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, compressed_dim),
            nn.ReLU(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(compressed_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.obs_predictor = nn.Sequential(
            nn.Linear(hidden_dim, obs_dim),
        )

        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )

        self.done_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )


    def forward(self, obs, action):
        encoded_state = self.encoder(obs)

        predictor_input = torch.cat([encoded_state, action.unsqueeze(-1)], dim=-1)

        out = self.predictor(predictor_input)
        next_obs = self.obs_predictor(out)
        reward = self.reward_predictor(out).squeeze(-1)
        done = self.done_predictor(out).squeeze(-1)

        return next_obs, reward, done


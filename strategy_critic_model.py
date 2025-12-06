import torch.nn as nn

class StrategyCriticModel(nn.Module):
    def __init__(self, strategy_params_dim, hidden_dim):
        super(StrategyCriticModel, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(strategy_params_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, strategy_params):
        value = self.critic(strategy_params)
        return value

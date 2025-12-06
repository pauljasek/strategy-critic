import torch.nn as nn

class PolicyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(PolicyModel, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        logits = self.actor(state)
        return logits

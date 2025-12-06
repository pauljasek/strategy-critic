import torch
import torch.nn.functional as F

class Policy:
    def __init__(self, encoder_model, policy_model):
        self.policy_model = policy_model
        self.encoder_model = encoder_model
    
    def __call__(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        encoded_state = self.encoder_model.encode(obs)
        action_logits = self.policy_model(encoded_state)
        action_probs = F.softmax(action_logits, dim=-1)
        categorical_dist = torch.distributions.Categorical(probs=action_probs)
        return int(categorical_dist.sample().detach())

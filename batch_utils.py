import torch

def create_batch(trajectories, shuffle=True):
    batch = {
        'obs': torch.cat([data['obs'] for data in trajectories]),
        'next_obs': torch.cat([data['next_obs'] for data in trajectories]),
        'action': torch.cat([data['action'] for data in trajectories]),
        'reward': torch.cat([data['reward'] for data in trajectories]),
        'terminated': torch.cat([data['terminated'] for data in trajectories]),
        'truncated': torch.cat([data['truncated'] for data in trajectories]),
        # Note: ignoring 'infos' for now
    }
    if shuffle:
        batch = shuffle_batch(batch)
    return batch

def shuffle_batch(batch):
    batch_len = batch['obs'].shape[0]
    shuffled_indices = torch.randperm(batch_len)
    shuffled_batch = {
        key: batch[key][shuffled_indices] for key in batch
    }

    return shuffled_batch

def create_minibatches(batch, minibatch_size, device="cuda"):
    minibatches = torch.split(batch['obs'], minibatch_size, dim=0)
    split_batch = {
        key: torch.split(batch[key].to(device), minibatch_size, dim=0) for key in batch
    }
    num_minibatches = len(split_batch['obs'])
    minibatches = [
        {key: split_batch[key][i] for key in batch} 
        for i in range(num_minibatches)
    ]
    return minibatches
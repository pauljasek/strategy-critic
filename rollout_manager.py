import multiprocessing
import torch
import numpy as np

def rollout_worker_fn(args):
    env, policy = args
    trajectory = []
    obs, info = env.reset()
    while True:
        if policy is not None:
            action = policy(obs)
        else:
            action = env.action_space.sample() # TODO: Replace with policy
        next_obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append(
            {
                'obs': obs,
                'next_obs': next_obs,
                'action': action,
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                'info': info,
            }
        )

        obs = next_obs

        if terminated or truncated:
            observation, info = env.reset()
            break
        
    trajectory = {
        'obs': np.array([data['obs'] for data in trajectory], dtype=np.float32),
        'next_obs': np.array([data['next_obs'] for data in trajectory], dtype=np.float32),
        'action': np.array([data['action'] for data in trajectory], dtype=np.float32),
        'reward': np.array([data['reward'] for data in trajectory], dtype=np.float32),
        'terminated': np.array([data['terminated'] for data in trajectory], dtype=np.float32),
        'truncated': np.array([data['terminated'] for data in trajectory], dtype=np.float32),
        'info': [data['info'] for data in trajectory]
    }

    trajectory['obs'] = torch.tensor(trajectory['obs'], dtype=torch.float32)
    trajectory['next_obs'] = torch.tensor(trajectory['next_obs'], dtype=torch.float32)
    trajectory['action'] = torch.tensor(trajectory['action'], dtype=torch.float32)
    trajectory['reward'] = torch.tensor(trajectory['reward'], dtype=torch.float32)
    trajectory['terminated'] = torch.tensor(trajectory['terminated'], dtype=torch.float32)
    trajectory['truncated'] = torch.tensor(trajectory['truncated'], dtype=torch.float32)

    return trajectory


class RolloutManager:
    def __init__(self, num_rollout_workers):
        self.num_rollout_workers = num_rollout_workers
        self.pool = multiprocessing.Pool(
            processes=self.num_rollout_workers
        )

    def rollout(self, env, policy=None):
        trajectories = self.pool.map(rollout_worker_fn, [(env, policy) for i in range(self.num_rollout_workers)])
        return trajectories

    def shutdown(self):
        self.pool.close()
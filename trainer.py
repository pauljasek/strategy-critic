import argparse

import torch

from model import WorldModel
from rollout_manager import RolloutManager
from env import get_cartpole_env
from batch_utils import create_batch, create_minibatches

def train():
    parser = argparse.ArgumentParser(description="Strategy Critic training script")
    parser.add_argument("--iters", type=int, default=10, help="The number of training iterations to run")
    parser.add_argument("--episodes-per-worker", type=int, default=1, help="The number of episodes for each worker to rollout")
    parser.add_argument("--minibatch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--num-sgd-iters", type=int, default=10, help="Number of SGD iters")
    parser.add_argument("--learning-rate", '-lr', type=float, default=1e-3, help="SGD learning rate")

    args = parser.parse_args()

    env = get_cartpole_env()
    
    world_model = WorldModel(
        obs_dim=4,
        hidden_dim=64,
        compressed_dim=8,
        action_dim=2,
    )

    rollout_manager = RolloutManager(
        num_rollout_workers=16
    )

    optimizer = torch.optim.Adam(world_model.parameters(), lr=args.learning_rate)

    for iter_num in range(args.iters):
        trajectories = []
        for episode_num in range(args.episodes_per_worker):
            trajectories.extend(rollout_manager.rollout(None, env))
        
        batch = create_batch(trajectories, shuffle=True)
        for sgd_iter in range(args.num_sgd_iters):
            for minibatch in create_minibatches(batch, args.minibatch_size):
                pass


if __name__ == "__main__":
    train()
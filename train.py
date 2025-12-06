import argparse
import tqdm

import torch
import torch.nn.functional as F

from world_model import WorldModel
from policy_model import PolicyModel
from strategy_critic_model import StrategyCriticModel

from policy import Policy

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    env = get_cartpole_env()
    
    world_model = WorldModel(
        obs_dim=4,
        hidden_dim=64,
        compressed_dim=8,
        action_dim=1,
    )

    policy_model = PolicyModel(
        input_dim=8,
        hidden_dim=8,
        action_dim=2,
    )
    
    strategy_critic_model = StrategyCriticModel(
        strategy_params_dim=64+8+16+2,
        hidden_dim=128,
    ).to(device)

    rollout_manager = RolloutManager(
        num_rollout_workers=16
    )

    optimizer = torch.optim.Adam(
        [
        {'params': world_model.parameters(), 'lr': args.learning_rate},
        {'params': policy_model.parameters(), 'lr': args.learning_rate},
        {'params': strategy_critic_model.parameters(), 'lr': args.learning_rate},
        ]
    )

    for iter_num in tqdm.tqdm(list(range(args.iters))):
        policy = Policy(
            policy_model=policy_model.to("cpu"),
            encoder_model=world_model.to("cpu"),
        )

        trajectories = []
        for episode_num in range(args.episodes_per_worker):
            trajectories.extend(rollout_manager.rollout(env, policy=policy))

        world_model = world_model.to(device)
        policy_model = policy_model.to(device)
        
        batch = create_batch(trajectories, shuffle=True)
        world_model_losses = []
        for sgd_iter in range(args.num_sgd_iters):
            for minibatch in create_minibatches(batch, args.minibatch_size, device=device):
                obs = minibatch['obs']
                action = minibatch['action']
                next_obs = minibatch['next_obs']
                reward = minibatch['reward']
                done = minibatch['terminated']

                pred_next_obs, pred_reward, pred_done = world_model.forward(obs, action)

                obs_loss = F.mse_loss(pred_next_obs, next_obs)
                reward_loss = F.mse_loss(pred_reward, reward)
                done_loss = F.mse_loss(pred_done, done)

                world_model_loss = torch.mean(obs_loss + reward_loss + done_loss)

                optimizer.zero_grad()
                world_model_loss.backward()
                optimizer.step()

                world_model_losses.append(float(world_model_loss.detach()))
        
        print('World Model Loss:', sum(world_model_losses)/len(world_model_losses))


if __name__ == "__main__":
    train()
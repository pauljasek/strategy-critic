import argparse
import tqdm
import copy

import torch
import torch.nn.functional as F

from world_model import WorldModel
from policy_model import PolicyModel
from strategy_critic_model import StrategyCriticModel

from policy import Policy

from rollout_manager import RolloutManager
from env import get_cartpole_env
from batch_utils import create_batch, shuffle_batch, create_minibatches

def get_policy_parameters(policy_model):
    return torch.cat([param.view(-1) if len(param.shape) == 2 else param for param in policy_model.parameters()], dim=-1)


def train():
    parser = argparse.ArgumentParser(description="Strategy Critic training script")
    parser.add_argument("--iters", type=int, default=10, help="The number of training iterations to run")
    parser.add_argument("--num-rollout-workers", type=int, default=16, help="The number of rollout workers")
    parser.add_argument("--episodes-per-worker", type=int, default=1, help="The number of episodes for each worker to rollout")
    parser.add_argument("--minibatch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--num-sgd-iters", type=int, default=30, help="Number of SGD iters")
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
    
    strategy_critic_model = StrategyCriticModel(
        strategy_params_dim=64+8+16+2,
        hidden_dim=128,
    ).to(device)

    rollout_manager = RolloutManager(
        num_rollout_workers=args.num_rollout_workers
    )

    world_model_optimizer = torch.optim.Adam(world_model.parameters(), lr=args.learning_rate)
    strategy_critic_optimizer = torch.optim.Adam(strategy_critic_model.parameters(), lr=args.learning_rate)

    for iter_num in tqdm.tqdm(list(range(args.iters))):
        num_policies = args.num_rollout_workers
        policies = []
        
        world_model_cpu = world_model.to("cpu")

        for i in range(num_policies):
            policy_model = PolicyModel(
                input_dim=8,
                hidden_dim=8,
                action_dim=2,
            ).to(device)

            policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.learning_rate)
            
            policy_losses = []
            for sgd_iter in range(args.num_sgd_iters):
                policy_parameters = get_policy_parameters(policy_model)
                pred_return = strategy_critic_model(policy_parameters)
                policy_loss = -pred_return

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                policy_losses.append(float(policy_loss.detach()))

            print('Policy Loss:', sum(policy_losses)/len(policy_losses))            

            policy_model = policy_model.to("cpu")
            policy = Policy(
                policy_model=policy_model,
                encoder_model=world_model_cpu,
            )
            policies.append(policy)

        world_model = world_model.to("cpu")

        trajectories = []
        total_returns = []
        for episode_num in range(args.episodes_per_worker):
            rollout_outputs = rollout_manager.rollout(env, policy=policy)
            trajectories.extend([output[0] for output in rollout_outputs])
            total_returns.extend([output[1] for output in rollout_outputs])

        world_model = world_model.to(device)

        batch = create_batch(trajectories, shuffle=True)
        returns_batch = torch.tensor(total_returns, dtype=torch.float32).to(device)

        policy_parameters_batch = torch.stack([get_policy_parameters(policy.policy_model).to(device) for policy in policies], dim=0)
        strategy_critic_losses = []
        for sgd_iter in range(args.num_sgd_iters):
            pred_returns = strategy_critic_model(policy_parameters_batch)
            strategy_critic_loss = torch.mean(F.mse_loss(pred_returns, returns_batch))

            strategy_critic_optimizer.zero_grad()
            strategy_critic_loss.backward()
            strategy_critic_optimizer.step()

            strategy_critic_losses.append(float(strategy_critic_loss.detach()))

        print('Strategy Critic Loss:', sum(strategy_critic_losses)/len(strategy_critic_losses))            

        world_model_losses = []
        for sgd_iter in range(args.num_sgd_iters):
            batch = shuffle_batch(batch)
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

                world_model_optimizer.zero_grad()
                world_model_loss.backward()
                world_model_optimizer.step()

                world_model_losses.append(float(world_model_loss.detach()))

        print('World Model Loss:', sum(world_model_losses)/len(world_model_losses))

        print('Mean return:', float(torch.mean(returns_batch).detach()))

if __name__ == "__main__":
    train()
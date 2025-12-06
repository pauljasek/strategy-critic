import multiprocessing

def rollout_worker_fn(policy, env):
    trajectory = []
    obs, info = env.reset()
    while True:
        action = env.action_space.sample() # TODO: Replace with policy
        next_obs, reward, terminated, truncated, info = env.step(action)

        trajectory.append(
            (obs, next_obs, action, reward, terminated, truncated, info)
        )

        obs = next_obs

        if terminated or truncated:
            observation, info = env.reset()
            return trajectory


class RolloutManager:
    def __init__(self, num_rollout_workers):
        self.num_rollout_workers = num_rollout_workers
        self.pool = multiprocessing.Pool(
            processes=self.num_rollout_workers
        )

    def rollout(self, policy, env):
        trajectories = self.pool.map(rollout_worker_fn, (policy, env))
        return trajectories

    def shutdown(self):
        self.pool.close()
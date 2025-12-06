import gymnasium as gym

def get_cartpole_env():
    return gym.make("CartPole-v1")

if __name__ == "__main__":
    env = get_cartpole_env()
    observation, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation, reward, action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()

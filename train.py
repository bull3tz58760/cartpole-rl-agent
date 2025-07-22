import gym
import torch
import torch.optim as optim
from policy import PolicyNetwork
from utils import discount_rewards
import numpy as np

env = gym.make("CartPole-v1")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = PolicyNetwork(obs_size, 128, n_actions)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)

def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy_net(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

for episode in range(1000):
    state = env.reset()
    log_probs = []
    rewards = []
    total_reward = 0

    while True:
        action, log_prob = select_action(state)
        next_state, reward, done, _ = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
        state = next_state

        if done:
            break

    discounted_rewards = discount_rewards(rewards)
    loss = []
    for log_prob, reward in zip(log_probs, discounted_rewards):
        loss.append(-log_prob * reward)
    loss = torch.stack(loss).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 10 == 0:
        print(f"Episode {episode}\tTotal reward: {total_reward}")


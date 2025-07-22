import numpy as np

def discount_rewards(rewards, gamma=0.99):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_total = 0
    for i in reversed(range(len(rewards))):
        running_total = rewards[i] + gamma * running_total
        discounted[i] = running_total
    return discounted


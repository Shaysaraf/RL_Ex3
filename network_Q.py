import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for OpenMP DLL error

import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Load environment
env = gym.make('FrozenLake-v1', is_slippery=False)  # Use deterministic env for better learning

# Define the neural network mapping 1x16 one-hot vector to a vector of 4 Q-values
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc(x)

# Instantiate model, loss, and optimizer
model = QNetwork()
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

# Set learning parameters
gamma = 0.99
epsilon = 1.0
num_episodes = 2000

jList = []
rList = []


for i in range(num_episodes):
    s = env.reset()[0]
    rAll = 0
    stop = False
    j = 0

    while j < 99:
        j += 1

        s_vec = torch.zeros(1, 16)
        s_vec[0][s] = 1.0

        with torch.no_grad():
            Q = model(s_vec)
        _, a = torch.max(Q, 1)

        # ε-greedy
        if np.random.rand() < epsilon:
            a[0] = env.action_space.sample()

        s1, reward, stop, _, _ = env.step(a.item())

        s1_vec = torch.zeros(1, 16)
        s1_vec[0][s1] = 1.0
        with torch.no_grad():
            Q1 = model(s1_vec)
        maxQ1 = torch.max(Q1).item()

        Q_target = Q.clone().detach()
        if stop:
            Q_target[0][a] = reward
        else:
            Q_target[0][a] = reward + gamma * maxQ1

        output = model(s_vec)
        Q_pred = output[0, a]
        loss = loss_fn(Q_pred, Q_target[0, a])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rAll += reward
        s = s1
        if stop:
            break

    epsilon = max(0.01, epsilon * 0.995)  # decay epsilon
    jList.append(j)
    rList.append(rAll)

# Report performance
success_rate = sum(rList) / num_episodes
print(f"Success rate: {100.0 * success_rate:.2f}%")



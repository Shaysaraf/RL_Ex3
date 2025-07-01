import gym
import numpy as np

env = gym.make('FrozenLake-v1', is_slippery=False)

Q = np.zeros([env.observation_space.n, env.action_space.n])

ALPHA = 0.8
GAMMA = 0.95
EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY = 0.995
num_episodes = 2000
rList = []

for i in range(num_episodes):
    s, _ = env.reset()
    stop = False
    rAll = 0

    while not stop:
        if np.random.rand() < EPSILON:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s])

        s1, reward, done, truncated, _ = env.step(a)
        stop = done or truncated

        Q[s, a] += ALPHA * (reward + GAMMA * np.max(Q[s1]) - Q[s, a])
        rAll += reward
        s = s1

    rList.append(rAll)
    EPSILON = max(MIN_EPSILON, EPSILON * DECAY)

print(f"Success rate: {100.0 * sum(rList) / num_episodes:.2f}%")
print("Final Q-table:")
print(Q)


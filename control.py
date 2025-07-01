"""
CS 229 Machine Learning, Fall 2017
Problem Set 4
Question: Reinforcement Learning: The inverted pendulum
Author: Sanyam Mehra, sanyam@stanford.edu
"""
from __future__ import division, print_function
from cart_pole import CartPole, Physics
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

# Simulation parameters
pause_time = 0.0001
min_trial_length_to_start_display = 100
display_started = min_trial_length_to_start_display == 0

NUM_STATES = 163
NUM_ACTIONS = 2
GAMMA = 0.995
TOLERANCE = 0.01
NO_LEARNING_THRESHOLD = 20

# Time cycle of the simulation
time = 0

# Bookkeeping variables
time_steps_to_failure = []
num_failures = 0
time_at_start_of_current_trial = 0

max_failures = 500

# Initialize cart pole
cart_pole = CartPole(Physics())

# Initial continuous state tuple
x, x_dot, theta, theta_dot = 0.0, 0.0, 0.0, 0.0
state_tuple = (x, x_dot, theta, theta_dot)

# Get initial discrete state
state = cart_pole.get_state(state_tuple)

###### BEGIN YOUR CODE ######
# Initialization of value function, transition counts, probabilities, rewards

# Initialize value function to small random values between 0 and 0.1
V = np.random.uniform(low=0.0, high=0.1, size=NUM_STATES)

# Transition counts: counts of transitions from state s, action a, to next_state
transition_counts = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES), dtype=int)

# Accumulated rewards per state (sum of rewards observed at state s)
reward_sums = np.zeros(NUM_STATES)

# Number of times each state was visited
state_visits = np.zeros(NUM_STATES, dtype=int)

# Transition probabilities (initialized uniformly)
T = np.ones((NUM_STATES, NUM_ACTIONS, NUM_STATES)) / NUM_STATES

# Reward per state initialized to zero
R = np.zeros(NUM_STATES)
###### END YOUR CODE ######

consecutive_no_learning_trials = 0

while consecutive_no_learning_trials < NO_LEARNING_THRESHOLD:

    ###### BEGIN YOUR CODE ######
    # Choose action greedily with respect to current value function and model
    action_values = np.zeros(NUM_ACTIONS)
    for a in range(NUM_ACTIONS):
        action_values[a] = np.sum(T[state, a, :] * (R + GAMMA * V))
    action = np.argmax(action_values)
    ###### END YOUR CODE ######

    # Simulate one time step
    state_tuple = cart_pole.simulate(action, state_tuple)
    time += 1
    new_state = cart_pole.get_state(state_tuple)

    # Reward: -1 if pole fell or out of bounds, else 0
    R_observed = -1 if new_state == NUM_STATES - 1 else 0

    ###### BEGIN YOUR CODE ######
    # Update transition counts and rewards
    transition_counts[state, action, new_state] += 1
    reward_sums[new_state] += R_observed
    state_visits[new_state] += 1
    ###### END YOUR CODE ######

    if new_state == NUM_STATES - 1:
        ###### BEGIN YOUR CODE ######
        # Update transition probability matrix T
        for s in range(NUM_STATES):
            for a in range(NUM_ACTIONS):
                total = np.sum(transition_counts[s, a, :])
                if total > 0:
                    T[s, a, :] = transition_counts[s, a, :] / total
                else:
                    T[s, a, :] = np.ones(NUM_STATES) / NUM_STATES


        # Update average rewards R per state
        for s in range(NUM_STATES):
            if state_visits[s] > 0:
                R[s] = reward_sums[s] / state_visits[s]
            else:
                R[s] = 0.0
        ###### END YOUR CODE ######

        ###### BEGIN YOUR CODE ######
        # Value iteration
        iteration = 0
        max_diff = float('inf')
        converged_in_one_iteration = False

        while max_diff > TOLERANCE:
            V_new = np.zeros_like(V)
            for s in range(NUM_STATES):
                if s == NUM_STATES - 1:
                    V_new[s] = R[s]
                else:
                    action_values = np.zeros(NUM_ACTIONS)
                    for a in range(NUM_ACTIONS):
                        action_values[a] = np.sum(T[s, a, :] * (R + GAMMA * V))
                    V_new[s] = np.max(action_values)

            max_diff = np.max(np.abs(V_new - V))
            V = V_new
            iteration += 1

            if iteration == 1 and max_diff < TOLERANCE:
                converged_in_one_iteration = True

        # Update convergence counter
        if converged_in_one_iteration:
            consecutive_no_learning_trials += 1
        else:
            consecutive_no_learning_trials = 0
        ###### END YOUR CODE ######

    # Handle pole fall
    if new_state == NUM_STATES - 1:
        num_failures += 1
        if num_failures >= max_failures:
            break
        print('[INFO] Failure number {}'.format(num_failures))
        time_steps_to_failure.append(time - time_at_start_of_current_trial)
        time_at_start_of_current_trial = time

        if time_steps_to_failure[num_failures - 1] > min_trial_length_to_start_display:
            display_started = 1

        # Reinitialize state randomly within bounds
        x = -1.1 + np.random.uniform() * 2.2
        x_dot, theta, theta_dot = 0.0, 0.0, 0.0
        state_tuple = (x, x_dot, theta, theta_dot)
        state = cart_pole.get_state(state_tuple)
    else:
        state = new_state

# Plot learning curve (log of time balanced vs. trial number)
log_tstf = np.log(np.array(time_steps_to_failure))
plt.plot(np.arange(len(time_steps_to_failure)), log_tstf, 'k')
window = 30
w = np.array([1/window for _ in range(window)])
weights = lfilter(w, 1, log_tstf)
x = np.arange(window//2, len(log_tstf) - window//2)
plt.plot(x, weights[window:len(log_tstf)], 'r--')
plt.xlabel('Num failures')
plt.ylabel('Num steps to failure')
plt.show()

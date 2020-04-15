import numpy as np
from collections import defaultdict

# Set a random seed to ease comparison between runs
np.random.seed(1)

# Function for priting a Q-table
def print_Q_table(Q_dictionary):
    print('\n\n--- Q-Table ---')
    for state in Q_dictionary.keys():
        print(state + ':\t', end='')
        for action in Q_dictionary[state].keys():
            print('{}= {:.2f} \t'.format(action, Q_dictionary[state][action]), end='')
        print('')

# Defines the environment and provides methods to interact with it
class Environment:
    def __init__(self):
        self.markov_process = {
            'docked-low': {
                'clean': (('docked-low', 'cleaning-low'), (0.8, 0.2), (-3, 1)),
                'wait': (('docked-low', 'docked-high'), (0.6, 0.4), (0, 0)),
            },
            'docked-high': {
                'clean': (('cleaning-high', 'cleaning-low'), (0.9, 0.1), (1, 1)),
                'wait': (('docked-high', ), (1, ), (0, ))
            },
            'cleaning-high': {
                'clean': (('cleaning-high', 'cleaning-low'), (0.8, 0.2), (1, 1)),
                'dock': (('docked-high', 'docked-low'), (0.9, 0.1), (0, 0))
            },
            'cleaning-low': {
                'clean': (('docked-low', 'cleaning-low'), (0.6, 0.4), (-3, 1)),
                'dock': (('docked-low', 'docked-low'), (0.8, 0.2), (0, -3))
            }
        }
        self.current_state = np.random.choice(list(self.markov_process.keys()), 1)[0]

    def available_actions(self):
        return list(self.markov_process[self.current_state].keys())

    def step(self, action):
        state_probability_and_rewards = self.markov_process[self.current_state][action]
        next_state_index = np.random.choice(list(range(len(state_probability_and_rewards[0]))), p=state_probability_and_rewards[1])

        next_state = state_probability_and_rewards[0][next_state_index]
        reward = state_probability_and_rewards[2][next_state_index]

        self.current_state = next_state

        return (next_state, reward)

# Instantiate the environment
env = Environment()

# Select an action based on epsilon-greedy algorithm
def episilon_greedy_choice(expected_returns, actions, episode=0, epsilon=1, epsilon_decay=0.999, epsilon_min=0.001):
    if len(expected_returns.keys()) == 0:
        return np.random.choice(actions)

    # Compute decaued epsilon
    epsilon = max(epsilon * (epsilon_decay ** episode), epsilon_min)

    # Compute probabilities for each action
    expected_returns = list(expected_returns.keys())
    index_of_max = np.argmax(expected_returns)
    probability_of_max = (1 - epsilon) + (epsilon / len(actions))
    probability_of_others = epsilon / len(actions)
    probabilities = [probability_of_others for i in range(len(actions))]
    probabilities[index_of_max] = probability_of_max

    # Return the action based on the calculated probabilities
    return np.random.choice(actions, p=probabilities)

ALPHA = 0.001 # Learning rate

Q = defaultdict(lambda: defaultdict(lambda: 0))
episode_reward = []
for episode in range(1, 1000): # Number of episodes
    visited_states = []

    # Collect Data by Iterating over the environment
    observations = []
    for i in range(50): # Steps per episode
        state = env.current_state
        action = episilon_greedy_choice(Q[state], env.available_actions(), episode=episode)

        (next_state, reward) = env.step(action)
        observations += [(state, action, reward, next_state)]

    # Update the Q Table with the information from the episode
    for index, obs in enumerate(observations):
        state, action, reward, next_state = obs
        discounted_return = sum(list(zip(*observations[index:]))[2])

        Q[state][action] = Q[state][action] + ALPHA * (discounted_return - Q[state][action])

    episode_reward += [sum(list(zip(*observations))[2])]
    # Print results
    print('\rEpisode: {}\tReward: {:.2f}'.format(episode, episode_reward[-1]), end='')
    if episode % 100 == 0:
        print('\rEpisode: {}\tAverage reward: {:.2f}'.format(episode, sum(episode_reward[-100:]) / 100))

print_Q_table(Q)

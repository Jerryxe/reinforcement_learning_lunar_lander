# Lunar Lander
import gym
import numpy as np
from collections import deque
from collections import namedtuple
import matplotlib.pyplot as plt
import random
import torch
import time

# Main reference of the code is from below Pytorch DQN tutorial, expecially the efficient implementation of Experience Replay and usage of namedtuple
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

torch.set_default_tensor_type('torch.FloatTensor')
env = gym.make('LunarLander-v2')
Experience = namedtuple('Transition', ('current_state', 'action', 'next_state', 'reward'))
time_begin = time.time()  # To measure time elapsed


# NN Model with 3 hidden layers, and each layer has 64 nodes.
# There will be 2 neural nets, one is for policy, and one is for target, policy net's weight keeps updating at every step,
# while target net's gets updated from policy net regularly, using either:
# 1. Directly copy from policy net in every couple of steps (usually few thousands, or roughly ~10 episodes)
# 2. Using soft update in every step.
class NeuralNet(torch.nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        self.input = torch.nn.Linear(8, 64)
        self.input.weight.data.normal_(0, 0.1)
        self.hidden1 = torch.nn.Linear(64, 64)
        self.hidden1.weight.data.normal_(0, 0.1)
        self.hidden2 = torch.nn.Linear(64, 64)
        self.hidden2.weight.data.normal_(0, 0.1)
        self.hidden3 = torch.nn.Linear(64, 64)
        self.hidden3.weight.data.normal_(0, 0.1)
        self.output = torch.nn.Linear(64, 4)
        self.output.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.input(x)
        x = torch.nn.functional.relu(x)
        x = self.hidden1(x)
        x = torch.nn.functional.relu(x)
        x = self.hidden2(x)
        x = torch.nn.functional.relu(x)
        x = self.hidden3(x)
        x = torch.nn.functional.relu(x)
        return self.output(x)


# Experience Replay, the capacity is controlled at creation. Whenever the ER is full, oldest experience will be removed.
# The ER saves information for each step: Current state, action taken, returned next state and step reward
class ExperienceReplay(object):

    def __init__(self, max_size):
        self.max_size = max_size
        self.experiences = []
        self.current_pos = 0

    def push(self, *args):
        if len(self.experiences) < self.max_size:
            self.experiences.append(None)
        self.experiences[self.current_pos] = Experience(*args)
        self.current_pos = (self.current_pos + 1) % self.max_size


# Policy Evaluation
# Epsilon-greedy algorithm, at the first pure_exploration steps, always use randomized action,
# then start from epsilon_begin, with exponential decay to epsilon_end
def evaluate(state, pure_exploration, episode):
    if episode >= pure_exploration and np.random.rand() > epsilon_end + (epsilon_begin - epsilon_end) * np.exp(
            -1.0 * (episode - pure_exploration) / epsilon_episodes):
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[np.random.randint(4)]], dtype=torch.long)


# Optimization
# At each step, we use backprop and gradient to update policy net's weights
def optimize():
    # Randomly fetch some steps from the experiences, this is to de-correlate experience,
    # since a bad episode could last few hundred of steps.
    sampled_experiences = random.sample(experiences.experiences, batch_size)
    sampled_experiences = Experience(*zip(*sampled_experiences))
    non_terminal_states_index = torch.tensor(tuple(map(lambda x: x is not None, sampled_experiences.next_state)),
                                             dtype=torch.uint8)
    sampled_next_states_non_terminal = torch.cat([x for x in sampled_experiences.next_state if x is not None])
    sampled_current_states = torch.cat(sampled_experiences.current_state)
    sampled_actions = torch.cat(sampled_experiences.action)
    sampled_rewards = torch.cat(sampled_experiences.reward)

    # Compute Q value of each state-action pair
    state_action_values = policy_net(sampled_current_states).gather(1, sampled_actions)

    # Compute Q value and V value for next state
    next_state_values = torch.zeros(batch_size).to(torch.float32)
    next_state_values[non_terminal_states_index] = target_net(sampled_next_states_non_terminal).max(1)[0].detach()

    # Compute expected Q values using Bellman Equation
    expected_state_action_values = (next_state_values * gamma) + sampled_rewards

    # Loss function: Huber L1 or MSE L2
    loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize model using gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Plot total rewards
def plot_rewards(episode_rewards):
    episode_rewards = torch.tensor(episode_rewards, dtype=torch.float)
    plt.figure(1)
    plt.clf()
    plt.title('Training Period')
    plt.xlabel('# of Episodes')
    plt.ylabel('Total Rewards Per Episode')
    episode1, = plt.plot(episode_rewards.numpy(), linestyle='', color='black', marker='o', markersize=1)
    # Take 10 episode averages and plot them too
    if len(episode_rewards) >= 100:
        means = episode_rewards.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        episode100, = plt.plot(means.numpy(), linestyle='-', linewidth=1, color='green')
        plt.legend([episode1, episode100], ['Single Episode', '100 Episodes Avg'], loc='lower right')
    plt.axhline(100, color='r', linestyle='--', linewidth=1)
    plt.axhline(200, color='r', linestyle='--', linewidth=1)
    plt.pause(0.001)


# Variables
episode_length = deque()
episode_rewards = deque()
episode_time = deque()
policy_net = NeuralNet()
target_net = NeuralNet()
target_net.load_state_dict(policy_net.state_dict())
optimizer = torch.optim.Adam(policy_net.parameters(), 0.0001)

# Hyper Parameters
pure_exploration = 1000  # number of episodes that we do pure exploration using random selection
epsilon_begin = 1  # After the pure exploration, we start to use epsilon-greedy, epsilon decays
epsilon_end = 0.05  # Epsilon decays until
epsilon_episodes = 500  # number of episodes to decay, the longer the slower
batch_size = 64  # number of experience every time we sample from the Experience Replay
experiences = ExperienceReplay(100000)  # Set very large size of experience storage to de-correlate experience
gamma = 0.99  # Discount rate
update_freq = 2000  # Directly copy weights from policy net to target net every # of steps
num_episodes = 5000  # numbers of episodes to train
max_episode_length = 1000  # Limit number of steps for each episode

# Train Model
def train_model():
    k = 0
    for i in range(num_episodes):
        # Initialization
        current_state = env.reset()
        current_state = torch.from_numpy(current_state).to(torch.float32).view(1, -1)
        total_rewards = 0
        j = 0
        # steps = deque()
        while True:
            j += 1
            k += 1

            # Move one step
            action = evaluate(current_state, pure_exploration, i)
            next_state, reward, done, info = env.step(action.item())
            next_state = torch.from_numpy(next_state).to(torch.float32).view(1, -1)
            total_rewards += reward
            reward = torch.tensor([reward], dtype=torch.float32)
            if done or j >= max_episode_length:
                next_state = None

            # Save history for experiemcne replay
            experiences.push(current_state, action, next_state, reward)
            current_state = next_state

            # Optimize weights and soft update if needed
            if i >= pure_exploration and len(experiences.experiences) >= batch_size:
                optimize()

            # Quit condition
            if done or j >= max_episode_length:
                episode_length.append(j)
                episode_rewards.append(total_rewards)
                plot_rewards(episode_rewards)
                break

            # Update the target net
            if k % update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Record time
        episode_time.append(time.time() - time_begin)
        if i % 10 == 0 and i >= 100:
            print('Finished: ', i, 'Episode, Second Passed: ', np.round(episode_time[i], 1),
                  ', Past 100 Episodes Avg Score:', np.round(sum(list(episode_rewards)[-100:]) / 100, 1))


def take_action(state, neural_net):
    with torch.no_grad():
        return neural_net(state).max(1)[1].view(1, 1)

def test_model(neural_net):
    total_rewards_list = []
    for i in range(100):
        current_state = env.reset()
        current_state = torch.from_numpy(current_state).to(torch.float32).view(1, -1)
        total_rewards = 0
        while True:
            action = take_action(current_state, neural_net)
            next_state, reward, done, info = env.step(action.item())
            next_state = torch.from_numpy(next_state).to(torch.float32).view(1, -1)
            total_rewards += reward
            current_state = next_state
            if done:
                total_rewards_list.append(total_rewards)
                break
    print('Avg score for 100 test episodes: ', round(sum(total_rewards_list) / 100, 0))
    plt.figure(2)
    plt.clf()
    plt.title('Test for 100 Episodes after Training')
    plt.xlabel('# of Episodes')
    plt.ylabel('Total Rewards Per Episode')
    plt.plot(total_rewards_list, marker='o', markersize=3, linestyle='', color='black')
    plt.axhline(sum(total_rewards_list)/100, color='r', linestyle='--')
    plt.text(-1, sum(total_rewards_list)/100-15, 'Average Score: ' + str(int(sum(total_rewards_list) / 100)), color='red')

def plot_epsilon():
    eps_period1 = np.ones(pure_exploration)
    eps_period2 = np.arange(num_episodes - pure_exploration)
    eps_period2 = epsilon_end + (epsilon_begin - epsilon_end) * np.exp(-1.0 * eps_period2 / epsilon_episodes)
    eps_all = np.append(eps_period1, eps_period2)
    plt.figure(3)
    plt.clf()
    plt.title('Epsilon Trend Line')
    plt.xlabel('# of Episodes')
    plt.ylabel('Epsilon')
    plt.plot(eps_all, linestyle='-', color='black', linewidth=1)
    plt.axvline(pure_exploration, color='red', linestyle='--', linewidth=1)
    plt.axhline(epsilon_end, color='blue', linestyle='--', linewidth=1)
    plt.text(2000, 0.99, 'Red Line: Epsilon Decay Begins', color='red')
    plt.text(2000, 0.94, 'Blue Line: Epsilon End Point', color='blue')

# Train Model
train_model()
# Test Model for 100 Episodes
test_model(target_net)
# Plot Epsilon Trendline
plot_epsilon()

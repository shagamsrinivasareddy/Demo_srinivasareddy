# drone_navigation.py

import numpy as np
import random
import time
from collections import deque
import plotly.graph_objects as go
from shapely.geometry import Point
from IPython.display import display, clear_output

# Hyperparameters
STATE_SIZE = (3, 10, 10)
ACTION_SPACE = 6
MEMORY_CAPACITY = 5000
BATCH_SIZE = 32
GAMMA = 0.95
LEARNING_RATE = 0.001
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.98
TRAIN_EPISODES = 500
MAX_STEPS = 100

# Define the Drone Simulation Environment
class DroneEnv:
    def __init__(self):
        self.position = np.array([5, 5, 5])
        self.obstacles = [Point(np.random.randint(0, 10), np.random.randint(0, 10), np.random.randint(0, 10)) for _ in range(10)]
        self.goal = np.array([9, 9, 9])
        self.path = [self.position.copy()]

    def reset(self):
        self.position = np.array([5, 5, 5])
        self.path = [self.position.copy()]
        return np.random.random(STATE_SIZE)

    def step(self, action):
        moves = [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]),
                 np.array([0, -1, 0]), np.array([0, 0, 1]), np.array([0, 0, -1])]
        new_position = self.position + moves[action]
        new_position = np.clip(new_position, 0, 9)

        # Collision check
        if any(obs.equals(Point(*new_position)) for obs in self.obstacles):
            reward = -10
        else:
            self.position = new_position
            self.path.append(self.position.copy())
            reward = -0.1

        # Goal check
        done = np.array_equal(self.position, self.goal)
        if done:
            reward = 20
        return np.random.random(STATE_SIZE), reward, done

# Deep Q-Network (DQN) Implementation
class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.epsilon = EPSILON
        self.q_table = np.zeros((10, 10, 10, ACTION_SPACE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, ACTION_SPACE - 1)
        return np.argmax(self.q_table[tuple(state)])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward + (GAMMA * np.max(self.q_table[tuple(next_state)]) if not done else 0)
            self.q_table[tuple(state)][action] = (1 - LEARNING_RATE) * self.q_table[tuple(state)][action] + LEARNING_RATE * target
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# Training the agent
env = DroneEnv()
agent = DQNAgent()

for episode in range(TRAIN_EPISODES):
    state = env.reset()
    state_index = (state * 9).astype(int)
    total_reward = 0
    for step in range(MAX_STEPS):
        action = agent.act(state_index)
        next_state, reward, done = env.step(action)
        next_state_index = (next_state * 9).astype(int)
        agent.remember(state_index, action, reward, next_state_index, done)
        agent.replay()
        total_reward += reward
        if done:
            break
        state_index = next_state_index
    print(f"Episode {episode + 1}: Reward = {total_reward}, Epsilon = {agent.epsilon:.2f}")

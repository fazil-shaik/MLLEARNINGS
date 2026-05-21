import numpy as np
from collections import deque
import random

class WarehouseEnvironment:
    """Simulates a real warehouse grid with obstacles and pick locations"""
    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        self.robot_pos = [0, 0]          # charging station
        self.pick_zones = [(15, 10), (5, 18), (18, 5)]
        self.obstacles = [(5,5), (6,5), (7,5), (8,5), (12,12), (13,12), (14,12)]
        self.carrying_item = False
        self.steps = 0
        
    def get_state(self):
        """Return sensor readings (simulating LiDAR + position)"""
        # Simplified: 1-hot for robot x,y + pickzone distances + obstacle proximity
        state = np.zeros(self.width * self.height * 2)
        idx = self.robot_pos[1] * self.width + self.robot_pos[0]
        state[idx] = 1  # robot position
        if self.carrying_item:
            state[self.width*self.height + idx] = 1
        return state
    
    def step(self, action):
        """action: 0=up,1=down,2=left,3=right"""
        self.steps += 1
        # Move robot
        if action == 0 and self.robot_pos[1] > 0:        # up
            self.robot_pos[1] -= 1
        elif action == 1 and self.robot_pos[1] < self.height-1:  # down
            self.robot_pos[1] += 1
        elif action == 2 and self.robot_pos[0] > 0:      # left
            self.robot_pos[0] -= 1
        elif action == 3 and self.robot_pos[0] < self.width-1:   # right
            self.robot_pos[0] += 1
        
        # Check collisions
        if tuple(self.robot_pos) in self.obstacles:
            return self.get_state(), -10, True  # collision penalty, episode ends
        
        reward = -0.05  # small step penalty (energy cost)
        
        # Pick up item
        if tuple(self.robot_pos) in self.pick_zones and not self.carrying_item:
            self.carrying_item = True
            reward += 20  # successful pickup
            print(f" Item picked at {self.robot_pos}")
        
        # Deliver to charging station (returns area)
        if self.robot_pos == [0,0] and self.carrying_item:
            reward += 50  # successful delivery
            self.carrying_item = False
            print(f" Item delivered! Total reward: {reward}")
            return self.get_state(), reward, True  # mission complete
        
        # Time limit
        done = self.steps >= 200
        return self.get_state(), reward, done

# Deep Q-Network for warehouse robot
class DQNAgent:
    def __init__(self, state_size, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.q_network = self._build_model()
    
    def _build_model(self):
        """Neural network: state -> Q-values for each action"""
        print(f"Building Q-network for {self.state_size} inputs → {self.action_size} outputs")
        return None
    
    def act(self, state):
        """Epsilon‑greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # explore
        return np.argmax(state[:4])  # dummy: use first 4 features
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=32):
        """Train on past experiences (experience replay)"""
        if len(self.memory) < batch_size:
            return
        pass

if __name__ == "__main__":
    env = WarehouseEnvironment()
    agent = DQNAgent(state_size=20*20*2)
    
    for episode in range(100):
        state = env.get_state()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
        
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        print(f"Episode {episode}: Total reward = {total_reward}")
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
    # Define action and observation spaces
        self.action_space = spaces.Discrete(2) # Example:
        # →two actions
        self.observation_space = spaces.Box(low=0, high=10)
        # →shape=(1,), dtype=np.float32)
        self.state = None # Initialize state
    def reset(self):
        """Reset␣the␣environment␣to␣an␣initial␣state."""
        self.state = np.array([5.0]) # Example initial
        # →state
        return self.state, {}
    
    def step(self, action):
        """Apply␣an␣action␣and␣return␣results."""
        reward = 1 if action == 1 else 0
        self.state = self.state + (action- 0.5)
        done = self.state[0] > 10 or self.state[0] < 0
        return self.state, reward, done, False, {}
    def render(self):
        """Render␣the␣environment␣(optional)."""
        print(f"Current␣state:␣{self.state}")
    def close(self):
        """Clean␣up␣resources␣(optional)."""
        pass
    
    
register(id="CustomEnv-v0", entry_point="__main__:CustomEnv")


#Create the environment
env = gym.make("CustomEnv-v0")
# Interact with the environment
obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample() # Random action
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        break
env.close()